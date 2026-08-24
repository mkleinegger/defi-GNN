from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from .config import DataConfig

DATASET_FORMAT_VERSION = 2
UNKNOWN_LABEL = "unknown"

BASELINE_FEATURES = (
    "in_degree",
    "tx_count_in",
    "eth_received",
    "tx_min_in",
    "tx_max_in",
    "tx_mean_in",
    "active_block_span",
    "avg_inter_tx_block_gap",
    "out_degree",
    "tx_count_out",
    "eth_sent",
    "tx_min_out",
    "tx_max_out",
    "tx_mean_out",
    "gas_used",
    "distinct_method_count",
    "tx_total",
    "active_days",
    "fTX",
)
GNN_FEATURES = tuple(
    feature for feature in BASELINE_FEATURES if feature not in {"in_degree", "out_degree"}
)

TRACE_COLUMNS = {
    "from_address",
    "to_address",
    "value",
    "gas",
    "input",
    "block_id",
}
ADDRESS_COLUMNS = {"address", "defi_service"}
MAPPING_COLUMNS = {"defi_service", "category"}
BLOCK_COLUMNS = {"number", "timestamp"}


@dataclass(frozen=True)
class BuildSummary:
    dataset_format_version: int
    graph_path: str
    metadata_path: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    known_nodes: int
    unknown_nodes: int
    duplicated_label_addresses: int
    duplicate_node_rows: int
    add_node_features: bool
    source_files: dict[str, dict[str, str | int]]
    feature_names: list[str]
    label_to_index: dict[str, int]


def _require_columns(actual: Iterable[str], required: set[str], source: Path) -> None:
    missing = sorted(required - set(actual))
    if missing:
        raise ValueError(f"{source} is missing required column(s): {', '.join(missing)}")


def validate_raw_inputs(config: DataConfig) -> None:
    """Fail early with actionable input and schema errors."""

    config.validate()
    paths = (
        config.trace_path,
        config.addresses_path,
        config.mappings_path,
        config.blocks_path,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n- " + "\n- ".join(missing))
    output_paths = (
        config.features_path,
        config.pagerank_path,
        config.processed_path,
        config.metadata_path,
    )
    if len(set(output_paths)) != len(output_paths) or any(
        output_path in paths for output_path in output_paths
    ):
        raise ValueError("Generated data paths must be distinct from each other and all raw inputs")

    _require_columns(
        pl.scan_parquet(config.trace_path).collect_schema().names(),
        TRACE_COLUMNS,
        config.trace_path,
    )
    _require_columns(
        pl.scan_csv(config.addresses_path).collect_schema().names(),
        ADDRESS_COLUMNS,
        config.addresses_path,
    )
    _require_columns(
        pl.scan_csv(config.mappings_path).collect_schema().names(),
        MAPPING_COLUMNS,
        config.mappings_path,
    )
    _require_columns(
        pl.scan_parquet(config.blocks_path).collect_schema().names(),
        BLOCK_COLUMNS,
        config.blocks_path,
    )


def _normalized_address(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strip_chars().str.to_lowercase()


def _load_service_labels(config: DataConfig) -> tuple[pl.DataFrame, int]:
    addresses = (
        pl.read_csv(config.addresses_path)
        .select(
            _normalized_address("address").alias("address"),
            pl.col("defi_service").cast(pl.String).str.strip_chars(),
        )
        .filter(
            pl.col("address").is_not_null()
            & pl.col("defi_service").is_not_null()
            & (pl.col("address") != "")
            & (pl.col("defi_service") != "")
        )
    )
    mappings = pl.read_csv(config.mappings_path).select(
        pl.col("defi_service").cast(pl.String).str.strip_chars(),
        pl.col("category").cast(pl.String).str.strip_chars().replace("", None),
    )
    conflicting_services = (
        mappings.group_by("defi_service")
        .agg(pl.col("category").drop_nulls().n_unique().alias("category_count"))
        .filter(pl.col("category_count") > 1)
    )
    if not conflicting_services.is_empty():
        examples = ", ".join(conflicting_services["defi_service"].head(5).to_list())
        raise ValueError(f"Service mappings contain conflicting categories for: {examples}")
    labeled = addresses.join(mappings, on="defi_service", how="inner").select("address", "category")
    duplicated_count = labeled.group_by("address").len().filter(pl.col("len") > 1).height
    return labeled, duplicated_count


def _numeric_columns(frame: pl.DataFrame, *, exclude: set[str]) -> list[str]:
    numeric_types = {
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.Int16,
        pl.Int8,
        pl.UInt64,
        pl.UInt32,
        pl.UInt16,
        pl.UInt8,
        pl.Boolean,
    }
    return [
        name
        for name, dtype in frame.schema.items()
        if name not in exclude and dtype in numeric_types
    ]


def _compute_address_features(traces: pl.LazyFrame) -> pl.DataFrame:
    in_metrics = (
        traces.group_by("to_address")
        .agg(
            pl.col("from_address").n_unique().alias("in_degree"),
            pl.len().alias("tx_count_in"),
            pl.col("eth_value").sum().alias("eth_received"),
            pl.col("eth_value").min().alias("tx_min_in"),
            pl.col("eth_value").max().alias("tx_max_in"),
            pl.col("eth_value").mean().alias("tx_mean_in"),
            (pl.col("timestamp").max() - pl.col("timestamp").min()).alias("active_block_span"),
            pl.when(pl.len() > 1)
            .then(
                (pl.col("timestamp").max() - pl.col("timestamp").min())
                / (pl.len().cast(pl.Float64) - 1)
            )
            .otherwise(0.0)
            .alias("avg_inter_tx_block_gap"),
        )
        .rename({"to_address": "address"})
    )
    out_metrics = (
        traces.group_by("from_address")
        .agg(
            pl.col("to_address").n_unique().alias("out_degree"),
            pl.len().alias("tx_count_out"),
            pl.col("eth_value").sum().alias("eth_sent"),
            pl.col("eth_value").min().alias("tx_min_out"),
            pl.col("eth_value").max().alias("tx_max_out"),
            pl.col("eth_value").mean().alias("tx_mean_out"),
            pl.col("gas").mean().alias("gas_used"),
        )
        .rename({"from_address": "address"})
    )
    method_metrics = (
        traces.filter(pl.col("input").is_not_null() & (pl.col("input") != "0x"))
        .with_columns(pl.col("input").str.slice(0, 10).alias("method_signature"))
        .group_by("from_address")
        .agg(pl.col("method_signature").n_unique().alias("distinct_method_count"))
        .rename({"from_address": "address"})
    )
    all_transactions = pl.concat(
        [
            traces.select(pl.col("to_address").alias("address"), "timestamp"),
            traces.select(pl.col("from_address").alias("address"), "timestamp"),
        ],
        how="vertical",
    )
    all_metrics = (
        all_transactions.group_by("address")
        .agg(
            pl.len().alias("tx_total"),
            ((pl.col("timestamp").max() - pl.col("timestamp").min()) / 86_400 + 1.0).alias(
                "active_days"
            ),
        )
        .with_columns((pl.col("tx_total") / pl.col("active_days")).alias("fTX"))
    )
    return (
        in_metrics.join(out_metrics, on="address", how="inner")
        .join(method_metrics, on="address", how="inner")
        .join(all_metrics, on="address", how="left")
        .fill_null(0)
        .collect()
    )


def _build_graph(config: DataConfig) -> tuple[Data, BuildSummary, pl.DataFrame]:
    service_labels, duplicated_label_addresses = _load_service_labels(config)

    traces = (
        pl.scan_parquet(config.trace_path)
        .select(
            _normalized_address("from_address").alias("from_address"),
            _normalized_address("to_address").alias("to_address"),
            pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0).alias("value"),
            pl.col("gas").cast(pl.Float64, strict=False).fill_null(0.0).alias("gas"),
            pl.col("input").cast(pl.String),
            pl.col("block_id").cast(pl.Int64),
        )
        .with_columns(
            (pl.col("value") / 1e18).alias("eth_value"),
        )
    )
    blocks = (
        pl.scan_parquet(config.blocks_path)
        .select(
            pl.col("number").cast(pl.Int64).alias("block_id"),
            pl.col("timestamp").cast(pl.Int64),
        )
        .unique(subset=["block_id"])
    )
    traces = traces.join(blocks, on="block_id", how="inner")

    address_features = (
        pl.read_parquet(config.features_path)
        if config.features_path.is_file()
        else _compute_address_features(traces)
    )

    edges = traces.select("from_address", "to_address").collect()
    if edges.is_empty():
        raise ValueError("No trace rows matched the supplied blocks; the graph would be empty")

    nodes = (
        pl.concat(
            [
                edges.select(pl.col("from_address").alias("address")),
                edges.select(pl.col("to_address").alias("address")),
            ],
            how="vertical",
        )
        .unique()
        .sort("address")
        .join(service_labels, on="address", how="left")
        .with_columns(pl.col("category").fill_null(UNKNOWN_LABEL))
        .with_row_index("node_id")
    )
    nodes = nodes.join(address_features, on="address", how="left")

    base_numeric_columns = _numeric_columns(nodes, exclude={"address", "category", "node_id"})
    nodes = nodes.with_columns(
        [
            pl.col(column).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            for column in base_numeric_columns
        ]
    )
    nodes = nodes.sort("node_id")

    node_lookup = nodes.select("address", "node_id")
    indexed_edges = (
        edges.join(node_lookup, left_on="from_address", right_on="address", how="inner")
        .rename({"node_id": "source"})
        .join(node_lookup, left_on="to_address", right_on="address", how="inner")
        .rename({"node_id": "target"})
    )
    edge_index = torch.from_numpy(
        np.vstack(
            [
                indexed_edges["source"].to_numpy(),
                indexed_edges["target"].to_numpy(),
            ]
        ).astype(np.int64, copy=False)
    )

    feature_names: list[str]
    if config.add_node_features:
        feature_names = list(GNN_FEATURES)
        features = np.nan_to_num(
            nodes.select(feature_names).to_numpy().astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        x = torch.from_numpy(features)
    else:
        feature_names = ["constant"]
        x = torch.zeros((nodes.height, 1), dtype=torch.float32)

    labels = nodes["category"].to_list()
    unique_labels = sorted(set(labels) | {UNKNOWN_LABEL})
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    y = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, y=y)
    graph.label_to_index = label_to_index
    graph.feature_names = feature_names
    graph.dataset_format_version = DATASET_FORMAT_VERSION

    known_nodes = sum(label != UNKNOWN_LABEL for label in labels)
    summary = BuildSummary(
        dataset_format_version=DATASET_FORMAT_VERSION,
        graph_path=str(config.processed_path),
        metadata_path=str(config.metadata_path),
        num_nodes=int(graph.num_nodes),
        num_edges=int(graph.num_edges),
        num_features=int(graph.num_node_features),
        num_classes=len(label_to_index),
        known_nodes=known_nodes,
        unknown_nodes=len(labels) - known_nodes,
        duplicated_label_addresses=duplicated_label_addresses,
        duplicate_node_rows=nodes.height - nodes["address"].n_unique(),
        add_node_features=config.add_node_features,
        source_files={
            name: {"path": str(path), "size_bytes": path.stat().st_size}
            for name, path in {
                "traces": config.trace_path,
                "addresses": config.addresses_path,
                "mappings": config.mappings_path,
                "blocks": config.blocks_path,
            }.items()
        },
        feature_names=feature_names,
        label_to_index=label_to_index,
    )
    return graph, summary, address_features


def prepare_data(config: DataConfig, *, force: bool = False) -> BuildSummary:
    """Build and atomically persist a processed PyG graph."""

    validate_raw_inputs(config)
    if config.processed_path.exists() and not force:
        raise FileExistsError(
            f"Processed graph already exists: {config.processed_path}. Pass --force to rebuild it."
        )

    graph, summary, address_features = _build_graph(config)
    config.processed_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    config.features_path.parent.mkdir(parents=True, exist_ok=True)

    graph_temporary = config.processed_path.with_suffix(config.processed_path.suffix + ".tmp")
    metadata_temporary = config.metadata_path.with_suffix(config.metadata_path.suffix + ".tmp")
    features_temporary = config.features_path.with_suffix(config.features_path.suffix + ".tmp")
    torch.save(graph, graph_temporary)
    address_features.write_parquet(features_temporary)
    metadata_temporary.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    graph_temporary.replace(config.processed_path)
    features_temporary.replace(config.features_path)
    metadata_temporary.replace(config.metadata_path)
    return summary


def prepare_pagerank(config: DataConfig, *, force: bool = False) -> Path:
    """Compute the original NetworkX PageRank feature used by the baselines."""

    import networkx as nx

    validate_raw_inputs(config)
    if config.pagerank_path.exists() and not force:
        raise FileExistsError(
            f"PageRank data already exists: {config.pagerank_path}. Pass --force to rebuild it."
        )

    edges = (
        pl.scan_parquet(config.trace_path)
        .select("from_address", "to_address")
        .filter(pl.col("from_address").is_not_null() & pl.col("to_address").is_not_null())
        .collect()
    )
    graph = nx.DiGraph()
    graph.add_edges_from(zip(edges["from_address"], edges["to_address"], strict=True))
    scores = nx.pagerank(graph, alpha=0.85, tol=1e-6, max_iter=100)

    config.pagerank_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.pagerank_path.with_suffix(config.pagerank_path.suffix + ".tmp")
    pl.DataFrame({"address": list(scores), "pagerank": list(scores.values())}).write_parquet(
        temporary
    )
    temporary.replace(config.pagerank_path)
    return config.pagerank_path


def load_graph(config: DataConfig) -> Data:
    """Load a processed graph and verify the public data contract."""

    if not config.processed_path.is_file():
        raise FileNotFoundError(
            f"Processed graph not found: {config.processed_path}. "
            "Run `defi-gnn prepare-data --config <config>` first."
        )
    graph = torch.load(
        config.processed_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(graph, Data):
        raise TypeError(f"{config.processed_path} does not contain a PyG Data object")
    if graph.x is None or graph.y is None or graph.edge_index is None:
        raise ValueError("Processed graph must contain x, y, and edge_index tensors")
    if graph.x.ndim != 2 or graph.y.ndim != 1 or graph.edge_index.shape[0] != 2:
        raise ValueError("Processed graph tensors have invalid dimensions")
    if graph.x.shape[0] != graph.y.shape[0]:
        raise ValueError("Processed graph has different node counts in x and y")
    if not hasattr(graph, "label_to_index") or UNKNOWN_LABEL not in graph.label_to_index:
        raise ValueError("Processed graph is missing a label_to_index mapping with 'unknown'")
    return graph


def read_metadata(config: DataConfig) -> dict[str, object]:
    if not config.metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {config.metadata_path}")
    return json.loads(config.metadata_path.read_text(encoding="utf-8"))
