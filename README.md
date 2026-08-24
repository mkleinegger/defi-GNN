# DeFi Service GNN

GraphSAGE classification of DeFi services from Ethereum transaction traces. The
repository contains the original experiment logic in a compact package: data
generation, the PyG model training path, and the four classical baselines from
the [project report](docs/report.pdf).

## Setup

The reproduced environment uses Python 3.12, PyTorch 2.6.0, PyG 2.6.1, and the
CUDA 12.4 `torch-sparse` backend used by PyG's `NeighborLoader`.

```bash
uv sync --extra dev --extra sampling
```

The `sampling` extra is the pinned Linux/Python 3.12/CUDA dependency. There is no
custom sampling fallback: training always uses PyG's `NeighborLoader`.

## Data

The default paths are defined in `configs/default.yaml`:

```text
data/
├── blocks/blocks.parquet
├── traces/combined_trace.parquet
├── service_mappings.csv
├── tvl_addr.csv
└── processed/
    ├── address_features.parquet
    ├── address_pagerank.parquet
    ├── graph.pt
    └── metadata.json
```

The large trace, block, and processed files are intentionally not tracked by
Git. The address labels, category mapping, and compact report baseline datasets
are tracked.

### Recreating the raw inputs

The historical experiment uses Ethereum mainnet blocks `20,600,000` through
`20,899,999`, inclusive. The current raw artifacts contain 300,000 block rows
and 302,730,797 call-trace rows. The report separately records 47,220,325
transactions and 10,120,062 distinct addresses in this interval.

There is one important reproducibility boundary: the exact trace snapshot was
provided by the Digital Currency Ecosystems group at the Complexity Science
Hub. The original repository did not contain the upstream trace extractor or a
versioned DeFiLlama API snapshot. The checked-in labels and the locally retained
Parquet files reproduce the historical experiment; a fresh extraction can be
schema-compatible, but is not guaranteed to be byte-identical. In particular,
different Ethereum clients can return slightly different trace records and the
live DeFiLlama metadata can change.

The preprocessing code needs only these columns:

| File | Required columns | Meaning |
| --- | --- | --- |
| `traces/combined_trace.parquet` | `from_address`, `to_address`, `value`, `gas`, `input`, `block_id` | One execution trace per row; addresses define a directed edge and `value` is in wei |
| `blocks/blocks.parquet` | `number`, `timestamp` | Ethereum block number and Unix timestamp |
| `tvl_addr.csv` | `address`, `defi_service` | Ethereum address to DeFi protocol slug |
| `service_mappings.csv` | `defi_service`, `category` | Protocol slug to classification label |

Extra columns are allowed and ignored. Addresses are stripped and converted to
lowercase by the project, and numeric trace values are cast during loading.

#### 1. Export traces and blocks

An independent reconstruction requires an archival Ethereum RPC exposing
Parity-style tracing. A normal endpoint that only exposes the `eth_*` methods
is not sufficient for historical internal call traces. One compatible option is
[Ethereum ETL](https://github.com/blockchain-etl/ethereum-etl), whose
[trace schema](https://github.com/blockchain-etl/ethereum-etl/blob/develop/docs/schema.md#tracescsv)
contains all fields needed here.

The following illustrates a full-range export. In practice, export smaller,
non-overlapping block ranges and retain the range in every filename so failed
chunks can be retried safely.

```bash
export DEFI_GNN_RPC_URL=http://127.0.0.1:8545
mkdir -p data/source/traces data/source/blocks

uvx --python 3.9 --with 'setuptools<81' \
  --from 'ethereum-etl==1.11.2' ethereumetl export_traces \
  --start-block 20600000 \
  --end-block 20899999 \
  --batch-size 100 \
  --provider-uri "$DEFI_GNN_RPC_URL" \
  --output data/source/traces/traces_20600000_20899999.csv

uvx --python 3.9 --with 'setuptools<81' \
  --from 'ethereum-etl==1.11.2' ethereumetl export_blocks_and_transactions \
  --start-block 20600000 \
  --end-block 20899999 \
  --batch-size 100 \
  --provider-uri "$DEFI_GNN_RPC_URL" \
  --blocks-output data/source/blocks/blocks_20600000_20899999.csv
```

Ethereum ETL calls the trace block field `block_number`; this project calls it
`block_id`. Convert and combine one or more CSV or compressed CSV chunks with
the project's existing Polars dependency:

```bash
uv run python - <<'PY'
from pathlib import Path

import polars as pl

Path("data/traces").mkdir(parents=True, exist_ok=True)
Path("data/blocks").mkdir(parents=True, exist_ok=True)

traces = pl.scan_csv("data/source/traces/*.csv*", infer_schema=False).select(
    pl.col("from_address"),
    pl.col("to_address"),
    pl.col("value"),
    pl.col("gas").cast(pl.Float64, strict=False),
    pl.col("input"),
    pl.col("block_number").cast(pl.Int64).alias("block_id"),
)
traces.sink_parquet("data/traces/combined_trace.parquet")

blocks = (
    pl.scan_csv("data/source/blocks/*.csv*", infer_schema=False)
    .select(
        pl.col("number").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Int64),
    )
    .unique(subset=["number"])
    .sort("number")
    .collect()
)
blocks.write_parquet("data/blocks/blocks.parquet")
PY
```

Only the six trace columns and two block columns used by this project are kept.
This is equivalent for feature and graph construction and substantially smaller
than retaining every field returned by the exporter. Do not mix chunks from
overlapping ranges unless duplicate trace rows are removed first.

#### 2. Recreate the labels

The historical label snapshot is already preserved in `data/tvl_addr.csv` and
`data/service_mappings.csv`; use these files for report reproduction. To build a
new snapshot, collect Ethereum contract addresses and their protocol slugs from
a dated DeFiLlama export or a pinned revision of its adapters, then write one
`address,defi_service` row per association. Retrieve the corresponding protocol
category from `https://api.llama.fi/protocol/{defi_service}` and write
`defi_service,category` rows.

Save the raw responses or source revision alongside any new dataset. The live
API is mutable, some services have no category, and one address may belong to
multiple services. The graph builder deliberately preserves those multiple
label rows to match the original implementation.

#### 3. Build features and graph data

Use fresh output paths whenever the trace range or labels change. In particular,
`prepare-data --force` replaces the graph but intentionally reuses an existing
`address_features.parquet`; do not reuse that cache with different raw inputs.
The output paths can be changed in a copy of `configs/default.yaml`.

Generate the 19 original address features and the PyG graph with:

```bash
uv run defi-gnn validate-data --config configs/default.yaml
uv run defi-gnn prepare-data --config configs/default.yaml
```

PageRank is generated separately because it is needed only when rebuilding the
classical baseline datasets; it is excluded from the GNN node features:

```bash
uv run defi-gnn prepare-pagerank --config configs/default.yaml
```

Finally, verify the generated tensors and recorded source metadata:

```bash
uv run defi-gnn inspect-data \
  --config configs/default.yaml \
  --verify-graph
```

The graph retains the original many-to-many label join so its structure matches
the historical artifact: 10,120,111 node rows and 302,771,232 edges. Metadata
records the resulting 49 duplicate node rows. Its feature matrix contains the 17
reported non-graph features, excluding in-degree, out-degree, and PageRank.

## Baselines

Run the report's Decision Tree, Logistic Regression, MLP, and Random Forest
experiments with five-fold cross-validation:

```bash
uv run defi-gnn baseline \
  --config configs/default.yaml \
  --output-dir outputs/baseline
```

The small final datasets from the four original runs are kept in
`data/report_baselines/`. Their row order is preserved because it determines the
shuffled cross-validation folds. All four regenerated metric rows match Table 4
to four decimal places.

## GraphSAGE

The executable original implementation has two `SAGEConv` layers with mean
aggregation, BatchNorm, ReLU, and dropout. This matches its two-hop `(150, 20)`
`NeighborLoader`. The report text calls it a three-layer model, but no matching
three-layer implementation or three-hop fanout was preserved.

Run one Top-10 experiment:

```bash
uv run defi-gnn train \
  --config configs/default.yaml \
  --device cuda:0 \
  --output-dir outputs/gnn-top10
```

Run the five report seeds (`12041500` through `12041504`) and aggregate them:

```bash
uv run defi-gnn train \
  --config configs/default.yaml \
  --device cuda:0 \
  --output-dir outputs/gnn-top10 \
  --five-seeds
```

Use all 42 labels with `--top-k-classes 42`. Each run writes a checkpoint,
training history, complete configuration, software versions, split information,
and macro accuracy/precision/recall/F1 metrics. Repeated runs additionally write
`summary.json`.

## Verification

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

The tests exercise data generation, the real PyG neighbor loader, model training,
configuration validation, and artifact creation.

## License

[MIT](LICENSE)
