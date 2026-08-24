from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class DataConfig:
    """Paths and options used to build the graph dataset."""

    dataset_root: str = "data"
    trace_file: str = "traces/combined_trace.parquet"
    addresses_file: str = "tvl_addr.csv"
    mappings_file: str = "service_mappings.csv"
    baseline_dir: str = "report_baselines"
    blocks_file: str = "blocks/blocks.parquet"
    features_file: str = "processed/address_features.parquet"
    pagerank_file: str = "processed/address_pagerank.parquet"
    processed_file: str = "processed/graph.pt"
    metadata_file: str = "processed/metadata.json"
    add_node_features: bool = True

    def validate(self) -> None:
        if not self.dataset_root.strip():
            raise ValueError("data.dataset_root must not be empty")
        for field_name in (
            "trace_file",
            "addresses_file",
            "mappings_file",
            "baseline_dir",
            "blocks_file",
            "features_file",
            "pagerank_file",
            "processed_file",
            "metadata_file",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"data.{field_name} must not be empty")

    @property
    def root(self) -> Path:
        return Path(self.dataset_root).expanduser().resolve()

    def resolve(self, relative_path: str) -> Path:
        path = Path(relative_path).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def trace_path(self) -> Path:
        return self.resolve(self.trace_file)

    @property
    def addresses_path(self) -> Path:
        return self.resolve(self.addresses_file)

    @property
    def mappings_path(self) -> Path:
        return self.resolve(self.mappings_file)

    @property
    def baseline_path(self) -> Path:
        return self.resolve(self.baseline_dir)

    @property
    def blocks_path(self) -> Path:
        return self.resolve(self.blocks_file)

    @property
    def features_path(self) -> Path:
        return self.resolve(self.features_file)

    @property
    def pagerank_path(self) -> Path:
        return self.resolve(self.pagerank_file)

    @property
    def processed_path(self) -> Path:
        return self.resolve(self.processed_file)

    @property
    def metadata_path(self) -> Path:
        return self.resolve(self.metadata_file)


@dataclass(frozen=True)
class TrainingConfig:
    """Model, split, sampling, and output settings."""

    output_dir: str = "outputs/default"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 12_041_500
    hidden_channels: int = 4096
    dropout: float = 0.15
    learning_rate: float = 0.003
    weight_decay: float = 1e-6
    label_smoothing: float = 0.0
    batch_size: int = 128
    num_neighbors: tuple[int, ...] = (150, 20)
    top_k_classes: int = 10
    unknown_sample_size: int = 800
    train_fraction: float = 0.70
    val_fraction: float = 0.10
    max_steps: int = 5_000
    eval_every: int = 200
    max_runtime_seconds: float | None = 600.0
    checkpoint_name: str = "best_model.pt"
    history_name: str = "history.csv"
    metrics_name: str = "metrics.json"
    print_model_summary: bool = True

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("training.seed must be non-negative")
        if self.hidden_channels < 1:
            raise ValueError("training.hidden_channels must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("training.dropout must be in [0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("training.label_smoothing must be in [0, 1)")
        if self.batch_size < 1:
            raise ValueError("training.batch_size must be at least 1")
        if len(self.num_neighbors) != 2:
            raise ValueError(
                "training.num_neighbors must contain two fanouts for two-hop GraphSAGE"
            )
        if any(value == 0 or value < -1 for value in self.num_neighbors):
            raise ValueError("training.num_neighbors must contain positive integers or -1")
        if self.top_k_classes < 2:
            raise ValueError("training.top_k_classes must be at least 2")
        if self.unknown_sample_size < 0:
            raise ValueError("training.unknown_sample_size must be non-negative")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("training.train_fraction must be in (0, 1)")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("training.val_fraction must be in (0, 1)")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("training train_fraction + val_fraction must be less than 1")
        if self.max_steps < 1:
            raise ValueError("training.max_steps must be at least 1")
        if self.eval_every < 1:
            raise ValueError("training.eval_every must be at least 1")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0.0:
            raise ValueError("training.max_runtime_seconds must be positive or null")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(
                f"training.device is {self.device!r}, but CUDA is not available; use 'cpu'"
            )
        if not self.output_dir.strip():
            raise ValueError("training.output_dir must not be empty")
        for field_name in ("checkpoint_name", "history_name", "metrics_name"):
            value = str(getattr(self, field_name))
            if not value or Path(value).name != value:
                raise ValueError(f"training.{field_name} must be a file name, not a path")


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()

    def validate(self) -> None:
        self.data.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_dataclass[ConfigType: (DataConfig, TrainingConfig)](
    cls: type[ConfigType], values: Mapping[str, Any]
) -> ConfigType:
    valid_fields = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {cls.__name__} option(s): {joined}")

    normalized = dict(values)
    if cls is TrainingConfig and "num_neighbors" in normalized:
        raw_neighbors = normalized["num_neighbors"]
        if not isinstance(raw_neighbors, (list, tuple)):
            raise ValueError("training.num_neighbors must be a YAML list")
        normalized["num_neighbors"] = tuple(int(value) for value in raw_neighbors)
    return cls(**normalized)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load a strict YAML configuration.

    Relative paths remain relative to the current working directory so the supplied
    ``configs/default.yaml`` behaves the same from source and after installation.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The configuration root must be a mapping")
    unknown_sections = sorted(set(raw) - {"data", "training"})
    if unknown_sections:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unknown_sections)}")

    data_values = raw.get("data", {})
    training_values = raw.get("training", {})
    if not isinstance(data_values, Mapping) or not isinstance(training_values, Mapping):
        raise ValueError("The data and training sections must be mappings")

    config = ProjectConfig(
        data=_build_dataclass(DataConfig, data_values),
        training=_build_dataclass(TrainingConfig, training_values),
    )
    config.validate()
    return config
