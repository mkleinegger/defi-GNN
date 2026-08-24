from __future__ import annotations

import copy
import json
import platform
import random
import time
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import summary as pyg_summary
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

from .config import ProjectConfig, TrainingConfig
from .data import UNKNOWN_LABEL, load_graph
from .model import GraphSAGE

EMPTY_METRICS = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


@dataclass(frozen=True)
class SplitInfo:
    selected_label_ids: list[int]
    selected_class_names: list[str]
    class_counts: dict[str, int]
    train_nodes: int
    val_nodes: int
    test_nodes: int
    sampled_unknown_nodes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Return accuracy and scikit-learn-compatible macro precision/recall/F1."""

    if targets.numel() == 0:
        return dict(EMPTY_METRICS)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    flat_index = (targets.to(torch.long) * num_classes + predictions.to(torch.long)).cpu()
    confusion = torch.bincount(flat_index, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    true_positives = confusion.diag().to(torch.float64)
    predicted_counts = confusion.sum(dim=0).to(torch.float64)
    actual_counts = confusion.sum(dim=1).to(torch.float64)

    precision = torch.where(
        predicted_counts > 0,
        true_positives / predicted_counts,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        actual_counts > 0,
        true_positives / actual_counts,
        torch.zeros_like(true_positives),
    )
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2 * precision * recall / denominator,
        torch.zeros_like(denominator),
    )
    active_classes = (actual_counts > 0) | (predicted_counts > 0)
    precision = precision[active_classes]
    recall = recall[active_classes]
    f1 = f1[active_classes]
    return {
        "accuracy": float(true_positives.sum().item() / actual_counts.sum().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }


def build_splits(data: Data, config: TrainingConfig) -> tuple[SplitInfo, torch.Tensor]:
    """Reproduce the original top-k, unknown sampling, and global random split."""

    label_to_index: dict[str, int] = data.label_to_index
    index_to_label = {index: label for label, index in label_to_index.items()}
    unknown_id = label_to_index[UNKNOWN_LABEL]
    class_counts = torch.bincount(data.y, minlength=len(label_to_index))
    non_empty_classes = int((class_counts > 0).sum())
    selected_count = min(config.top_k_classes, non_empty_classes)
    selected_label_ids = torch.topk(class_counts, k=selected_count).indices.tolist()
    selected_mask = torch.zeros_like(data.y, dtype=torch.bool)
    for label_id in selected_label_ids:
        selected_mask |= data.y == label_id

    known_indices = ((data.y != unknown_id) & selected_mask).nonzero(as_tuple=True)[0]
    unknown_indices = ((data.y == unknown_id) & selected_mask).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(config.seed)
    sampled_unknown_nodes = min(config.unknown_sample_size, int(unknown_indices.numel()))
    if sampled_unknown_nodes:
        permutation = torch.randperm(unknown_indices.numel(), generator=generator)
        unknown_indices = unknown_indices[permutation[:sampled_unknown_nodes]]
    else:
        unknown_indices = unknown_indices[:0]

    combined_indices = torch.cat((known_indices, unknown_indices))
    if combined_indices.numel() < 3:
        raise ValueError("The selected classes contain fewer than three training candidates")
    combined_indices = combined_indices[
        torch.randperm(combined_indices.numel(), generator=generator)
    ]
    train_count = int(config.train_fraction * combined_indices.numel())
    validation_count = int(config.val_fraction * combined_indices.numel())
    train_indices = combined_indices[:train_count]
    validation_end = train_count + validation_count
    val_indices = combined_indices[train_count:validation_end]
    test_indices = combined_indices[validation_end:]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    label_lookup = torch.full((len(label_to_index),), -1, dtype=torch.long)
    label_lookup[torch.tensor(selected_label_ids)] = torch.arange(len(selected_label_ids))
    selected_class_names = [index_to_label[label_id] for label_id in selected_label_ids]
    selected_counts = {
        index_to_label[label_id]: (
            sampled_unknown_nodes if label_id == unknown_id else int(class_counts[label_id])
        )
        for label_id in selected_label_ids
    }
    split_info = SplitInfo(
        selected_label_ids=selected_label_ids,
        selected_class_names=selected_class_names,
        class_counts=selected_counts,
        train_nodes=int(data.train_mask.sum()),
        val_nodes=int(data.val_mask.sum()),
        test_nodes=int(data.test_mask.sum()),
        sampled_unknown_nodes=sampled_unknown_nodes,
    )
    return split_info, label_lookup


def create_loaders(
    data: Data, config: TrainingConfig
) -> tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
    if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
        raise RuntimeError(
            "PyG NeighborLoader requires the sampling extra. "
            "Install it with `uv sync --extra sampling`."
        )
    loader_options = {
        "data": data,
        "num_neighbors": list(config.num_neighbors),
        "batch_size": config.batch_size,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using 'NeighborSampler' without a 'pyg-lib' installation is deprecated",
        )
        return (
            NeighborLoader(input_nodes=data.train_mask, **loader_options),
            NeighborLoader(input_nodes=data.val_mask, **loader_options),
            NeighborLoader(input_nodes=data.test_mask, **loader_options),
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: NeighborLoader,
    mask_attribute: str,
    device: str,
    label_lookup: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index)
        mask = getattr(batch, mask_attribute)
        target = label_lookup[batch.y[mask]]
        if bool((target < 0).any()):
            raise ValueError("A sampled evaluation node has a label outside the selected classes")
        predictions.append(output[mask].argmax(dim=1).cpu())
        targets.append(target.cpu())

    if not targets:
        return dict(EMPTY_METRICS)
    combined_predictions = torch.cat(predictions)
    combined_targets = torch.cat(targets)
    num_classes = int((label_lookup >= 0).sum())
    return classification_metrics(combined_predictions, combined_targets, num_classes)


def _print_model_summary(model: nn.Module, loader: NeighborLoader, device: str) -> None:
    try:
        batch = next(iter(loader)).to(device)
    except StopIteration:
        print("Model summary unavailable: the training loader is empty.")
        return

    was_training = model.training
    model.eval()
    try:
        print("\n=== Model summary ===")
        print(pyg_summary(model, batch.x, batch.edge_index))
        print(f"Seed nodes: {int(batch.batch_size):,}")
        print(f"Sampled nodes: {int(batch.num_nodes):,}")
        print(f"Sampled edges: {int(batch.num_edges):,}")
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(f"Trainable parameters: {trainable_parameters:,}")
    finally:
        model.train(was_training)


def _next_batch(iterator: Iterator[Data], loader: NeighborLoader) -> tuple[Data, Iterator[Data]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_model(
    model: nn.Module,
    train_loader: NeighborLoader,
    val_loader: NeighborLoader,
    label_lookup: torch.Tensor,
    config: TrainingConfig,
) -> tuple[nn.Module, list[dict[str, float | int]], dict[str, object]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    started_at = time.monotonic()
    deadline = (
        started_at + config.max_runtime_seconds if config.max_runtime_seconds is not None else None
    )
    iterator = iter(train_loader)
    history: list[dict[str, float | int]] = []
    best_validation_accuracy = -1.0
    best_validation_metrics = dict(EMPTY_METRICS)
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    completed_steps = 0
    stop_reason: str | None = None

    for step in range(1, config.max_steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = f"maximum runtime ({config.max_runtime_seconds:.1f}s) reached"
            break

        batch, iterator = _next_batch(iterator, train_loader)
        batch = batch.to(config.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.x, batch.edge_index)
        target = label_lookup[batch.y[batch.train_mask]]
        if bool((target < 0).any()):
            raise ValueError("A sampled training node has a label outside the selected classes")
        loss = F.cross_entropy(
            output[batch.train_mask],
            target,
            label_smoothing=config.label_smoothing,
        )
        loss.backward()
        optimizer.step()
        completed_steps = step

        record: dict[str, float | int] = {"step": step, "loss": float(loss.item())}
        should_evaluate = step % config.eval_every == 0 or step == config.max_steps
        if should_evaluate:
            train_metrics = evaluate(model, train_loader, "train_mask", config.device, label_lookup)
            validation_metrics = evaluate(
                model, val_loader, "val_mask", config.device, label_lookup
            )
            record.update({f"train_{name}": value for name, value in train_metrics.items()})
            record.update({f"val_{name}": value for name, value in validation_metrics.items()})
            print(
                f"step={step:05d} loss={loss.item():.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={validation_metrics['accuracy']:.4f} "
                f"val_f1={validation_metrics['f1']:.4f}"
            )
            if validation_metrics["accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation_metrics["accuracy"]
                best_validation_metrics = dict(validation_metrics)
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)

    if best_step == 0:
        best_validation_metrics = evaluate(
            model, val_loader, "val_mask", config.device, label_lookup
        )
        best_validation_accuracy = best_validation_metrics["accuracy"]
        best_step = completed_steps
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    training_summary = {
        "completed_steps": completed_steps,
        "best_step": best_step,
        "best_validation_metrics": best_validation_metrics,
        "elapsed_seconds": time.monotonic() - started_at,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
    }
    return model, history, training_summary


def run_training(config: ProjectConfig) -> dict[str, object]:
    config.validate()
    set_seed(config.training.seed)
    overall_started_at = time.monotonic()
    data = load_graph(config.data)
    split_info, label_lookup = build_splits(data, config.training)
    label_lookup = label_lookup.to(config.training.device)
    train_loader, val_loader, test_loader = create_loaders(data, config.training)

    model = GraphSAGE(
        input_channels=data.num_node_features,
        hidden_channels=config.training.hidden_channels,
        output_channels=len(split_info.selected_label_ids),
        dropout=config.training.dropout,
    ).to(config.training.device)
    if config.training.print_model_summary:
        _print_model_summary(model, train_loader, config.training.device)

    print("\n=== Training ===")
    model, history, training_summary = train_model(
        model,
        train_loader,
        val_loader,
        label_lookup,
        config.training,
    )
    test_metrics = evaluate(model, test_loader, "test_mask", config.training.device, label_lookup)

    output_dir = Path(config.training.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / config.training.history_name
    checkpoint_path = output_dir / config.training.checkpoint_name
    metrics_path = output_dir / config.training.metrics_name
    pd.DataFrame(history).to_csv(history_path, index=False)

    checkpoint_temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "project_config": config.to_dict(),
            "feature_names": data.feature_names,
            "source_label_to_index": data.label_to_index,
            "selected_label_ids": split_info.selected_label_ids,
            "selected_class_names": split_info.selected_class_names,
        },
        checkpoint_temporary,
    )
    checkpoint_temporary.replace(checkpoint_path)

    metrics: dict[str, object] = {
        "best_validation": training_summary["best_validation_metrics"],
        "best_step": training_summary["best_step"],
        "test": test_metrics,
        "graph": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": int(data.num_node_features),
        },
        "split": asdict(split_info),
        "training": training_summary,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "metrics": str(metrics_path),
        },
        "config": config.to_dict(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "total_elapsed_seconds": time.monotonic() - overall_started_at,
    }
    metrics_temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    metrics_temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_temporary.replace(metrics_path)

    print("\n=== Final metrics ===")
    final_metrics = {"best_validation": metrics["best_validation"], "test": test_metrics}
    print(json.dumps(final_metrics, indent=2))
    print(f"Artifacts: {output_dir}")
    return metrics


def run_repeated_training(
    config: ProjectConfig,
    seeds: list[int],
) -> dict[str, object]:
    """Run the report's repeated-seed protocol and aggregate its test metrics."""

    if not seeds:
        raise ValueError("At least one random seed is required")
    base_output = Path(config.training.output_dir).expanduser().resolve()
    runs = []
    for seed in seeds:
        run_config = replace(
            config,
            training=replace(
                config.training,
                seed=seed,
                output_dir=str(base_output / f"seed-{seed}"),
            ),
        )
        runs.append(run_training(run_config))

    aggregate = {
        metric: {
            "mean": float(np.mean([run["test"][metric] for run in runs])),
            "std": float(np.std([run["test"][metric] for run in runs])),
        }
        for metric in EMPTY_METRICS
    }
    summary: dict[str, object] = {
        "seeds": seeds,
        "test": aggregate,
        "runs": [run["artifacts"]["metrics"] for run in runs],
    }
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
