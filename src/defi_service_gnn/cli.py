from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .baseline import run_baselines
from .config import ProjectConfig, load_project_config
from .data import load_graph, prepare_data, prepare_pagerank, read_metadata, validate_raw_inputs
from .training import run_repeated_training, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="defi-gnn",
        description="Prepare and train the DeFi service GraphSAGE classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-data", help="Validate raw input files and their schemas"
    )
    validate_parser.add_argument("--config", default=None)

    prepare_parser = subparsers.add_parser(
        "prepare-data", help="Build the processed PyTorch Geometric graph"
    )
    prepare_parser.add_argument("--config", default=None)
    prepare_parser.add_argument(
        "--force", action="store_true", help="Replace an existing processed graph"
    )

    pagerank_parser = subparsers.add_parser(
        "prepare-pagerank", help="Build the original NetworkX PageRank feature"
    )
    pagerank_parser.add_argument("--config", default=None)
    pagerank_parser.add_argument(
        "--force", action="store_true", help="Replace existing PageRank data"
    )

    inspect_parser = subparsers.add_parser("inspect-data", help="Show processed graph metadata")
    inspect_parser.add_argument("--config", default=None)
    inspect_parser.add_argument(
        "--verify-graph",
        action="store_true",
        help="Load and validate the graph tensors as well as metadata",
    )

    train_parser = subparsers.add_parser("train", help="Train and evaluate GraphSAGE")
    train_parser.add_argument("--config", default=None)
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--top-k-classes", type=int, default=None)
    train_parser.add_argument("--unknown-sample-size", type=int, default=None)
    train_parser.add_argument("--max-steps", type=int, default=None)
    train_parser.add_argument("--max-runtime-seconds", type=float, default=None)
    train_parser.add_argument(
        "--no-model-summary", action="store_true", help="Skip the sampled model summary"
    )
    train_parser.add_argument(
        "--five-seeds",
        action="store_true",
        help="Run seed, seed+1, ..., seed+4 and write an aggregate summary",
    )

    baseline_parser = subparsers.add_parser(
        "baseline", help="Reproduce the four scikit-learn baselines from the report"
    )
    baseline_parser.add_argument("--config", default=None)
    baseline_parser.add_argument("--output-dir", default="outputs/baseline")
    return parser


def _training_overrides(config: ProjectConfig, arguments: argparse.Namespace) -> ProjectConfig:
    overrides: dict[str, object] = {}
    if arguments.output_dir is not None:
        overrides["output_dir"] = arguments.output_dir
    if arguments.device is not None:
        overrides["device"] = arguments.device
    if arguments.seed is not None:
        overrides["seed"] = arguments.seed
    if arguments.top_k_classes is not None:
        overrides["top_k_classes"] = arguments.top_k_classes
    if arguments.unknown_sample_size is not None:
        overrides["unknown_sample_size"] = arguments.unknown_sample_size
    if arguments.max_steps is not None:
        overrides["max_steps"] = arguments.max_steps
    if arguments.max_runtime_seconds is not None:
        overrides["max_runtime_seconds"] = arguments.max_runtime_seconds
    if arguments.no_model_summary:
        overrides["print_model_summary"] = False
    updated = replace(config, training=replace(config.training, **overrides))
    updated.validate()
    return updated


def _run(arguments: argparse.Namespace) -> None:
    config = (
        load_project_config(Path(arguments.config))
        if arguments.config is not None
        else ProjectConfig()
    )
    if arguments.command == "validate-data":
        validate_raw_inputs(config.data)
        print("Raw data validation passed.")
        return
    if arguments.command == "prepare-data":
        summary = prepare_data(config.data, force=arguments.force)
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return
    if arguments.command == "prepare-pagerank":
        path = prepare_pagerank(config.data, force=arguments.force)
        print(f"Wrote {path}")
        return
    if arguments.command == "inspect-data":
        metadata = read_metadata(config.data)
        if arguments.verify_graph:
            graph = load_graph(config.data)
            metadata["verified_tensor_shapes"] = {
                "x": list(graph.x.shape),
                "edge_index": list(graph.edge_index.shape),
                "y": list(graph.y.shape),
            }
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    if arguments.command == "train":
        training_config = _training_overrides(config, arguments)
        if arguments.five_seeds:
            first_seed = training_config.training.seed
            run_repeated_training(training_config, list(range(first_seed, first_seed + 5)))
        else:
            run_training(training_config)
        return
    if arguments.command == "baseline":
        results = run_baselines(config.data, arguments.output_dir)
        print(json.dumps(results, indent=2))
        return
    raise AssertionError(f"Unhandled command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        _run(arguments)
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
