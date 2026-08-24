from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import polars as pl
from scipy.optimize import OptimizeWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import DataConfig

RANDOM_STATE = 12_041_500
REPORT_RESULTS = {
    "Decision Tree": {"accuracy": 0.6514, "precision": 0.5172, "recall": 0.5269, "f1": 0.5149},
    "Logistic Regression": {
        "accuracy": 0.3306,
        "precision": 0.2597,
        "recall": 0.2367,
        "f1": 0.2103,
    },
    "MLP": {"accuracy": 0.2764, "precision": 0.2132, "recall": 0.1969, "f1": 0.1632},
    "Random Forest": {
        "accuracy": 0.7014,
        "precision": 0.5791,
        "recall": 0.5638,
        "f1": 0.5577,
    },
}
REPORT_DATASETS = {
    "Decision Tree": "decision_tree.parquet",
    "Logistic Regression": "logistic_regression.parquet",
    "MLP": "mlp.parquet",
    "Random Forest": "random_forest.parquet",
}


def _read_labels(config: DataConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    services = pl.read_csv(config.addresses_path)
    duplicated_addresses = services.group_by("address").len().filter(pl.col("len") > 1)["address"]
    services = services.filter(~pl.col("address").is_in(duplicated_addresses))

    mappings = pl.read_csv(config.mappings_path)
    if "Category" in mappings.columns:
        mappings = mappings.rename({"Category": "category"})
    mappings = mappings.filter(pl.col("category").is_not_null())
    return services, mappings


def build_baseline_dataset(
    config: DataConfig,
    *,
    top_categories: int = 10,
    use_pagerank: bool = True,
    sample_non_service: bool = True,
    with_replacement: bool = False,
    random_state: int = RANDOM_STATE,
) -> pl.DataFrame:
    """Recreate the report's classical-model dataset from the original sampling code."""

    if not config.features_path.is_file():
        raise FileNotFoundError(
            f"Address features not found: {config.features_path}. "
            "Run `defi-gnn prepare-data` first."
        )
    if use_pagerank and not config.pagerank_path.is_file():
        raise FileNotFoundError(
            f"PageRank data not found: {config.pagerank_path}. "
            "Run `defi-gnn prepare-pagerank` first."
        )

    services, mappings = _read_labels(config)
    merged_labels = services.join(mappings, on="defi_service", how="inner")
    unknown_addresses = services.join(mappings, on="defi_service", how="left").filter(
        pl.col("category").is_null()
    )
    features = pl.read_parquet(config.features_path)
    frame = features.join(services, on="address", how="inner")
    frame = frame.join(mappings, on="defi_service", how="inner")
    if use_pagerank:
        frame = frame.join(pl.read_parquet(config.pagerank_path), on="address", how="inner")
    frame = frame.filter(~pl.col("address").is_in(unknown_addresses["address"]))
    frame = frame.join(merged_labels, on="address", how="left")
    frame = frame.with_columns(
        pl.when(pl.col("category").is_null())
        .then(pl.lit("user"))
        .otherwise(pl.col("category"))
        .alias("category")
    )

    selected_categories = (
        frame.group_by("category")
        .len()
        .sort("len", descending=True)
        .head(top_categories)["category"]
    )
    frame = frame.filter(pl.col("category").is_in(selected_categories))

    if sample_non_service:
        counts = frame.group_by("category").len()
        sample_target = int(counts.select(pl.col("len").mean()).item())
        sampled_services = []
        for _, group in frame.group_by("category"):
            count = sample_target if with_replacement else min(sample_target, len(group))
            sampled_services.append(
                group.sample(n=count, with_replacement=with_replacement, seed=random_state)
            )

        user_pool = features.join(services.select("address"), on="address", how="anti")
        if use_pagerank:
            user_pool = user_pool.join(
                pl.read_parquet(config.pagerank_path), on="address", how="inner"
            )
        user_count = sample_target if with_replacement else min(sample_target, len(user_pool))
        users = user_pool.sample(
            n=user_count,
            with_replacement=with_replacement,
            seed=random_state,
        ).with_columns(pl.lit("user").alias("category"))
        sampled_services_frame = pl.concat(sampled_services)
        for column_name, column_type in sampled_services_frame.schema.items():
            if column_name not in users.columns:
                users = users.with_columns(pl.lit(None, dtype=column_type).alias(column_name))
        users = users.select(sampled_services_frame.columns)
        frame = pl.concat([sampled_services_frame, users], how="vertical")

    frame = frame.drop(
        "address",
        "",
        "defi_service",
        "_right",
        "defi_service_right",
        "category_right",
    ).drop_nans()
    return frame.sample(fraction=1.0, with_replacement=False, seed=random_state)


def _classifiers(random_state: int) -> dict[str, Callable[[], object]]:
    return {
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=100_000, random_state=random_state
        ),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(1024,),
            activation="relu",
            solver="adam",
            max_iter=25_000,
            random_state=random_state,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }


def run_baselines(
    config: DataConfig,
    output_dir: str | Path,
    *,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, object]]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    results: list[dict[str, object]] = []
    generated_dataset: pl.DataFrame | None = None
    for name, create_classifier in _classifiers(random_state).items():
        report_dataset = config.baseline_path / REPORT_DATASETS[name]
        if report_dataset.is_file():
            dataset = pl.read_parquet(report_dataset)
        else:
            if generated_dataset is None:
                generated_dataset = build_baseline_dataset(config, random_state=random_state)
            dataset = generated_dataset
        dataset.write_parquet(output_path / REPORT_DATASETS[name])

        labels = LabelEncoder().fit_transform(dataset["category"].to_list())
        features = dataset.drop("category").to_numpy()
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            scores = cross_validate(
                create_classifier(),
                features,
                labels,
                cv=folds,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
            )
        row: dict[str, object] = {"model": name}
        for metric in scoring:
            key = metric.replace("_macro", "")
            row[key] = float(scores[f"test_{metric}"].mean())
            row[f"{key}_std"] = float(scores[f"test_{metric}"].std())
        row["matches_report_4dp"] = all(
            round(float(row[metric]), 4) == expected
            for metric, expected in REPORT_RESULTS[name].items()
        )
        results.append(row)

    pd.DataFrame(results).to_csv(output_path / "metrics.csv", index=False)
    (output_path / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
