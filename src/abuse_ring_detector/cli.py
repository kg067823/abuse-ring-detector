"""Command-line entry point for the end-to-end POC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .dataset_quality import compute_dataset_quality
from .evaluation import CostModel, choose_threshold, evaluate_predictions
from .features import (
    build_baseline_features,
    build_combined_features,
    build_extended_features,
    build_full_features,
    build_two_hop_extended_features,
)
from .models import fit_model, predict_scores
from .reporting import write_frame, write_json, write_report
from .ring_evaluation import (
    compare_models,
    evaluate_rings,
    save_ring_evaluation_artifacts,
)
from .splits import split_by_time
from .synthetic import generate_ecosystem


def _save_dataset(dataset, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ["customers", "orders", "returns", "labels", "ground_truth", "rings", "ring_memberships"]:
        write_frame(getattr(dataset, name), out / f"{name}.csv.gz")
    write_json(dataset.metadata, out / "metadata.json")


def _load_dataset(out: Path):
    from .schemas import SyntheticDataset
    frames = {name: pd.read_csv(out / f"{name}.csv.gz") for name in ["customers", "orders", "returns", "labels", "ground_truth", "rings", "ring_memberships"]}
    frames["orders"]["event_time"] = pd.to_datetime(frames["orders"]["event_time"])
    return SyntheticDataset(**frames, metadata=json.loads((out / "metadata.json").read_text()))


def _save_quality_artifacts(quality_report, out: Path) -> dict:
    q_dir = out / "dataset_quality"
    q_dir.mkdir(parents=True, exist_ok=True)
    write_frame(quality_report.split_summary, q_dir / "split_summary.csv")
    write_frame(quality_report.rings_by_type, q_dir / "rings_by_type.csv")
    q_json = {
        "metrics": quality_report.metrics,
        "ring_statistics": quality_report.ring_statistics,
        "entity_overlap": quality_report.entity_overlap,
        "split_summary": quality_report.split_summary.to_dict(orient="records"),
        "rings_by_type": quality_report.rings_by_type.to_dict(orient="records"),
    }
    write_json(q_json, q_dir / "dataset_quality.json")
    return q_json


def _execute_pipeline(dataset, config, out: Path) -> dict:
    labels = dataset.labels.set_index("order_id")
    split = split_by_time(dataset.orders, config.split["train"], config.split["validation"])
    split_ids = {"train": split.train.order_id, "validation": split.validation.order_id, "test": split.test.order_id}
    results = {}
    test_eval_map = {}
    test_scores_map = {}
    ring_results_map = {}

    quality_report = compute_dataset_quality(dataset, split)
    quality_json = _save_quality_artifacts(quality_report, out)

    modes = ["baseline", "graph", "graph_temporal", "graph_temporal_custrel", "graph_temporal_custrel_2hop"]
    cost = CostModel(config.costs["review_cost"], config.costs["false_positive_block_cost"])

    for mode in modes:
        if mode == "baseline":
            fs = build_baseline_features(dataset.orders, dataset.labels, config.graph["history_days"])
        elif mode == "graph":
            fs = build_combined_features(dataset.orders, dataset.labels, config.graph["history_days"])
        elif mode == "graph_temporal":
            fs = build_full_features(dataset.orders, dataset.labels, config.graph["history_days"])
        elif mode == "graph_temporal_custrel":
            fs = build_extended_features(dataset.orders, dataset.labels, config.graph["history_days"])
        else:
            fs = build_two_hop_extended_features(dataset.orders, dataset.labels, config.graph["history_days"])

        ids = fs.X.index
        train_ids, val_ids, test_ids = [pd.Index(v) for v in [split_ids["train"], split_ids["validation"], split_ids["test"]]]
        train_x, val_x, test_x = fs.X.loc[ids.intersection(train_ids)], fs.X.loc[ids.intersection(val_ids)], fs.X.loc[ids.intersection(test_ids)]
        model = fit_model(train_x, fs.y.loc[train_x.index], config.model["backend"], config.seed,
                          {k: v for k, v in config.model.items() if k not in {"backend"}})
        val_scores = predict_scores(model, val_x)
        test_scores = predict_scores(model, test_x)
        test_scores_map[mode] = test_scores
        val_loss = labels.loc[val_x.index, "loss_amount"].astype(float)
        test_loss = labels.loc[test_x.index, "loss_amount"].astype(float)
        val_eval = evaluate_predictions(fs.y.loc[val_x.index], val_scores, loss_amount=val_loss, cost=cost)
        threshold = choose_threshold(val_eval)
        test_eval = evaluate_predictions(fs.y.loc[test_x.index], test_scores, threshold, test_loss, cost)
        test_eval_map[mode] = test_eval
        results[mode] = {"backend": model.backend, "threshold": threshold, "validation": val_eval.metrics, "test": test_eval.metrics,
                         "threshold_table": test_eval.threshold_table.to_dict(orient="records"), "feature_count": len(fs.X.columns)}
        write_frame(test_eval.threshold_table, out / f"{mode}_thresholds.csv")

        ring_res = evaluate_rings(split.test, dataset.labels, test_scores, threshold)
        ring_results_map[mode] = ring_res

    comparison_df = compare_models(ring_results_map, test_eval_map)
    save_ring_evaluation_artifacts(out / "ring_evaluation", ring_results_map, comparison_df)

    ring_eval_summary = {
        "summary": comparison_df.to_dict(orient="records"),
    }
    for mode in modes:
        ring_eval_summary[mode] = {
            "metrics": ring_results_map[mode].metrics,
            "by_ring_type": ring_results_map[mode].by_ring_type.to_dict(orient="records"),
            "top_k": ring_results_map[mode].top_k.to_dict(orient="records"),
        }

    summary = {
        "dataset": {
            "customers": len(dataset.customers),
            "orders": len(dataset.orders),
            "returns": len(dataset.returns),
            "rings": len(dataset.rings),
            "abuse_orders": int(dataset.labels.is_abuse.sum()),
        },
        "split": {
            "train_end": str(split.train_end),
            "validation_end": str(split.validation_end),
        },
        "models": results,
        "dataset_quality": quality_json,
        "ring_evaluation": ring_eval_summary,
        "limitations": [
            "Synthetic labels and entity relationships may not represent a production merchant.",
            "NetworkX is suitable for this POC scale but not an operational graph backend.",
            "Graph and temporal velocity features are relationship signals, not proof of abuse.",
        ],
    }
    write_json(summary, out / "run_manifest.json")
    write_report(summary, out / "report.md")
    return summary


def evaluate_saved_run(output_dir: str, config_path: str = "configs/default.yaml") -> dict:
    out = Path(output_dir)
    config = load_config(config_path)
    dataset = _load_dataset(out)
    return _execute_pipeline(dataset, config, out)


def run_poc(config_path: str, output_dir: str, seed: int | None = None) -> dict:
    config = load_config(config_path, {"seed": seed} if seed is not None else None)
    out = Path(output_dir)
    dataset = generate_ecosystem(config)
    _save_dataset(dataset, out)
    return _execute_pipeline(dataset, config, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the defensive AbuseRing synthetic POC")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-poc")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--output-dir", default="artifacts/run")
    run.add_argument("--seed", type=int)
    eval_cmd = sub.add_parser("evaluate-rings")
    eval_cmd.add_argument("--config", default="configs/default.yaml")
    eval_cmd.add_argument("--output-dir", default="artifacts/full-run")
    gen = sub.add_parser("generate")
    gen.add_argument("--config", default="configs/default.yaml")
    gen.add_argument("--output-dir", default="artifacts/run")
    args = parser.parse_args()
    if args.command == "run-poc":
        print(json.dumps(run_poc(args.config, args.output_dir, args.seed), indent=2, default=str))
    elif args.command == "evaluate-rings":
        print(json.dumps(evaluate_saved_run(args.output_dir, args.config), indent=2, default=str))
    else:
        config = load_config(args.config)
        _save_dataset(generate_ecosystem(config), Path(args.output_dir))


if __name__ == "__main__":
    main()
