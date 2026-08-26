"""Command-line entry point for the end-to-end POC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .evaluation import CostModel, choose_threshold, evaluate_predictions
from .features import build_baseline_features, build_combined_features
from .models import fit_model, predict_scores
from .reporting import write_frame, write_json, write_report
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


def run_poc(config_path: str, output_dir: str, seed: int | None = None) -> dict:
    config = load_config(config_path, {"seed": seed} if seed is not None else None)
    out = Path(output_dir)
    dataset = generate_ecosystem(config)
    _save_dataset(dataset, out)
    labels = dataset.labels.set_index("order_id")
    split = split_by_time(dataset.orders, config.split["train"], config.split["validation"])
    split_ids = {"train": split.train.order_id, "validation": split.validation.order_id, "test": split.test.order_id}
    results = {}
    for mode in ["baseline", "graph"]:
        fs = build_baseline_features(dataset.orders, dataset.labels, config.graph["history_days"]) if mode == "baseline" else build_combined_features(dataset.orders, dataset.labels, config.graph["history_days"])
        ids = fs.X.index
        train_ids, val_ids, test_ids = [pd.Index(v) for v in [split_ids["train"], split_ids["validation"], split_ids["test"]]]
        train_x, val_x, test_x = fs.X.loc[ids.intersection(train_ids)], fs.X.loc[ids.intersection(val_ids)], fs.X.loc[ids.intersection(test_ids)]
        model = fit_model(train_x, fs.y.loc[train_x.index], config.model["backend"], config.seed,
                          {k: v for k, v in config.model.items() if k not in {"backend"}})
        val_scores = predict_scores(model, val_x)
        test_scores = predict_scores(model, test_x)
        val_loss = labels.loc[val_x.index, "loss_amount"].astype(float)
        test_loss = labels.loc[test_x.index, "loss_amount"].astype(float)
        val_eval = evaluate_predictions(fs.y.loc[val_x.index], val_scores, loss_amount=val_loss, cost=CostModel(config.costs["review_cost"], config.costs["false_positive_block_cost"]))
        threshold = choose_threshold(val_eval)
        test_eval = evaluate_predictions(fs.y.loc[test_x.index], test_scores, threshold, test_loss, CostModel(config.costs["review_cost"], config.costs["false_positive_block_cost"]))
        results[mode] = {"backend": model.backend, "threshold": threshold, "validation": val_eval.metrics, "test": test_eval.metrics,
                         "threshold_table": test_eval.threshold_table.to_dict(orient="records"), "feature_count": len(fs.X.columns)}
        write_frame(test_eval.threshold_table, out / f"{mode}_thresholds.csv")
    summary = {"dataset": {"customers": len(dataset.customers), "orders": len(dataset.orders), "returns": len(dataset.returns), "rings": len(dataset.rings), "abuse_orders": int(dataset.labels.is_abuse.sum())},
               "split": {"train_end": str(split.train_end), "validation_end": str(split.validation_end)}, "models": results,
               "limitations": ["Synthetic labels and entity relationships may not represent a production merchant.", "NetworkX is suitable for this POC scale but not an operational graph backend.", "Graph features are relationship signals, not proof of abuse."]}
    write_json(summary, out / "run_manifest.json")
    write_report(summary, out / "report.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the defensive AbuseRing synthetic POC")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-poc")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--output-dir", default="artifacts/run")
    run.add_argument("--seed", type=int)
    gen = sub.add_parser("generate")
    gen.add_argument("--config", default="configs/default.yaml")
    gen.add_argument("--output-dir", default="artifacts/run")
    args = parser.parse_args()
    if args.command == "run-poc":
        print(json.dumps(run_poc(args.config, args.output_dir, args.seed), indent=2, default=str))
    else:
        config = load_config(args.config)
        _save_dataset(generate_ecosystem(config), Path(args.output_dir))


if __name__ == "__main__":
    main()
