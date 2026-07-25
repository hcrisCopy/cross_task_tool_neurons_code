from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pp_common import (
    PP_STAGE_VERSION,
    flatten_probe_predictions,
    grouped_classification_metrics,
    pp_subdir,
    probe_prefill_root,
    read_json,
    read_jsonl,
    should_skip,
    stable_sha256,
    subset_values,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProbePrefill stage 2: train a logistic probe on CTD activations.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--reg", type=float, default=10000.0, help="L2 regularization lambda; sklearn C=1/reg.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold only for reporting classification metrics.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_dir(features_root: Path, model_alias: str, subset: str) -> Path:
    return features_root / model_alias / subset


def probe_dir(probes_root: Path, model_alias: str, subset: str) -> Path:
    return probes_root / model_alias / subset


def load_feature_split(features_root: Path, model_alias: str, subset: str, split: str) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    base = feature_dir(features_root, model_alias, subset)
    payload_path = base / f"{split}_features.pt"
    meta_path = base / f"{split}_meta.jsonl"
    summary_path = base / f"{split}_summary.json"
    if not payload_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing ProbePrefill features for {subset}/{split}: {base}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    features = payload["features"].float()
    labels = payload["labels"].cpu().numpy().astype(np.int64)
    meta = read_jsonl(meta_path)
    summary = read_json(summary_path) if summary_path.exists() else {}
    return features, labels, meta, summary


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    train_summary: dict[str, Any],
    test_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "pp_02_train_ctd_probe",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "subset": subset,
        "reg": args.reg,
        "C": 1.0 / args.reg,
        "max_iter": args.max_iter,
        "threshold": args.threshold,
        "feature_train_summary": train_summary,
        "feature_test_summary": test_summary,
    }


def fit_probe(X_train: torch.Tensor, y_train: np.ndarray, *, c_value: float, max_iter: int) -> tuple[StandardScaler, LogisticRegression]:
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training labels contain only one class; cannot train a binary probe")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.numpy())
    clf = LogisticRegression(C=c_value, solver="lbfgs", max_iter=max_iter, random_state=42)
    clf.fit(X_scaled, y_train)
    return scaler, clf


def predict_probs(scaler: StandardScaler, clf: LogisticRegression, X: torch.Tensor) -> np.ndarray:
    scaled = scaler.transform(X.numpy())
    return clf.predict_proba(scaled)[:, 1]


def coefficients_rows(coef: np.ndarray, neuron_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for weight, neuron in zip(coef.tolist(), neuron_rows):
        row = {
            "coef": float(weight),
            "abs_coef": abs(float(weight)),
            "layer": int(neuron["layer"]),
            "module": str(neuron["module"]),
            "index": int(neuron["index"]),
            "rank": neuron.get("rank", ""),
            "score_min": neuron.get("score_min", ""),
            "score_mean": neuron.get("score_mean", ""),
        }
        rows.append(row)
    rows.sort(key=lambda row: (-row["abs_coef"], row["layer"], row["module"], row["index"]))
    for rank, row in enumerate(rows, start=1):
        row["coef_rank"] = rank
    return rows


def train_subset(args: argparse.Namespace, *, subset: str, features_root: Path, probes_root: Path) -> dict[str, Any]:
    X_train, y_train, meta_train, train_summary = load_feature_split(features_root, args.model_alias, subset, "train")
    X_test, y_test, meta_test, test_summary = load_feature_split(features_root, args.model_alias, subset, "test")
    feature_payload = torch.load(feature_dir(features_root, args.model_alias, subset) / "train_features.pt", map_location="cpu", weights_only=False)
    neuron_rows = feature_payload["neuron_rows"]
    out_dir = probe_dir(probes_root, args.model_alias, subset)
    params = expected_params(args, subset=subset, train_summary=train_summary, test_summary=test_summary)
    expected = [out_dir / "probe_no_reasoning.pt", out_dir / "probe_results_no_reasoning.json", out_dir / "test_predictions.jsonl"]
    if should_skip(out_dir, params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=probes_root):
        return read_json(out_dir / "probe_results_no_reasoning.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    c_value = 1.0 / args.reg
    scaler, clf = fit_probe(X_train, y_train, c_value=c_value, max_iter=args.max_iter)
    train_prob = predict_probs(scaler, clf, X_train)
    test_prob = predict_probs(scaler, clf, X_test)
    train_metrics = grouped_classification_metrics(y_train, train_prob, meta_train, threshold=args.threshold)
    test_metrics = grouped_classification_metrics(y_test, test_prob, meta_test, threshold=args.threshold)
    results = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": "CTD logistic probe",
        "feature_dim": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "reg": args.reg,
        "C": c_value,
        "threshold": args.threshold,
        "train": train_metrics,
        "test": test_metrics,
        "test_auroc": test_metrics["overall"]["auroc"],
        "test_accuracy": test_metrics["overall"]["accuracy"],
        "selected_by": "single fixed CTD feature set; no test-set layer search",
    }
    torch.save(
        {
            "coef": torch.from_numpy(clf.coef_[0]).float(),
            "intercept": float(clf.intercept_[0]),
            "scaler_mean": torch.from_numpy(scaler.mean_).float(),
            "scaler_scale": torch.from_numpy(scaler.scale_).float(),
            "C": c_value,
            "reg": args.reg,
            "mode": "no_reasoning",
            "feature_set": "CTD",
            "feature_dim": int(X_train.shape[1]),
            "neuron_rows": neuron_rows,
            "train_feature_sha256": stable_sha256(train_summary),
        },
        out_dir / "probe_no_reasoning.pt",
    )
    write_json(out_dir / "probe_results_no_reasoning.json", results)
    write_jsonl(out_dir / "train_predictions.jsonl", flatten_probe_predictions(meta_train, train_prob, threshold=args.threshold))
    write_jsonl(out_dir / "test_predictions.jsonl", flatten_probe_predictions(meta_test, test_prob, threshold=args.threshold))
    write_csv(out_dir / "probe_coefficients.csv", coefficients_rows(clf.coef_[0], neuron_rows))
    write_json(out_dir / "manifest.json", {"params": params, "summary": results["test"]["overall"]})
    print(
        f"{subset}: CTD probe AUROC={results['test']['overall']['auroc']} "
        f"Acc={results['test']['overall']['accuracy']:.4f} dim={X_train.shape[1]}"
    )
    return results


def main() -> None:
    args = parse_args()
    if args.reg <= 0:
        raise ValueError("--reg must be positive")
    root = probe_prefill_root(args.output_root)
    features_root = pp_subdir(root, "features")
    probes_root = pp_subdir(root, "probes")
    root_manifest: dict[str, Any] = {
        "stage": "pp_02_train_ctd_probe",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    for subset in subset_values(args.subset):
        root_manifest["subsets"][subset] = train_subset(
            args,
            subset=subset,
            features_root=features_root,
            probes_root=probes_root,
        )
    manifest_path = probes_root / args.model_alias / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote ProbePrefill probe manifest: {manifest_path}")


if __name__ == "__main__":
    main()
