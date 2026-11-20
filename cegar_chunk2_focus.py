"""
Chunk 2: focused comparison of coordinate descent vs hybrid warm-start.

Inference-time only. No retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_NLTK_DATA = REPO_ROOT / "nltk_data"
os.environ["NLTK_DATA"] = str(LOCAL_NLTK_DATA)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import gensim.downloader as api

import nltk

if str(LOCAL_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(LOCAL_NLTK_DATA))
nltk.download("wordnet", quiet=True)

from cegar_chunk1_fastpass import (
    TOP_FRACTION,
    VIOLATION_THRESHOLD,
    copy_boxes,
    ensure_dirs,
    format_hparams,
    forward_violation,
    mean_box_displacement,
    mean_box_size,
    project_valid_box,
    repair_pair,
    run_coord_descent,
    run_hybrid,
    sample_constraints,
)
from inference_cegar import measure_galois_consistency_gap
from inference_cegar_verb import (
    build_verb_concept_set,
    build_verb_isa_graph,
    encode_verb_concepts,
)
from src.box_lattice import box_midpoint, box_volume
from src.improved_trainer import GaloisModel
from eval_hyperlex_embeddings import BOX_DIM, HIDDEN_DIM


SEEDS = [42, 43]
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
DELTA = 0.75
EPSILON = 0.75
ZETA = 1.0
ETA = 1.0
MEANINGFUL_REL_IMPROVEMENT = 0.01
MEANINGFUL_ABS_IMPROVEMENT = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results_cegar_chunk2")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--n_concepts", type=int, default=5000)
    parser.add_argument("--max_pairs", type=int, default=1200)
    parser.add_argument("--gap_sample", type=int, default=250)
    parser.add_argument("--neighbor_sample", type=int, default=250)
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--only_method", default="all", choices=["all", "coord", "hybrid"])
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warm_start_iters", type=int, default=None)
    parser.add_argument("--screen_iterations", type=int, default=10)
    return parser.parse_args()


def load_cached_model(seed: int) -> GaloisModel:
    weights_path = Path("results/inference_cegar_model.npz")
    if not weights_path.exists():
        raise FileNotFoundError("Missing results/inference_cegar_model.npz. Run inference_cegar.py first.")
    data = np.load(weights_path, allow_pickle=True)
    model = GaloisModel(300, HIDDEN_DIM, BOX_DIM, np.random.default_rng(seed))
    for key in model.params_dict():
        setattr(model, key, data[key])
    return model


def compute_pair_stats(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    threshold: float = VIOLATION_THRESHOLD,
) -> dict[str, object]:
    pair_rows = []
    hop_details: dict[str, dict[str, int | float]] = {}
    original_violations = 0
    fixed_count = 0
    new_violations = 0
    final_violations = 0

    for (hypo, hyper), hop in constraints.items():
        l0_i, u0_i = boxes_before[hypo]
        l0_j, u0_j = boxes_before[hyper]
        l1_i, u1_i = boxes_after[hypo]
        l1_j, u1_j = boxes_after[hyper]
        vf_before = forward_violation(l0_i, u0_i, l0_j, u0_j)
        vf_after = forward_violation(l1_i, u1_i, l1_j, u1_j)
        before_flag = vf_before > threshold
        after_flag = vf_after > threshold
        hop_bucket = str(hop) if hop <= 3 else "4+"
        hop_state = hop_details.setdefault(hop_bucket, {
            "original_violations": 0,
            "fixed_count": 0,
            "new_violations": 0,
            "final_violations": 0,
        })
        if before_flag:
            original_violations += 1
            hop_state["original_violations"] += 1
        if after_flag:
            final_violations += 1
            hop_state["final_violations"] += 1
        if before_flag and not after_flag:
            fixed_count += 1
            hop_state["fixed_count"] += 1
        if not before_flag and after_flag:
            new_violations += 1
            hop_state["new_violations"] += 1
        pair_rows.append({
            "hyponym": hypo,
            "hypernym": hyper,
            "hop_bucket": hop_bucket,
            "hop": hop,
            "was_violated": int(before_flag),
            "is_violated": int(after_flag),
        })

    total_pairs = len(pair_rows)
    for hop_bucket, state in hop_details.items():
        state["fix_rate"] = state["fixed_count"] / max(state["original_violations"], 1)
        state["net_improvement"] = state["fixed_count"] - state["new_violations"]

    return {
        "pair_rows": pair_rows,
        "original_violations": original_violations,
        "fixed_count": fixed_count,
        "new_violations": new_violations,
        "net_improvement": fixed_count - new_violations,
        "final_violations": final_violations,
        "final_violation_rate": final_violations / max(total_pairs, 1),
        "fix_rate": fixed_count / max(original_violations, 1),
        "total_pairs": total_pairs,
        "hop_details": hop_details,
    }


def mean_box_volume_change(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    changes = []
    for key in boxes_before:
        l0, u0 = boxes_before[key]
        l1, u1 = boxes_after[key]
        vol0 = float(np.asarray(box_volume(l0[None, :], u0[None, :])).reshape(-1)[0])
        vol1 = float(np.asarray(box_volume(l1[None, :], u1[None, :])).reshape(-1)[0])
        changes.append(abs(vol1 - vol0))
    return float(np.mean(changes)) if changes else 0.0


def neighborhood_preservation(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
    sample_size: int,
    k: int,
    rng: np.random.Generator,
) -> float:
    keys = list(boxes_before.keys())
    if len(keys) <= 2:
        return 1.0
    sample_size = min(sample_size, len(keys))
    selected = rng.choice(keys, size=sample_size, replace=False)
    mids_before = np.stack([np.asarray(box_midpoint(*boxes_before[key])).reshape(-1) for key in selected])
    mids_after = np.stack([np.asarray(box_midpoint(*boxes_after[key])).reshape(-1) for key in selected])

    def cosine_matrix(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        normalized = x / norms
        return normalized @ normalized.T

    sim_before = cosine_matrix(mids_before)
    sim_after = cosine_matrix(mids_after)
    overlaps = []
    for index in range(sample_size):
        before_neighbors = np.argsort(sim_before[index])[::-1][1:k + 1]
        after_neighbors = np.argsort(sim_after[index])[::-1][1:k + 1]
        overlap = len(set(before_neighbors) & set(after_neighbors)) / max(k, 1)
        overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 0.0


def asymmetry_preservation(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
) -> float:
    before_scores = []
    after_scores = []
    for hypo, hyper in constraints.keys():
        l0_i, u0_i = boxes_before[hypo]
        l0_j, u0_j = boxes_before[hyper]
        l1_i, u1_i = boxes_after[hypo]
        l1_j, u1_j = boxes_after[hyper]
        before_scores.append(
            float(
                (np.maximum(0.0, l0_i - l0_j).sum() + np.maximum(0.0, u0_j - u0_i).sum())
                - forward_violation(l0_i, u0_i, l0_j, u0_j)
            )
        )
        after_scores.append(
            float(
                (np.maximum(0.0, l1_i - l1_j).sum() + np.maximum(0.0, u1_j - u1_i).sum())
                - forward_violation(l1_i, u1_i, l1_j, u1_j)
            )
        )
    if len(before_scores) < 2:
        return 1.0
    rho = np.corrcoef(before_scores, after_scores)[0, 1]
    if np.isnan(rho):
        return 0.5
    return float((rho + 1.0) / 2.0)


def hop_weighted_improvement(hop_details: dict[str, dict[str, int | float]]) -> float:
    weights = {"1": 0.1, "2": 0.2, "3": 0.3, "4+": 0.4}
    score = 0.0
    for bucket, weight in weights.items():
        state = hop_details.get(bucket, None)
        if not state:
            continue
        score += weight * (state["net_improvement"] / max(state["original_violations"], 1))
    return float(score)


def evaluate_run(
    method: str,
    hyperparameters: dict[str, object],
    seed: int,
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    embeddings: dict[str, np.ndarray],
    model: GaloisModel,
    gap_pairs: list[tuple[str, str]],
    base_gap: float,
    box_size_scale: float,
    rng: np.random.Generator,
    neighbor_sample: int,
    neighbor_k: int,
) -> dict[str, object]:
    pair_stats = compute_pair_stats(boxes_before, boxes_after, constraints)
    gap_after = measure_galois_consistency_gap(boxes_after, embeddings, model, gap_pairs, n_sample=len(gap_pairs))
    gap_change = float(gap_after["mean"] - base_gap)
    displacement = mean_box_displacement(boxes_before, boxes_after)
    volume_change = mean_box_volume_change(boxes_before, boxes_after)
    neighborhood = neighborhood_preservation(boxes_before, boxes_after, neighbor_sample, neighbor_k, rng)
    asymmetry = asymmetry_preservation(boxes_before, boxes_after, constraints)
    preservation = 0.5 * neighborhood + 0.5 * asymmetry
    new_violation_rate = pair_stats["new_violations"] / max(pair_stats["total_pairs"], 1)
    net_norm = pair_stats["net_improvement"] / max(pair_stats["original_violations"], 1)
    gap_increase_norm = max(gap_change, 0.0) / max(abs(base_gap), 1e-8)
    displacement_norm = displacement / max(box_size_scale, 1e-8)
    hop_improvement = hop_weighted_improvement(pair_stats["hop_details"])
    aggregate = (
        ALPHA * pair_stats["fix_rate"]
        + BETA * net_norm
        - GAMMA * new_violation_rate
        - DELTA * gap_increase_norm
        - EPSILON * displacement_norm
        + ZETA * hop_improvement
        + ETA * preservation
    )
    return {
        "method": method,
        "hyperparameters": hyperparameters,
        "seed": seed,
        "fixed_count": pair_stats["fixed_count"],
        "fix_rate": pair_stats["fix_rate"],
        "new_violations": pair_stats["new_violations"],
        "net_improvement": pair_stats["net_improvement"],
        "final_violation_rate": pair_stats["final_violation_rate"],
        "Galois_gap_change": gap_change,
        "mean_box_displacement": displacement,
        "mean_box_volume_change": volume_change,
        "preservation_metric": preservation,
        "neighborhood_preservation": neighborhood,
        "asymmetry_preservation": asymmetry,
        "hop_weighted_improvement": hop_improvement,
        "aggregate_score": aggregate,
        "original_violations": pair_stats["original_violations"],
        "pair_rows": pair_stats["pair_rows"],
        "hop_details": pair_stats["hop_details"],
    }


def config_key(method: str, hyperparameters: dict[str, object]) -> tuple[str, tuple[tuple[str, object], ...]]:
    return method, tuple(sorted(hyperparameters.items()))


def matching_row(rows: list[dict[str, object]], seed: int) -> dict[str, object]:
    for row in rows:
        if int(row["seed"]) == seed:
            return row
    raise KeyError(f"Missing row for seed {seed}")


def bootstrap_confidence_intervals(pair_rows: list[dict[str, object]], n_boot: int = 500) -> dict[str, tuple[float, float]]:
    if not pair_rows:
        return {}
    rng = np.random.default_rng(123)
    n = len(pair_rows)
    fix_rates = []
    nets = []
    final_rates = []
    rows = np.array(pair_rows, dtype=object)
    for _ in range(n_boot):
        sample = rows[rng.integers(0, n, size=n)]
        original = sum(int(row["was_violated"]) for row in sample)
        fixed = sum(1 for row in sample if int(row["was_violated"]) == 1 and int(row["is_violated"]) == 0)
        new = sum(1 for row in sample if int(row["was_violated"]) == 0 and int(row["is_violated"]) == 1)
        final_v = sum(int(row["is_violated"]) for row in sample)
        fix_rates.append(fixed / max(original, 1))
        nets.append(fixed - new)
        final_rates.append(final_v / max(n, 1))
    return {
        "fix_rate_ci": (float(np.percentile(fix_rates, 2.5)), float(np.percentile(fix_rates, 97.5))),
        "net_improvement_ci": (float(np.percentile(nets, 2.5)), float(np.percentile(nets, 97.5))),
        "final_violation_rate_ci": (float(np.percentile(final_rates, 2.5)), float(np.percentile(final_rates, 97.5))),
    }


def format_ci(label: str, ci_map: dict[str, tuple[float, float]] | None, key: str) -> str:
    if not ci_map or key not in ci_map:
        return f"- {label}: n/a"
    low, high = ci_map[key]
    return f"- {label}: [{low:.4f}, {high:.4f}]"


def meaningful_lower(value: float, baseline: float) -> bool:
    margin = max(MEANINGFUL_ABS_IMPROVEMENT, abs(baseline) * MEANINGFUL_REL_IMPROVEMENT)
    return value < baseline - margin


def meaningful_higher(value: float, baseline: float) -> bool:
    margin = max(MEANINGFUL_ABS_IMPROVEMENT, abs(baseline) * MEANINGFUL_REL_IMPROVEMENT)
    return value > baseline + margin


def write_main_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method",
            "hyperparameters",
            "seed",
            "fixed_count",
            "fix_rate",
            "new_violations",
            "net_improvement",
            "final_violation_rate",
            "Galois_gap_change",
            "mean_box_displacement",
            "mean_box_volume_change",
            "preservation_metric",
            "hop_weighted_improvement",
            "aggregate_score",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": row["method"],
                "hyperparameters": format_hparams(row["hyperparameters"]),
                "seed": row["seed"],
                "fixed_count": row["fixed_count"],
                "fix_rate": row["fix_rate"],
                "new_violations": row["new_violations"],
                "net_improvement": row["net_improvement"],
                "final_violation_rate": row["final_violation_rate"],
                "Galois_gap_change": row["Galois_gap_change"],
                "mean_box_displacement": row["mean_box_displacement"],
                "mean_box_volume_change": row["mean_box_volume_change"],
                "preservation_metric": row["preservation_metric"],
                "hop_weighted_improvement": row["hop_weighted_improvement"],
                "aggregate_score": row["aggregate_score"],
            })


def write_hop_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method",
            "hyperparameters",
            "seed",
            "hop_bucket",
            "original_violations",
            "fixed_count",
            "fix_rate",
            "new_violations",
            "net_improvement",
        ])
        writer.writeheader()
        for row in rows:
            for hop_bucket, details in row["hop_details"].items():
                writer.writerow({
                    "method": row["method"],
                    "hyperparameters": format_hparams(row["hyperparameters"]),
                    "seed": row["seed"],
                    "hop_bucket": hop_bucket,
                    "original_violations": details["original_violations"],
                    "fixed_count": details["fixed_count"],
                    "fix_rate": details["fix_rate"],
                    "new_violations": details["new_violations"],
                    "net_improvement": details["net_improvement"],
                })


def plot_scatter(path: Path, rows: list[dict[str, object]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 6))
    for row in rows:
        plt.scatter(row[x_key], row[y_key], s=60)
        plt.text(row[x_key], row[y_key], f"{row['method']}:{row['seed']}", fontsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_convergence(path: Path, histories: dict[str, list[dict[str, object]]]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 6))
    for label, history in histories.items():
        xs = [item["iteration"] for item in history]
        ys = [item["final_violation_rate"] for item in history]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Final violation rate")
    plt.title("Best configuration convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hop_grouped_bar(path: Path, coord_row: dict[str, object], hybrid_row: dict[str, object]) -> None:
    if plt is None:
        return
    buckets = ["1", "2", "3", "4+"]
    coord_vals = [coord_row["hop_details"].get(bucket, {}).get("fix_rate", 0.0) for bucket in buckets]
    hybrid_vals = [hybrid_row["hop_details"].get(bucket, {}).get("fix_rate", 0.0) for bucket in buckets]
    x = np.arange(len(buckets))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, coord_vals, width, label="Coordinate descent")
    plt.bar(x + width / 2, hybrid_vals, width, label="Hybrid")
    plt.xticks(x, buckets)
    plt.ylabel("Fix rate")
    plt.title("Hop-stratified fix rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def mean_hop_details(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    buckets = ["1", "2", "3", "4+"]
    metrics = ["original_violations", "fixed_count", "fix_rate", "new_violations", "net_improvement"]
    averaged = {}
    for bucket in buckets:
        averaged[bucket] = {}
        for metric in metrics:
            values = [float(row["hop_details"].get(bucket, {}).get(metric, 0.0)) for row in rows]
            averaged[bucket][metric] = float(np.mean(values)) if values else 0.0
    return averaged


def plot_pareto(path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 6))
    for row in rows:
        plt.scatter(row["mean_box_displacement"], row["fix_rate"], s=60)
        plt.text(row["mean_box_displacement"], row["fix_rate"], f"{row['method']}:{row['seed']}", fontsize=7)
    plt.xlabel("Mean box displacement")
    plt.ylabel("Fix rate")
    plt.title("Repair-preservation Pareto view")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def record_history(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_current: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    embeddings: dict[str, np.ndarray],
    model: GaloisModel,
    gap_pairs: list[tuple[str, str]],
    base_gap: float,
) -> dict[str, object]:
    stats = compute_pair_stats(boxes_before, boxes_current, constraints)
    gap_after = measure_galois_consistency_gap(boxes_current, embeddings, model, gap_pairs, n_sample=len(gap_pairs))
    return {
        "fixed_count": stats["fixed_count"],
        "new_violations": stats["new_violations"],
        "net_improvement": stats["net_improvement"],
        "final_violation_rate": stats["final_violation_rate"],
        "galois_gap_change": float(gap_after["mean"] - base_gap),
        "mean_box_displacement": mean_box_displacement(boxes_before, boxes_current),
    }


def run_coord_with_history(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    embeddings: dict[str, np.ndarray],
    model: GaloisModel,
    gap_pairs: list[tuple[str, str]],
    base_gap: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    history = []
    for iteration in range(1, iterations + 1):
        current_boxes, _ = run_coord_descent(boxes, constraints, 1, step_size, False, embeddings=embeddings, model=model, gap_pairs=gap_pairs, box_size_scale=1.0)
        boxes = current_boxes
        snapshot = record_history(boxes_before, boxes, constraints, embeddings, model, gap_pairs, base_gap)
        snapshot["iteration"] = iteration
        history.append(snapshot)
    return boxes, history


def run_hybrid_with_history(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    warm_start_iters: int,
    embeddings: dict[str, np.ndarray],
    model: GaloisModel,
    gap_pairs: list[tuple[str, str]],
    base_gap: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    history = []
    persistence: dict[tuple[str, str], int] = {}

    for iteration in range(1, iterations + 1):
        current_violated = []
        for (hypo, hyper), hop in constraints.items():
            l_i, u_i = boxes[hypo]
            l_j, u_j = boxes[hyper]
            vf = forward_violation(l_i, u_i, l_j, u_j)
            if vf > VIOLATION_THRESHOLD:
                current_violated.append((hypo, hyper, vf))
        if not current_violated:
            snapshot = record_history(boxes_before, boxes, constraints, embeddings, model, gap_pairs, base_gap)
            snapshot["iteration"] = iteration
            history.append(snapshot)
            continue

        if iteration <= warm_start_iters:
            for hypo, hyper, _ in current_violated:
                repair_pair(boxes, hypo, hyper, step_size)
        else:
            keys = {(hypo, hyper) for hypo, hyper, _ in current_violated}
            for key in keys:
                persistence[key] = persistence.get(key, 0) + 1
            scored = []
            for hypo, hyper, vf in current_violated:
                severity = vf + persistence.get((hypo, hyper), 0)
                scored.append((severity, hypo, hyper))
            scored.sort(reverse=True)
            top_n = max(1, int(np.ceil(len(scored) * TOP_FRACTION)))
            for _, hypo, hyper in scored[:top_n]:
                repair_pair(boxes, hypo, hyper, step_size)

        snapshot = record_history(boxes_before, boxes, constraints, embeddings, model, gap_pairs, base_gap)
        snapshot["iteration"] = iteration
        history.append(snapshot)
    return boxes, history


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    output_paths = ensure_dirs(Path(args.output_dir))

    print(f"\n{'=' * 88}")
    print("  Chunk 2: Coordinate Descent vs Hybrid Warm-Start")
    print(f"{'=' * 88}")
    print(f"  Seeds: {seeds}")
    print(f"  Output dir: {output_paths['root']}")

    if args.only_method == "coord":
        if args.step_size is None or args.iterations is None:
            raise ValueError("--only_method coord requires --step_size and --iterations.")
        coord_configs = [{"step_size": args.step_size, "iterations": args.iterations}]
        hybrid_configs = []
    elif args.only_method == "hybrid":
        if args.step_size is None or args.iterations is None or args.warm_start_iters is None:
            raise ValueError("--only_method hybrid requires --step_size, --iterations, and --warm_start_iters.")
        coord_configs = []
        hybrid_configs = [{
            "step_size": args.step_size,
            "iterations": args.iterations,
            "warm_start_iters": args.warm_start_iters,
        }]
    else:
        coord_configs = [{"step_size": step_size, "iterations": iterations}
                         for step_size in [0.2, 0.3, 0.4]
                         for iterations in [15, 20, 25]]
        hybrid_configs = [{"step_size": step_size, "iterations": iterations, "warm_start_iters": warm}
                          for step_size in [0.2, 0.3, 0.4]
                          for iterations in [15, 20, 25]
                          for warm in [3, 5, 7]]

    glove = api.load("glove-wiki-gigaword-300")
    all_rows: list[dict[str, object]] = []
    screen_by_config: dict[tuple[str, tuple[tuple[str, object], ...]], list[dict[str, object]]] = defaultdict(list)
    per_seed_context = {}

    print("\n[1/5] Building per-seed repair settings...")
    for seed in seeds:
        model = load_cached_model(seed)
        concept_set = build_verb_concept_set(
            set(glove.key_to_index.keys()),
            n_concepts=args.n_concepts,
            min_depth=2,
            max_depth=8,
            rng=np.random.default_rng(seed),
        )
        concept_lemmas = [item[1] for item in concept_set]
        boxes, embeddings = encode_verb_concepts(concept_lemmas, model, glove)
        _, all_constraints = build_verb_isa_graph(concept_set, boxes)
        constraints = sample_constraints(all_constraints, args.max_pairs, np.random.default_rng(seed + 1000))
        gap_pairs = list(constraints.keys())[: min(args.gap_sample, len(constraints))]
        base_gap = measure_galois_consistency_gap(boxes, embeddings, model, gap_pairs, n_sample=len(gap_pairs))["mean"]
        box_size_scale = mean_box_size(boxes)
        per_seed_context[seed] = {
            "model": model,
            "boxes": boxes,
            "embeddings": embeddings,
            "constraints": constraints,
            "gap_pairs": gap_pairs,
            "base_gap": base_gap,
            "box_size_scale": box_size_scale,
        }
        base_stats = compute_pair_stats(boxes, boxes, constraints)
        print(f"  Seed {seed}: constraints={len(constraints)} original_violations={base_stats['original_violations']}")

    print(f"\n[2/5] Screening all configs for {args.screen_iterations} iterations...")
    for seed in seeds:
        context = per_seed_context[seed]
        boxes = context["boxes"]
        constraints = context["constraints"]
        for cfg in coord_configs:
            boxes_after, _ = run_coord_descent(boxes, constraints, int(args.screen_iterations), float(cfg["step_size"]), False,
                                               embeddings=context["embeddings"], model=context["model"],
                                               gap_pairs=context["gap_pairs"], box_size_scale=context["box_size_scale"])
            row = evaluate_run("Coordinate descent", {**cfg, "actual_iterations": int(args.screen_iterations), "screen_only": True}, seed,
                               boxes, boxes_after, constraints, context["embeddings"], context["model"],
                               context["gap_pairs"], context["base_gap"], context["box_size_scale"],
                               np.random.default_rng(seed + 2000), args.neighbor_sample, args.neighbor_k)
            screen_by_config[config_key("Coordinate descent", cfg)].append(row)

        for cfg in hybrid_configs:
            boxes_after, _ = run_hybrid(boxes, constraints, int(args.screen_iterations), float(cfg["step_size"]), int(cfg["warm_start_iters"]),
                                        1.0, 0.0, TOP_FRACTION, False,
                                        embeddings=context["embeddings"], model=context["model"],
                                        gap_pairs=context["gap_pairs"], box_size_scale=context["box_size_scale"])
            row = evaluate_run("Hybrid warm-start", {**cfg, "actual_iterations": int(args.screen_iterations), "screen_only": True}, seed,
                               boxes, boxes_after, constraints, context["embeddings"], context["model"],
                               context["gap_pairs"], context["base_gap"], context["box_size_scale"],
                               np.random.default_rng(seed + 3000), args.neighbor_sample, args.neighbor_k)
            screen_by_config[config_key("Hybrid warm-start", cfg)].append(row)

    ranked_coord = sorted(
        [(
            key,
            np.mean([row["aggregate_score"] for row in rows]),
        ) for key, rows in screen_by_config.items() if key[0] == "Coordinate descent"],
        key=lambda item: item[1],
        reverse=True,
    )
    ranked_hybrid = sorted(
        [(
            key,
            np.mean([row["aggregate_score"] for row in rows]),
        ) for key, rows in screen_by_config.items() if key[0] == "Hybrid warm-start"],
        key=lambda item: item[1],
        reverse=True,
    )
    selected_coord = [dict(key[1]) for key, _ in ranked_coord[:2]] if ranked_coord else []
    selected_hybrid = [dict(key[1]) for key, _ in ranked_hybrid[:3]] if ranked_hybrid else []

    print("\n[3/5] Full evaluation of selected configs...")
    for seed in seeds:
        context = per_seed_context[seed]
        boxes = context["boxes"]
        constraints = context["constraints"]
        for cfg in coord_configs:
            if cfg in selected_coord:
                boxes_after, _ = run_coord_descent(boxes, constraints, int(cfg["iterations"]), float(cfg["step_size"]), False,
                                                   embeddings=context["embeddings"], model=context["model"],
                                                   gap_pairs=context["gap_pairs"], box_size_scale=context["box_size_scale"])
                row = evaluate_run("Coordinate descent", cfg, seed, boxes, boxes_after, constraints,
                                   context["embeddings"], context["model"], context["gap_pairs"], context["base_gap"],
                                   context["box_size_scale"], np.random.default_rng(seed + 4000), args.neighbor_sample, args.neighbor_k)
            else:
                row = matching_row(screen_by_config[config_key("Coordinate descent", cfg)], seed)
            all_rows.append(row)

        for cfg in hybrid_configs:
            if cfg in selected_hybrid:
                boxes_after, _ = run_hybrid(boxes, constraints, int(cfg["iterations"]), float(cfg["step_size"]),
                                            int(cfg["warm_start_iters"]), 1.0, 0.0, TOP_FRACTION, False,
                                            embeddings=context["embeddings"], model=context["model"],
                                            gap_pairs=context["gap_pairs"], box_size_scale=context["box_size_scale"])
                row = evaluate_run("Hybrid warm-start", cfg, seed, boxes, boxes_after, constraints,
                                   context["embeddings"], context["model"], context["gap_pairs"], context["base_gap"],
                                   context["box_size_scale"], np.random.default_rng(seed + 5000), args.neighbor_sample, args.neighbor_k)
            else:
                row = matching_row(screen_by_config[config_key("Hybrid warm-start", cfg)], seed)
            all_rows.append(row)

    aggregated: dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, object]] = {}
    for row in all_rows:
        base_hparams = {key: value for key, value in row["hyperparameters"].items() if key not in {"actual_iterations", "screen_only"}}
        key = config_key(row["method"], base_hparams)
        bucket = aggregated.setdefault(key, {"method": row["method"], "hyperparameters": base_hparams, "rows": []})
        bucket["rows"].append(row)

    for bucket in aggregated.values():
        rows = bucket["rows"]
        bucket["mean_aggregate_score"] = float(np.mean([row["aggregate_score"] for row in rows]))
        bucket["mean_fix_rate"] = float(np.mean([row["fix_rate"] for row in rows]))
        bucket["mean_net_improvement"] = float(np.mean([row["net_improvement"] for row in rows]))
        bucket["mean_final_violation_rate"] = float(np.mean([row["final_violation_rate"] for row in rows]))
        bucket["mean_Galois_gap_change"] = float(np.mean([row["Galois_gap_change"] for row in rows]))
        bucket["mean_box_displacement"] = float(np.mean([row["mean_box_displacement"] for row in rows]))
        bucket["mean_box_volume_change"] = float(np.mean([row["mean_box_volume_change"] for row in rows]))
        bucket["mean_preservation_metric"] = float(np.mean([row["preservation_metric"] for row in rows]))
        bucket["mean_hop_weighted_improvement"] = float(np.mean([row["hop_weighted_improvement"] for row in rows]))

    coord_buckets = [bucket for bucket in aggregated.values() if bucket["method"] == "Coordinate descent"]
    hybrid_buckets = [bucket for bucket in aggregated.values() if bucket["method"] == "Hybrid warm-start"]
    best_coord = max(coord_buckets, key=lambda bucket: bucket["mean_aggregate_score"]) if coord_buckets else None
    best_hybrid = max(hybrid_buckets, key=lambda bucket: bucket["mean_aggregate_score"]) if hybrid_buckets else None
    best_overall = max(aggregated.values(), key=lambda bucket: bucket["mean_aggregate_score"])

    if best_coord is not None:
        coord_pair_rows = [pair for row in best_coord["rows"] for pair in row["pair_rows"]]
        best_coord["bootstrap"] = bootstrap_confidence_intervals(coord_pair_rows)
    if best_hybrid is not None:
        hybrid_pair_rows = [pair for row in best_hybrid["rows"] for pair in row["pair_rows"]]
        best_hybrid["bootstrap"] = bootstrap_confidence_intervals(hybrid_pair_rows)

    print("\n[4/5] Writing tables, logs, and plots...")
    write_main_table(output_paths["root"] / "main_table.csv", all_rows)
    write_hop_table(output_paths["root"] / "hop_table.csv", all_rows)

    with (output_paths["root"] / "top_configs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "category",
            "method",
            "hyperparameters",
            "mean_aggregate_score",
            "mean_fix_rate",
            "mean_net_improvement",
            "mean_final_violation_rate",
            "mean_Galois_gap_change",
            "mean_box_displacement",
            "mean_box_volume_change",
            "mean_preservation_metric",
            "mean_hop_weighted_improvement",
        ])
        writer.writeheader()
        top_rows = [("best_overall", best_overall)]
        if best_coord is not None:
            top_rows.insert(0, ("best_coordinate_descent", best_coord))
        if best_hybrid is not None:
            insert_at = 1 if best_coord is not None else 0
            top_rows.insert(insert_at, ("best_hybrid", best_hybrid))
        for category, bucket in top_rows:
            writer.writerow({
                "category": category,
                "method": bucket["method"],
                "hyperparameters": format_hparams(bucket["hyperparameters"]),
                "mean_aggregate_score": bucket["mean_aggregate_score"],
                "mean_fix_rate": bucket["mean_fix_rate"],
                "mean_net_improvement": bucket["mean_net_improvement"],
                "mean_final_violation_rate": bucket["mean_final_violation_rate"],
                "mean_Galois_gap_change": bucket["mean_Galois_gap_change"],
                "mean_box_displacement": bucket["mean_box_displacement"],
                "mean_box_volume_change": bucket["mean_box_volume_change"],
                "mean_preservation_metric": bucket["mean_preservation_metric"],
                "mean_hop_weighted_improvement": bucket["mean_hop_weighted_improvement"],
            })

    convergence_histories = {}
    for bucket in [item for item in [best_coord, best_hybrid] if item is not None]:
        for row in bucket["rows"]:
            seed = row["seed"]
            context = per_seed_context[seed]
            if bucket["method"] == "Coordinate descent":
                _, history = run_coord_with_history(
                    context["boxes"], context["constraints"], int(bucket["hyperparameters"]["iterations"]),
                    float(bucket["hyperparameters"]["step_size"]), context["embeddings"], context["model"],
                    context["gap_pairs"], context["base_gap"])
            else:
                _, history = run_hybrid_with_history(
                    context["boxes"], context["constraints"], int(bucket["hyperparameters"]["iterations"]),
                    float(bucket["hyperparameters"]["step_size"]), int(bucket["hyperparameters"]["warm_start_iters"]),
                    context["embeddings"], context["model"], context["gap_pairs"], context["base_gap"])
            label = f"{bucket['method']}_seed{seed}_{format_hparams(bucket['hyperparameters']).replace(',', '_').replace('=', '-')}"
            convergence_histories[label] = history
            (output_paths["logs"] / f"{label}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            with (output_paths["logs"] / f"{label}_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row["pair_rows"][0].keys()))
                writer.writeheader()
                writer.writerows(row["pair_rows"])

    plot_convergence(output_paths["plots"] / "convergence_best.png", convergence_histories)
    plot_scatter(output_paths["plots"] / "fix_vs_newviolations.png", all_rows, "new_violations", "fixed_count",
                 "Fixes vs new violations", "New violations", "Fixed count")
    plot_scatter(output_paths["plots"] / "net_vs_galois_gap.png", all_rows, "Galois_gap_change", "net_improvement",
                 "Net improvement vs Galois gap change", "Galois gap change", "Net improvement")
    if best_coord is not None and best_hybrid is not None:
        plot_hop_grouped_bar(
            output_paths["plots"] / "hop_grouped_bar.png",
            {"hop_details": mean_hop_details(best_coord["rows"])},
            {"hop_details": mean_hop_details(best_hybrid["rows"])},
        )
    plot_pareto(output_paths["plots"] / "pareto_tradeoff.png", all_rows)

    q1 = best_hybrid is not None and best_coord is not None and best_hybrid["mean_fix_rate"] >= best_coord["mean_fix_rate"] - 0.01
    q2 = (
        best_hybrid is not None and best_coord is not None and (
            meaningful_higher(best_hybrid["mean_preservation_metric"], best_coord["mean_preservation_metric"])
            or meaningful_lower(best_hybrid["mean_Galois_gap_change"], best_coord["mean_Galois_gap_change"])
            or meaningful_lower(best_hybrid["mean_box_displacement"], best_coord["mean_box_displacement"])
        )
    )
    q3 = (
        best_hybrid is not None
        and best_coord is not None
        and meaningful_higher(best_hybrid["mean_hop_weighted_improvement"], best_coord["mean_hop_weighted_improvement"])
    )
    q4 = best_hybrid is not None and best_coord is not None and best_hybrid["mean_aggregate_score"] > best_coord["mean_aggregate_score"]
    q5 = (
        best_hybrid is not None
        and best_coord is not None
        and (
            q4
            or (
                q1
                and (q2 or q3)
                and best_hybrid["mean_aggregate_score"] >= best_coord["mean_aggregate_score"] - 0.02
            )
        )
    )

    chunk1_main = Path("results_cegar_chunk1/main_table.csv")
    best_naive_reference = None
    if chunk1_main.exists():
        with chunk1_main.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        naive_rows = [row for row in rows if row["method"] == "Naive CEGAR"]
        if naive_rows:
            best_naive_reference = max(naive_rows, key=lambda row: float(row["aggregate_score"]))

    hard_hop_bucket = "4+"
    coord_hard = np.mean([row["hop_details"].get(hard_hop_bucket, {}).get("fix_rate", 0.0) for row in best_coord["rows"]]) if best_coord is not None else 0.0
    hybrid_hard = np.mean([row["hop_details"].get(hard_hop_bucket, {}).get("fix_rate", 0.0) for row in best_hybrid["rows"]]) if best_hybrid is not None else 0.0
    if best_coord is not None and best_hybrid is not None:
        best_hard_hop = "Hybrid warm-start" if hybrid_hard > coord_hard else "Coordinate descent"
        best_galois = "Hybrid warm-start" if best_hybrid["mean_Galois_gap_change"] < best_coord["mean_Galois_gap_change"] else "Coordinate descent"
        best_displacement = "Hybrid warm-start" if best_hybrid["mean_box_displacement"] < best_coord["mean_box_displacement"] else "Coordinate descent"
    else:
        best_hard_hop = best_overall["method"]
        best_galois = best_overall["method"]
        best_displacement = best_overall["method"]

    if q1 and (q2 or q3 or q4):
        decision_strength = "Strong success"
    elif (not q1) and (q2 or q3 or q4):
        decision_strength = "Moderate success"
    elif best_hybrid is not None:
        decision_strength = "Weak success"
    else:
        decision_strength = "Coordinate-only rerun"

    summary_lines = [
        "# Chunk 2 Summary",
        "",
        f"- Scope: {len(seeds)} seed(s), representative subset of up to {args.max_pairs} constraints per seed",
        f"- Screening iterations: {args.screen_iterations}; full iterations sweep: 15/20/25",
        f"- Seed note: {'single-seed directional run due runtime constraints' if len(seeds) == 1 else 'two-seed run'}",
        f"- Best method on raw fix rate: {best_coord['method'] if best_hybrid is None or (best_coord is not None and best_coord['mean_fix_rate'] >= best_hybrid['mean_fix_rate']) else best_hybrid['method']}",
        f"- Best method on net improvement: {best_coord['method'] if best_hybrid is None or (best_coord is not None and best_coord['mean_net_improvement'] >= best_hybrid['mean_net_improvement']) else best_hybrid['method']}",
        f"- Best method on Galois preservation: {best_galois}",
        f"- Best method on displacement preservation: {best_displacement}",
        f"- Best method on hard-hop repair: {best_hard_hop}",
        f"- Best overall tradeoff: {best_overall['method']} ({format_hparams(best_overall['hyperparameters'])})",
        f"- Decision strength: {decision_strength}",
        "",
        f"- Q1. Does hybrid substantially narrow the gap to coordinate descent on raw repair? {'Yes' if q1 else 'No'}",
        f"- Q2. Does hybrid beat coordinate descent on any preservation metric? {'Yes' if q2 else 'No'}",
        f"- Q3. Does hybrid beat coordinate descent on hard-hop repair or hop-weighted improvement? {'Yes' if q3 else 'No'}",
        f"- Q4. Does hybrid beat coordinate descent on the aggregate tradeoff score? {'Yes' if q4 else 'No'}",
        f"- Q5. Is there now a credible paper claim that hybrid refinement offers a better repair-preservation tradeoff than pure coordinate descent? {'Yes' if q5 else 'No'}",
        "",
        "## Best configs",
    ]
    if best_coord is not None:
        summary_lines.append(f"- Coordinate descent: {format_hparams(best_coord['hyperparameters'])}")
    if best_hybrid is not None:
        summary_lines.append(f"- Hybrid warm-start: {format_hparams(best_hybrid['hyperparameters'])}")
    if best_coord is not None or best_hybrid is not None:
        summary_lines.extend([
            "",
            "## Bootstrap 95% CI",
        ])
    if best_coord is not None:
        summary_lines.extend([
            f"- Coordinate descent fix_rate mean: {best_coord['mean_fix_rate']:.4f}",
            format_ci("Coordinate descent fix_rate CI", best_coord.get("bootstrap"), "fix_rate_ci"),
            f"- Coordinate descent net_improvement mean: {best_coord['mean_net_improvement']:.4f}",
            format_ci("Coordinate descent net_improvement CI", best_coord.get("bootstrap"), "net_improvement_ci"),
            f"- Coordinate descent final_violation_rate mean: {best_coord['mean_final_violation_rate']:.4f}",
            format_ci("Coordinate descent final_violation_rate CI", best_coord.get("bootstrap"), "final_violation_rate_ci"),
        ])
    if best_hybrid is not None:
        summary_lines.extend([
            f"- Hybrid warm-start fix_rate mean: {best_hybrid['mean_fix_rate']:.4f}",
            format_ci("Hybrid warm-start fix_rate CI", best_hybrid.get("bootstrap"), "fix_rate_ci"),
            f"- Hybrid warm-start net_improvement mean: {best_hybrid['mean_net_improvement']:.4f}",
            format_ci("Hybrid warm-start net_improvement CI", best_hybrid.get("bootstrap"), "net_improvement_ci"),
            f"- Hybrid warm-start final_violation_rate mean: {best_hybrid['mean_final_violation_rate']:.4f}",
            format_ci("Hybrid warm-start final_violation_rate CI", best_hybrid.get("bootstrap"), "final_violation_rate_ci"),
        ])
    if best_naive_reference is not None:
        summary_lines.extend([
            "",
            "## Chunk 1 naive reference",
            f"- Naive CEGAR reference: {best_naive_reference['hyperparameters']} with aggregate score {best_naive_reference['aggregate_score']}",
        ])
    (output_paths["root"] / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    rerun_coord = ""
    if best_coord is not None:
        rerun_coord = (
            "python cegar_chunk2_focus.py "
            f"--output_dir {args.output_dir}_coord_rerun --seeds {args.seeds} --n_concepts {args.n_concepts} "
            f"--max_pairs {args.max_pairs} --gap_sample {args.gap_sample} --neighbor_sample {args.neighbor_sample} "
            f"--neighbor_k {args.neighbor_k} --only_method coord --step_size {best_coord['hyperparameters']['step_size']} "
            f"--iterations {best_coord['hyperparameters']['iterations']}"
        )
    rerun_hybrid = ""
    if best_hybrid is not None:
        rerun_hybrid = (
            "python cegar_chunk2_focus.py "
            f"--output_dir {args.output_dir}_hybrid_rerun --seeds {args.seeds} --n_concepts {args.n_concepts} "
            f"--max_pairs {args.max_pairs} --gap_sample {args.gap_sample} --neighbor_sample {args.neighbor_sample} "
            f"--neighbor_k {args.neighbor_k} --only_method hybrid --step_size {best_hybrid['hyperparameters']['step_size']} "
            f"--iterations {best_hybrid['hyperparameters']['iterations']} --warm_start_iters {best_hybrid['hyperparameters']['warm_start_iters']}"
        )
    (output_paths["configs"] / "rerun_commands.txt").write_text(
        f"Best coordinate descent config:\n{rerun_coord}\n\nBest hybrid config:\n{rerun_hybrid}\n",
        encoding="utf-8",
    )

    with (output_paths["configs"] / "selected_configs.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "selected_coordinate_configs": selected_coord,
            "selected_hybrid_configs": selected_hybrid,
            "best_coordinate_config": best_coord["hyperparameters"] if best_coord is not None else None,
            "best_hybrid_config": best_hybrid["hyperparameters"] if best_hybrid is not None else None,
            "best_overall": {
                "method": best_overall["method"],
                "hyperparameters": best_overall["hyperparameters"],
            },
            "bootstrap": {
                "coordinate_descent": best_coord["bootstrap"] if best_coord is not None else {},
                "hybrid": best_hybrid["bootstrap"] if best_hybrid is not None else {},
            },
        }, handle, indent=2)

    print("\n[5/5] Done.")
    print(f"  Main table: {output_paths['root'] / 'main_table.csv'}")
    print(f"  Top configs: {output_paths['root'] / 'top_configs.csv'}")
    print(f"  Summary: {output_paths['root'] / 'summary.md'}")


if __name__ == "__main__":
    main()
