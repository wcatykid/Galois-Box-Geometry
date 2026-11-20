"""
Fast first-pass triage for inference-time geometric refinement methods.

Inference-time only: loads an existing cached box model and evaluates repair
methods on a noun-trained -> verb WordNet domain-shift setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_NLTK_DATA = REPO_ROOT / "nltk_data"
os.environ["NLTK_DATA"] = str(LOCAL_NLTK_DATA)

import numpy as np

import gensim.downloader as api

import nltk
if str(LOCAL_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(LOCAL_NLTK_DATA))

nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn

from eval_hyperlex_embeddings import BOX_DIM, HIDDEN_DIM
from inference_cegar import measure_galois_consistency_gap
from inference_cegar_verb import (
    build_verb_concept_set,
    build_verb_isa_graph,
    encode_verb_concepts,
)
from src.improved_trainer import GaloisModel

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SEED = 42
VIOLATION_THRESHOLD = 0.5
TOP_FRACTION = 0.20
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
DELTA = 0.5
EPSILON = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results_cegar_chunk1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_concepts", type=int, default=5000)
    parser.add_argument("--max_pairs", type=int, default=1200)
    parser.add_argument("--gap_sample", type=int, default=200)
    parser.add_argument("--only_method", default="all",
                        choices=["all", "none", "random", "coord", "naive", "severity", "hybrid"])
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lambda_p", type=float, default=None)
    parser.add_argument("--lambda_a", type=float, default=None)
    parser.add_argument("--warm_start_iters", type=int, default=None)
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
        "logs": root / "logs",
        "configs": root / "configs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_boxes(boxes: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {key: (bounds[0].copy(), bounds[1].copy()) for key, bounds in boxes.items()}


def forward_violation(l_i: np.ndarray, u_i: np.ndarray, l_j: np.ndarray, u_j: np.ndarray) -> float:
    return float(np.maximum(0.0, l_j - l_i).sum() + np.maximum(0.0, u_i - u_j).sum())


def backward_violation(l_i: np.ndarray, u_i: np.ndarray, l_j: np.ndarray, u_j: np.ndarray) -> float:
    return float(np.maximum(0.0, l_i - l_j).sum() + np.maximum(0.0, u_j - u_i).sum())


def project_valid_box(l: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(l, u), np.maximum(l, u)


def repair_pair(boxes: dict[str, tuple[np.ndarray, np.ndarray]], hypo: str, hyper: str, step_size: float) -> float:
    l_i, u_i = boxes[hypo]
    l_j, u_j = boxes[hyper]

    lower_viol = np.maximum(0.0, l_j - l_i)
    upper_viol = np.maximum(0.0, u_i - u_j)
    displacement = 0.0

    l_j_new = l_j - step_size * lower_viol
    u_j_new = u_j + step_size * upper_viol
    l_j_new, u_j_new = project_valid_box(l_j_new, u_j_new)
    displacement += float(np.abs(l_j_new - l_j).sum() + np.abs(u_j_new - u_j).sum())
    boxes[hyper] = (l_j_new, u_j_new)

    lower_viol_2 = np.maximum(0.0, l_j_new - l_i)
    upper_viol_2 = np.maximum(0.0, u_i - u_j_new)
    l_i_new = l_i + step_size * lower_viol_2
    u_i_new = u_i - step_size * upper_viol_2
    l_i_new, u_i_new = project_valid_box(l_i_new, u_i_new)
    displacement += float(np.abs(l_i_new - l_i).sum() + np.abs(u_i_new - u_i).sum())
    boxes[hypo] = (l_i_new, u_i_new)

    return displacement


def random_pair_update(
    boxes: dict[str, tuple[np.ndarray, np.ndarray]],
    hypo: str,
    hyper: str,
    step_size: float,
    rng: np.random.Generator,
) -> float:
    l_i, u_i = boxes[hypo]
    l_j, u_j = boxes[hyper]

    lower_viol = np.maximum(0.0, l_j - l_i)
    upper_viol = np.maximum(0.0, u_i - u_j)
    lower_sign = rng.choice(np.array([-1.0, 1.0]), size=l_i.shape)
    upper_sign = rng.choice(np.array([-1.0, 1.0]), size=u_i.shape)
    displacement = 0.0

    l_j_new = l_j + lower_sign * step_size * lower_viol
    u_j_new = u_j + upper_sign * step_size * upper_viol
    l_j_new, u_j_new = project_valid_box(l_j_new, u_j_new)
    displacement += float(np.abs(l_j_new - l_j).sum() + np.abs(u_j_new - u_j).sum())
    boxes[hyper] = (l_j_new, u_j_new)

    lower_sign_2 = rng.choice(np.array([-1.0, 1.0]), size=l_i.shape)
    upper_sign_2 = rng.choice(np.array([-1.0, 1.0]), size=u_i.shape)
    l_i_new = l_i + lower_sign_2 * step_size * lower_viol
    u_i_new = u_i + upper_sign_2 * step_size * upper_viol
    l_i_new, u_i_new = project_valid_box(l_i_new, u_i_new)
    displacement += float(np.abs(l_i_new - l_i).sum() + np.abs(u_i_new - u_i).sum())
    boxes[hypo] = (l_i_new, u_i_new)

    return displacement


def sample_constraints(
    constraints: dict[tuple[str, str], int],
    max_pairs: int,
    rng: np.random.Generator,
) -> dict[tuple[str, str], int]:
    if len(constraints) <= max_pairs:
        return dict(constraints)

    buckets = {"1": [], "2": [], "3": [], "4+": []}
    for pair, hop in constraints.items():
        bucket = str(hop) if hop <= 3 else "4+"
        buckets[bucket].append((pair, hop))

    selected: list[tuple[tuple[str, str], int]] = []
    total = len(constraints)
    for bucket in ["1", "2", "3", "4+"]:
        items = buckets[bucket]
        if not items:
            continue
        target = max(1, int(round(max_pairs * len(items) / total)))
        target = min(target, len(items))
        indices = rng.choice(len(items), size=target, replace=False)
        selected.extend([items[index] for index in indices])

    if len(selected) > max_pairs:
        indices = rng.choice(len(selected), size=max_pairs, replace=False)
        selected = [selected[index] for index in indices]
    elif len(selected) < max_pairs:
        chosen = {pair for pair, _ in selected}
        remaining = [(pair, hop) for pair, hop in constraints.items() if pair not in chosen]
        if remaining:
            extra = min(max_pairs - len(selected), len(remaining))
            indices = rng.choice(len(remaining), size=extra, replace=False)
            selected.extend([remaining[index] for index in indices])

    return {pair: hop for pair, hop in selected}


def compute_pair_stats(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    threshold: float = VIOLATION_THRESHOLD,
) -> dict[str, object]:
    original_violations = 0
    fixed_count = 0
    new_violations = 0
    final_violations = 0
    hop_stats = {}
    pair_rows = []

    for (hypo, hyper), hop in constraints.items():
        if hypo not in boxes_before or hyper not in boxes_before:
            continue
        l0_i, u0_i = boxes_before[hypo]
        l0_j, u0_j = boxes_before[hyper]
        l1_i, u1_i = boxes_after[hypo]
        l1_j, u1_j = boxes_after[hyper]
        vf_before = forward_violation(l0_i, u0_i, l0_j, u0_j)
        vf_after = forward_violation(l1_i, u1_i, l1_j, u1_j)
        vb_before = backward_violation(l0_i, u0_i, l0_j, u0_j)
        vb_after = backward_violation(l1_i, u1_i, l1_j, u1_j)
        before_flag = vf_before > threshold
        after_flag = vf_after > threshold
        bucket = str(hop) if hop <= 3 else "4+"

        stats = hop_stats.setdefault(bucket, {"original": 0, "fixed": 0, "new": 0, "final": 0})
        if before_flag:
            original_violations += 1
            stats["original"] += 1
        if after_flag:
            final_violations += 1
            stats["final"] += 1
        if before_flag and not after_flag:
            fixed_count += 1
            stats["fixed"] += 1
        if not before_flag and after_flag:
            new_violations += 1
            stats["new"] += 1

        pair_rows.append({
            "hyponym": hypo,
            "hypernym": hyper,
            "hop": hop,
            "viol_fwd_before": vf_before,
            "viol_fwd_after": vf_after,
            "viol_bwd_before": vb_before,
            "viol_bwd_after": vb_after,
            "was_violated": before_flag,
            "is_violated": after_flag,
        })

    total_pairs = len(pair_rows)
    for bucket, stats in hop_stats.items():
        stats["fix_rate"] = stats["fixed"] / max(stats["original"], 1)

    return {
        "pair_rows": pair_rows,
        "hop_stats": hop_stats,
        "total_pairs": total_pairs,
        "original_violations": original_violations,
        "fixed_count": fixed_count,
        "new_violations": new_violations,
        "net_improvement": fixed_count - new_violations,
        "final_violations": final_violations,
        "final_violation_rate": final_violations / max(total_pairs, 1),
        "fix_rate": fixed_count / max(original_violations, 1),
    }


def mean_box_displacement(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    displacements = []
    for key in boxes_before:
        l0, u0 = boxes_before[key]
        l1, u1 = boxes_after[key]
        displacements.append(float(np.abs(l1 - l0).sum() + np.abs(u1 - u0).sum()))
    return float(np.mean(displacements)) if displacements else 0.0


def mean_box_size(boxes: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    widths = []
    for lower, upper in boxes.values():
        widths.append(float(np.abs(upper - lower).sum()))
    return float(np.mean(widths)) if widths else 1.0


def get_current_violated(
    boxes: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    threshold: float = VIOLATION_THRESHOLD,
) -> list[dict[str, object]]:
    violated = []
    for (hypo, hyper), hop in constraints.items():
        l_i, u_i = boxes[hypo]
        l_j, u_j = boxes[hyper]
        vf = forward_violation(l_i, u_i, l_j, u_j)
        if vf > threshold:
            vb = backward_violation(l_i, u_i, l_j, u_j)
            violated.append({
                "hypo": hypo,
                "hyper": hyper,
                "hop": hop,
                "vf": vf,
                "vb": vb,
            })
    return violated


def compute_iteration_snapshot(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_current: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    embeddings: dict[str, np.ndarray],
    model: GaloisModel,
    gap_pairs: list[tuple[str, str]],
    box_size_scale: float,
    iteration: int,
) -> dict[str, object]:
    pair_stats = compute_pair_stats(boxes_before, boxes_current, constraints)
    gap = measure_galois_consistency_gap(boxes_current, embeddings, model, gap_pairs, n_sample=len(gap_pairs))
    return {
        "iteration": iteration,
        "fixed_count": pair_stats["fixed_count"],
        "new_violations": pair_stats["new_violations"],
        "net_improvement": pair_stats["net_improvement"],
        "final_violation_rate": pair_stats["final_violation_rate"],
        "mean_box_displacement": mean_box_displacement(boxes_before, boxes_current),
        "galois_gap": gap["mean"],
        "galois_gap_change": gap["mean"],
        "displacement_norm": mean_box_displacement(boxes_before, boxes_current) / max(box_size_scale, 1e-8),
    }


def run_no_refinement(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    **_: object,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    return copy_boxes(boxes_before), []


def run_random(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    rng: np.random.Generator,
    record: bool = False,
    **context: object,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    targeted_pairs = get_current_violated(boxes_before, constraints)
    history = []

    for iteration in range(1, iterations + 1):
        for pair in targeted_pairs:
            hypo = str(pair["hypo"])
            hyper = str(pair["hyper"])
            l_i, u_i = boxes[hypo]
            l_j, u_j = boxes[hyper]
            if forward_violation(l_i, u_i, l_j, u_j) > VIOLATION_THRESHOLD:
                random_pair_update(boxes, hypo, hyper, step_size, rng)
        if record:
            history.append(compute_iteration_snapshot(boxes_before, boxes, constraints, **context, iteration=iteration))

    return boxes, history


def run_coord_descent(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    record: bool = False,
    **context: object,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    history = []
    for iteration in range(1, iterations + 1):
        violated = get_current_violated(boxes, constraints)
        if not violated:
            break
        for pair in violated:
            repair_pair(boxes, str(pair["hypo"]), str(pair["hyper"]), step_size)
        if record:
            history.append(compute_iteration_snapshot(boxes_before, boxes, constraints, **context, iteration=iteration))
    return boxes, history


def run_naive_cegar(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    record: bool = False,
    **context: object,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    targeted_pairs = get_current_violated(boxes_before, constraints)
    history = []
    for iteration in range(1, iterations + 1):
        for pair in targeted_pairs:
            hypo = str(pair["hypo"])
            hyper = str(pair["hyper"])
            l_i, u_i = boxes[hypo]
            l_j, u_j = boxes[hyper]
            if forward_violation(l_i, u_i, l_j, u_j) > VIOLATION_THRESHOLD:
                repair_pair(boxes, hypo, hyper, step_size)
        if record:
            history.append(compute_iteration_snapshot(boxes_before, boxes, constraints, **context, iteration=iteration))
    return boxes, history


def run_severity_cegar(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    lambda_p: float,
    lambda_a: float,
    top_fraction: float = TOP_FRACTION,
    record: bool = False,
    **context: object,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    history = []
    persistence: dict[tuple[str, str], int] = {}

    for iteration in range(1, iterations + 1):
        current = get_current_violated(boxes, constraints)
        if not current:
            break
        current_keys = {(str(pair["hypo"]), str(pair["hyper"])) for pair in current}
        for key in current_keys:
            persistence[key] = persistence.get(key, 0) + 1
        for key in list(persistence):
            if key not in current_keys:
                persistence[key] = max(0, persistence[key] - 1)

        scored = []
        for pair in current:
            hypo = str(pair["hypo"])
            hyper = str(pair["hyper"])
            severity = (
                pair["vf"]
                + lambda_p * persistence.get((hypo, hyper), 0)
                + lambda_a * abs(float(pair["vb"]) - float(pair["vf"]))
            )
            scored.append((severity, hypo, hyper))

        scored.sort(reverse=True)
        n_repair = max(1, int(np.ceil(len(scored) * top_fraction)))
        for _, hypo, hyper in scored[:n_repair]:
            repair_pair(boxes, hypo, hyper, step_size)
        if record:
            history.append(compute_iteration_snapshot(boxes_before, boxes, constraints, **context, iteration=iteration))

    return boxes, history


def run_hybrid(
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    iterations: int,
    step_size: float,
    warm_start_iters: int,
    lambda_p: float,
    lambda_a: float,
    top_fraction: float = TOP_FRACTION,
    record: bool = False,
    **context: object,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    boxes = copy_boxes(boxes_before)
    history = []
    persistence: dict[tuple[str, str], int] = {}

    for iteration in range(1, iterations + 1):
        current = get_current_violated(boxes, constraints)
        if not current:
            break
        if iteration <= warm_start_iters:
            for pair in current:
                repair_pair(boxes, str(pair["hypo"]), str(pair["hyper"]), step_size)
        else:
            current_keys = {(str(pair["hypo"]), str(pair["hyper"])) for pair in current}
            for key in current_keys:
                persistence[key] = persistence.get(key, 0) + 1
            for key in list(persistence):
                if key not in current_keys:
                    persistence[key] = max(0, persistence[key] - 1)

            scored = []
            for pair in current:
                hypo = str(pair["hypo"])
                hyper = str(pair["hyper"])
                severity = (
                    pair["vf"]
                    + lambda_p * persistence.get((hypo, hyper), 0)
                    + lambda_a * abs(float(pair["vb"]) - float(pair["vf"]))
                )
                scored.append((severity, hypo, hyper))
            scored.sort(reverse=True)
            n_repair = max(1, int(np.ceil(len(scored) * top_fraction)))
            for _, hypo, hyper in scored[:n_repair]:
                repair_pair(boxes, hypo, hyper, step_size)
        if record:
            history.append(compute_iteration_snapshot(boxes_before, boxes, constraints, **context, iteration=iteration))

    return boxes, history


def evaluate_run(
    method: str,
    hyperparameters: dict[str, object],
    boxes_before: dict[str, tuple[np.ndarray, np.ndarray]],
    boxes_after: dict[str, tuple[np.ndarray, np.ndarray]],
    constraints: dict[tuple[str, str], int],
    embeddings: dict[str, np.ndarray],
    model: GaloisModel,
    gap_pairs: list[tuple[str, str]],
    base_gap: float,
    box_size_scale: float,
) -> dict[str, object]:
    pair_stats = compute_pair_stats(boxes_before, boxes_after, constraints)
    gap_after = measure_galois_consistency_gap(boxes_after, embeddings, model, gap_pairs, n_sample=len(gap_pairs))
    gap_change = float(gap_after["mean"] - base_gap)
    displacement = mean_box_displacement(boxes_before, boxes_after)
    new_violation_rate = pair_stats["new_violations"] / max(pair_stats["total_pairs"], 1)
    net_norm = pair_stats["net_improvement"] / max(pair_stats["original_violations"], 1)
    gap_increase_norm = max(gap_change, 0.0) / max(abs(base_gap), 1e-8)
    displacement_norm = displacement / max(box_size_scale, 1e-8)
    score = (
        ALPHA * pair_stats["fix_rate"]
        + BETA * net_norm
        - GAMMA * new_violation_rate
        - DELTA * gap_increase_norm
        - EPSILON * displacement_norm
    )
    return {
        "method": method,
        "hyperparameters": hyperparameters,
        "fixed_count": pair_stats["fixed_count"],
        "fix_rate": pair_stats["fix_rate"],
        "new_violations": pair_stats["new_violations"],
        "net_improvement": pair_stats["net_improvement"],
        "final_violation_rate": pair_stats["final_violation_rate"],
        "Galois_gap_change": gap_change,
        "mean_box_displacement": displacement,
        "aggregate_score": score,
        "original_violations": pair_stats["original_violations"],
        "final_violations": pair_stats["final_violations"],
        "total_pairs": pair_stats["total_pairs"],
        "hop_stratified_fix_rate": {bucket: stats["fix_rate"] for bucket, stats in pair_stats["hop_stats"].items()},
        "hop_details": pair_stats["hop_stats"],
        "pair_rows": pair_stats["pair_rows"],
    }


def format_hparams(hparams: dict[str, object]) -> str:
    if not hparams:
        return "-"
    return ",".join(f"{key}={value}" for key, value in hparams.items())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method",
            "hyperparameters",
            "fixed_count",
            "fix_rate",
            "new_violations",
            "net_improvement",
            "final_violation_rate",
            "Galois_gap_change",
            "mean_box_displacement",
            "aggregate_score",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": row["method"],
                "hyperparameters": format_hparams(row["hyperparameters"]),
                "fixed_count": row["fixed_count"],
                "fix_rate": row["fix_rate"],
                "new_violations": row["new_violations"],
                "net_improvement": row["net_improvement"],
                "final_violation_rate": row["final_violation_rate"],
                "Galois_gap_change": row["Galois_gap_change"],
                "mean_box_displacement": row["mean_box_displacement"],
                "aggregate_score": row["aggregate_score"],
            })


def save_pair_rows(path: Path, pair_rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)


def plot_convergence(path: Path, convergence_logs: dict[str, list[dict[str, object]]]) -> None:
    if plt is None or not convergence_logs:
        return
    plt.figure(figsize=(10, 6))
    for label, history in convergence_logs.items():
        xs = [row["iteration"] for row in history]
        ys = [row["final_violation_rate"] for row in history]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Final violation rate")
    plt.title("Convergence of top methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_fix_vs_new(path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 6))
    for row in rows:
        plt.scatter(row["new_violations"], row["fixed_count"], s=60)
        plt.text(row["new_violations"] + 0.2, row["fixed_count"] + 0.2, row["method"], fontsize=8)
    plt.xlabel("New violations")
    plt.ylabel("Fixed count")
    plt.title("Fixes vs introduced violations")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_paths = ensure_dirs(Path(args.output_dir))
    rng = np.random.default_rng(args.seed)

    print(f"\n{'=' * 88}")
    print("  Fast First-Pass Inference-Time Repair Triage")
    print(f"{'=' * 88}")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {output_paths['root']}")

    print("\n[1/5] Loading cached noun-trained model and GloVe...")
    glove = api.load("glove-wiki-gigaword-300")
    model_cache = Path("results/inference_cegar_model.npz")
    if not model_cache.exists():
        raise FileNotFoundError("Missing results/inference_cegar_model.npz. Run inference_cegar.py first.")

    model_weights = np.load(model_cache, allow_pickle=True)
    model = GaloisModel(300, HIDDEN_DIM, BOX_DIM, np.random.default_rng(args.seed))
    for key in model.params_dict():
        setattr(model, key, model_weights[key])

    print("\n[2/5] Building verb domain-shift constraint set...")
    verb_concepts = build_verb_concept_set(
        set(glove.key_to_index.keys()),
        n_concepts=args.n_concepts,
        min_depth=2,
        max_depth=8,
        rng=np.random.default_rng(args.seed),
    )
    verb_lemmas = [item[1] for item in verb_concepts]
    boxes, embeddings = encode_verb_concepts(verb_lemmas, model, glove)
    _, all_constraints = build_verb_isa_graph(verb_concepts, boxes)
    constraints = sample_constraints(all_constraints, args.max_pairs, rng)
    gap_pairs = list(constraints.keys())[: min(args.gap_sample, len(constraints))]
    base_gap = measure_galois_consistency_gap(boxes, embeddings, model, gap_pairs, n_sample=len(gap_pairs))["mean"]
    box_size_scale = mean_box_size(boxes)

    baseline_stats = compute_pair_stats(boxes, boxes, constraints)
    print(f"  Representative subset: {len(constraints)} constraints from {len(all_constraints)} total")
    print(f"  Original violations: {baseline_stats['original_violations']}/{baseline_stats['total_pairs']}")

    context = {
        "embeddings": embeddings,
        "model": model,
        "gap_pairs": gap_pairs,
        "box_size_scale": box_size_scale,
    }

    print("\n[3/5] Running fast triage sweep...")
    results: list[dict[str, object]] = []

    def record_run(method: str, hparams: dict[str, object], boxes_after: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
        row = evaluate_run(method, hparams, boxes, boxes_after, constraints, embeddings, model, gap_pairs, base_gap, box_size_scale)
        results.append(row)
        print(
            f"  {method:24s} {format_hparams(hparams):26s} "
            f"fix={row['fixed_count']:4d} ({100 * row['fix_rate']:5.1f}%) "
            f"new={row['new_violations']:4d} net={row['net_improvement']:+5d} "
            f"gap={row['Galois_gap_change']:+.4f} disp={row['mean_box_displacement']:.2f} "
            f"S={row['aggregate_score']:+.4f}"
        )
        return row

    method_filter = args.only_method
    if method_filter in ("all", "none"):
        record_run("No refinement", {}, copy_boxes(boxes))

    random_stage1 = []
    coord_stage1 = []
    naive_stage1 = []
    sev_stage1 = []

    if method_filter in ("all", "random"):
        for step_size in ([args.step_size] if args.step_size is not None else [0.1, 0.3]):
            boxes_after, _ = run_random(boxes, constraints, args.iterations or 10, step_size, np.random.default_rng(99), False, **context)
            random_stage1.append(record_run("Random perturbation", {"step_size": step_size, "iterations": args.iterations or 10}, boxes_after))

    if method_filter in ("all", "coord"):
        for step_size in ([args.step_size] if args.step_size is not None else [0.1, 0.3]):
            boxes_after, _ = run_coord_descent(boxes, constraints, args.iterations or 10, step_size, False, **context)
            coord_stage1.append(record_run("Coordinate descent", {"step_size": step_size, "iterations": args.iterations or 10}, boxes_after))

    if method_filter in ("all", "naive"):
        for step_size in ([args.step_size] if args.step_size is not None else [0.1, 0.3]):
            boxes_after, _ = run_naive_cegar(boxes, constraints, args.iterations or 10, step_size, False, **context)
            naive_stage1.append(record_run("Naive CEGAR", {"step_size": step_size, "iterations": args.iterations or 10}, boxes_after))

    if method_filter in ("all", "severity"):
        lambda_ps = [args.lambda_p] if args.lambda_p is not None else [0.0, 1.0]
        lambda_as = [args.lambda_a] if args.lambda_a is not None else [0.0, 0.5]
        step_sizes = [args.step_size] if args.step_size is not None else [0.1, 0.3]
        for step_size in step_sizes:
            for lambda_p in lambda_ps:
                for lambda_a in lambda_as:
                    boxes_after, _ = run_severity_cegar(
                        boxes, constraints, args.iterations or 10, step_size, lambda_p, lambda_a, TOP_FRACTION, False, **context)
                    sev_stage1.append(record_run(
                        "Severity-weighted CEGAR",
                        {"step_size": step_size, "iterations": args.iterations or 10, "lambda_p": lambda_p, "lambda_a": lambda_a, "top_fraction": TOP_FRACTION},
                        boxes_after,
                    ))

    if method_filter == "all":
        best_random = max(random_stage1, key=lambda row: row["aggregate_score"]) if random_stage1 else None
        best_coord = max(coord_stage1, key=lambda row: row["aggregate_score"]) if coord_stage1 else None
        best_naive = max(naive_stage1, key=lambda row: row["aggregate_score"]) if naive_stage1 else None
        best_severity_candidates = sorted(sev_stage1, key=lambda row: row["aggregate_score"], reverse=True)[:3]

        for row in [best_random, best_coord, best_naive]:
            if row and row["hyperparameters"]["iterations"] != 15:
                hparams = dict(row["hyperparameters"])
                hparams["iterations"] = 15
                if row["method"] == "Random perturbation":
                    boxes_after, _ = run_random(boxes, constraints, 15, float(hparams["step_size"]), np.random.default_rng(99), False, **context)
                elif row["method"] == "Coordinate descent":
                    boxes_after, _ = run_coord_descent(boxes, constraints, 15, float(hparams["step_size"]), False, **context)
                else:
                    boxes_after, _ = run_naive_cegar(boxes, constraints, 15, float(hparams["step_size"]), False, **context)
                record_run(row["method"], hparams, boxes_after)

        for row in best_severity_candidates:
            hparams = dict(row["hyperparameters"])
            hparams["iterations"] = 15
            boxes_after, _ = run_severity_cegar(
                boxes,
                constraints,
                15,
                float(hparams["step_size"]),
                float(hparams["lambda_p"]),
                float(hparams["lambda_a"]),
                TOP_FRACTION,
                False,
                **context,
            )
            record_run("Severity-weighted CEGAR", hparams, boxes_after)

        best_severity = max(
            [row for row in results if row["method"] == "Severity-weighted CEGAR"],
            key=lambda row: row["aggregate_score"],
        )
        for warm_start in [3, 5]:
            for iterations in [10, 15]:
                if iterations <= warm_start:
                    continue
                boxes_after, _ = run_hybrid(
                    boxes,
                    constraints,
                    iterations,
                    float(best_severity["hyperparameters"]["step_size"]),
                    warm_start,
                    float(best_severity["hyperparameters"]["lambda_p"]),
                    float(best_severity["hyperparameters"]["lambda_a"]),
                    TOP_FRACTION,
                    False,
                    **context,
                )
                record_run(
                    "Hybrid warm-start",
                    {
                        "step_size": float(best_severity["hyperparameters"]["step_size"]),
                        "iterations": iterations,
                        "warm_start_iters": warm_start,
                        "lambda_p": float(best_severity["hyperparameters"]["lambda_p"]),
                        "lambda_a": float(best_severity["hyperparameters"]["lambda_a"]),
                        "top_fraction": TOP_FRACTION,
                    },
                    boxes_after,
                )

    if method_filter == "hybrid":
        if None in (args.step_size, args.iterations, args.lambda_p, args.lambda_a, args.warm_start_iters):
            raise ValueError("Hybrid single-run mode requires --step_size, --iterations, --lambda_p, --lambda_a, and --warm_start_iters.")
        boxes_after, _ = run_hybrid(
            boxes, constraints, args.iterations, args.step_size, args.warm_start_iters, args.lambda_p, args.lambda_a, TOP_FRACTION, False, **context)
        record_run(
            "Hybrid warm-start",
            {
                "step_size": args.step_size,
                "iterations": args.iterations,
                "warm_start_iters": args.warm_start_iters,
                "lambda_p": args.lambda_p,
                "lambda_a": args.lambda_a,
                "top_fraction": TOP_FRACTION,
            },
            boxes_after,
        )

    if method_filter in ("severity", "coord", "naive", "random") and args.iterations == 15:
        pass

    ranked = sorted(results, key=lambda row: row["aggregate_score"], reverse=True)
    top_methods = ranked[:3]

    print("\n[4/5] Writing outputs and detailed logs...")
    write_csv(output_paths["root"] / "main_table.csv", ranked)
    write_csv(output_paths["root"] / "top_methods.csv", top_methods)

    convergence_logs = {}
    for index, row in enumerate(top_methods, start=1):
        method = row["method"]
        hparams = row["hyperparameters"]
        if method == "No refinement":
            final_boxes, history = run_no_refinement(boxes)
        elif method == "Random perturbation":
            final_boxes, history = run_random(boxes, constraints, int(hparams["iterations"]), float(hparams["step_size"]), np.random.default_rng(99), True, **context)
        elif method == "Coordinate descent":
            final_boxes, history = run_coord_descent(boxes, constraints, int(hparams["iterations"]), float(hparams["step_size"]), True, **context)
        elif method == "Naive CEGAR":
            final_boxes, history = run_naive_cegar(boxes, constraints, int(hparams["iterations"]), float(hparams["step_size"]), True, **context)
        elif method == "Severity-weighted CEGAR":
            final_boxes, history = run_severity_cegar(boxes, constraints, int(hparams["iterations"]), float(hparams["step_size"]), float(hparams["lambda_p"]), float(hparams["lambda_a"]), TOP_FRACTION, True, **context)
        else:
            final_boxes, history = run_hybrid(
                boxes,
                constraints,
                int(hparams["iterations"]),
                float(hparams["step_size"]),
                int(hparams["warm_start_iters"]),
                float(hparams["lambda_p"]),
                float(hparams["lambda_a"]),
                TOP_FRACTION,
                True,
                **context,
            )

        for item in history:
            item["galois_gap_change"] = item["galois_gap"] - base_gap
        label = f"{index}_{method}_{format_hparams(hparams).replace(',', '_').replace('=', '-')}"
        convergence_logs[label] = history
        (output_paths["logs"] / f"{label}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        pair_rows = compute_pair_stats(boxes, final_boxes, constraints)["pair_rows"]
        save_pair_rows(output_paths["logs"] / f"{label}_pairs.csv", pair_rows)

    plot_convergence(output_paths["plots"] / "convergence_top3.png", convergence_logs)
    plot_fix_vs_new(output_paths["plots"] / "fix_vs_newviolations.png", top_methods)

    best_fix = max(ranked, key=lambda row: row["fix_rate"])
    best_net = max(ranked, key=lambda row: row["net_improvement"])
    best_geom = min(ranked, key=lambda row: max(row["Galois_gap_change"], 0.0) + row["mean_box_displacement"])
    best_overall = ranked[0]
    best_naive = max([row for row in ranked if row["method"] == "Naive CEGAR"], key=lambda row: row["aggregate_score"], default=None)
    best_severity = max([row for row in ranked if row["method"] == "Severity-weighted CEGAR"], key=lambda row: row["aggregate_score"], default=None)
    best_hybrid = max([row for row in ranked if row["method"] == "Hybrid warm-start"], key=lambda row: row["aggregate_score"], default=None)
    best_coord = max([row for row in ranked if row["method"] == "Coordinate descent"], key=lambda row: row["aggregate_score"], default=None)
    best_cegar_tradeoff = max([row for row in ranked if "CEGAR" in row["method"] or "Hybrid" in row["method"]], key=lambda row: row["aggregate_score"], default=None)

    q1 = best_severity is not None and best_naive is not None and best_severity["aggregate_score"] > best_naive["aggregate_score"]
    q2 = best_hybrid is not None and best_naive is not None and best_hybrid["aggregate_score"] > best_naive["aggregate_score"]
    q3 = best_cegar_tradeoff is not None and best_coord is not None and (best_coord["aggregate_score"] - best_cegar_tradeoff["aggregate_score"]) < 0.10
    q4 = best_cegar_tradeoff is not None and best_coord is not None and (
        best_cegar_tradeoff["aggregate_score"] > best_coord["aggregate_score"]
        or (
            best_cegar_tradeoff["mean_box_displacement"] < best_coord["mean_box_displacement"]
            and best_cegar_tradeoff["new_violations"] <= best_coord["new_violations"]
        )
    )

    advance = []
    for row in ranked:
        if row["method"] not in [item["method"] for item in advance] and row["method"] != "No refinement":
            advance.append(row)
        if len(advance) >= 2:
            break

    promising = q1 or q2 or q3 or q4
    subset_note = (
        f"A representative stratified subset of {len(constraints)} constraints was used "
        f"from {len(all_constraints)} total verb IS-A constraints to keep this chunk fast."
    )
    summary = f"""# Fast First-Pass Chunk 1 Summary

{subset_note}

## Winners

- Best method on raw fix rate: {best_fix['method']} ({format_hparams(best_fix['hyperparameters'])})
- Best method on net improvement: {best_net['method']} ({format_hparams(best_net['hyperparameters'])})
- Best method on geometry preservation: {best_geom['method']} ({format_hparams(best_geom['hyperparameters'])})
- Best overall tradeoff: {best_overall['method']} ({format_hparams(best_overall['hyperparameters'])})

## Decision Rules

- Q1. Does severity-weighted CEGAR beat naive CEGAR? {'Yes' if q1 else 'No'}
- Q2. Does hybrid beat naive CEGAR? {'Yes' if q2 else 'No'}
- Q3. Does any CEGAR-based method come meaningfully closer to coordinate descent? {'Yes' if q3 else 'No'}
- Q4. Is there evidence that CEGAR-based methods are better on tradeoff metrics even if not on raw fix rate? {'Yes' if q4 else 'No'}
- Q5. Which 2 methods should advance to Chunk 2? {advance[0]['method']} ({format_hparams(advance[0]['hyperparameters'])}) and {advance[1]['method']} ({format_hparams(advance[1]['hyperparameters'])})

## Recommendation

Improved CEGAR looks {'promising enough for Chunk 2' if promising else 'too weak to justify Chunk 2 right now'}.
"""
    (output_paths["root"] / "summary.md").write_text(summary, encoding="utf-8")

    rerun_full = (
        "python cegar_chunk1_fastpass.py "
        f"--output_dir {args.output_dir} --seed {args.seed} --n_concepts {args.n_concepts} --max_pairs {args.max_pairs} --gap_sample {args.gap_sample}"
    )
    best_cfg = best_overall["hyperparameters"]
    rerun_best = rerun_full
    if best_overall["method"] != "No refinement":
        method_arg = {
            "Random perturbation": "random",
            "Coordinate descent": "coord",
            "Naive CEGAR": "naive",
            "Severity-weighted CEGAR": "severity",
            "Hybrid warm-start": "hybrid",
        }[best_overall["method"]]
        rerun_best = (
            "python cegar_chunk1_fastpass.py "
            f"--output_dir {args.output_dir}_best_rerun --seed {args.seed} --n_concepts {args.n_concepts} --max_pairs {args.max_pairs} --gap_sample {args.gap_sample} "
            f"--only_method {method_arg} --step_size {best_cfg.get('step_size', 0.1)} --iterations {best_cfg.get('iterations', 10)}"
        )
        if "lambda_p" in best_cfg:
            rerun_best += f" --lambda_p {best_cfg['lambda_p']} --lambda_a {best_cfg['lambda_a']}"
        if "warm_start_iters" in best_cfg:
            rerun_best += f" --warm_start_iters {best_cfg['warm_start_iters']}"
    (output_paths["configs"] / "rerun_commands.txt").write_text(
        f"Full chunk:\n{rerun_full}\n\nBest configuration:\n{rerun_best}\n",
        encoding="utf-8",
    )
    (output_paths["configs"] / "experiment_config.json").write_text(
        json.dumps({
            "seed": args.seed,
            "n_concepts": args.n_concepts,
            "max_pairs": args.max_pairs,
            "gap_sample": args.gap_sample,
            "subset_constraints": len(constraints),
            "total_constraints": len(all_constraints),
            "base_gap": base_gap,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n[5/5] Done.")
    print(f"  Main table: {output_paths['root'] / 'main_table.csv'}")
    print(f"  Top methods: {output_paths['root'] / 'top_methods.csv'}")
    print(f"  Summary: {output_paths['root'] / 'summary.md'}")
    print(f"  Best overall: {best_overall['method']} ({format_hparams(best_overall['hyperparameters'])})")


if __name__ == "__main__":
    main()
