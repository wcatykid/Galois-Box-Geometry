"""
Chunk 2: Full-scale evaluation of the two Chunk-1 winners.

Runs Coordinate descent and Hybrid warm-start on ALL verb IS-A constraints
(no sub-sampling) to determine the final best inference-time repair method.
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

# Reuse all the core functions from chunk 1
from cegar_chunk1_fastpass import (
    VIOLATION_THRESHOLD,
    TOP_FRACTION,
    ALPHA,
    BETA,
    GAMMA,
    DELTA,
    EPSILON,
    copy_boxes,
    forward_violation,
    backward_violation,
    project_valid_box,
    repair_pair,
    compute_pair_stats,
    mean_box_displacement,
    mean_box_size,
    get_current_violated,
    compute_iteration_snapshot,
    evaluate_run,
    format_hparams,
    write_csv,
    save_pair_rows,
    run_no_refinement,
    run_coord_descent,
    run_hybrid,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results_cegar_chunk2")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_concepts", type=int, default=5000)
    parser.add_argument("--gap_sample", type=int, default=400)
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
    plt.title("Chunk 2: Convergence on full constraint set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_fix_vs_new(path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 6))
    for row in rows:
        label = f"{row['method']}\n({format_hparams(row['hyperparameters'])})"
        plt.scatter(row["new_violations"], row["fixed_count"], s=80)
        plt.text(row["new_violations"] + 0.5, row["fixed_count"] + 0.5, label, fontsize=7)
    plt.xlabel("New violations")
    plt.ylabel("Fixed count")
    plt.title("Chunk 2: Fixes vs introduced violations (full constraint set)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_iteration_details(path: Path, convergence_logs: dict[str, list[dict[str, object]]]) -> None:
    """Multi-panel convergence plot: violation rate, net improvement, displacement, Galois gap."""
    if plt is None or not convergence_logs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("final_violation_rate", "Violation rate"),
        ("net_improvement", "Net improvement (cumulative)"),
        ("mean_box_displacement", "Mean box displacement"),
        ("galois_gap_change", "Galois gap change"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        for label, history in convergence_logs.items():
            xs = [row["iteration"] for row in history]
            ys = [row.get(key, 0) for row in history]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Chunk 2: Full constraint set convergence", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_paths = ensure_dirs(Path(args.output_dir))
    rng = np.random.default_rng(args.seed)

    print(f"\n{'=' * 88}")
    print("  Chunk 2: Full-Scale Evaluation of Chunk-1 Winners")
    print(f"{'=' * 88}")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {output_paths['root']}")

    # ---- Load model and embeddings ----
    print("\n[1/5] Loading cached noun-trained model and GloVe...")
    glove = api.load("glove-wiki-gigaword-300")
    model_cache = Path("results/inference_cegar_model.npz")
    if not model_cache.exists():
        raise FileNotFoundError("Missing results/inference_cegar_model.npz. Run inference_cegar.py first.")

    model_weights = np.load(model_cache, allow_pickle=True)
    model = GaloisModel(300, HIDDEN_DIM, BOX_DIM, np.random.default_rng(args.seed))
    for key in model.params_dict():
        setattr(model, key, model_weights[key])

    # ---- Build FULL verb constraint set (no sub-sampling) ----
    print("\n[2/5] Building FULL verb domain-shift constraint set (no sub-sampling)...")
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

    # Use ALL constraints -- this is the key difference from Chunk 1
    constraints = dict(all_constraints)

    gap_pairs = list(constraints.keys())[: min(args.gap_sample, len(constraints))]
    base_gap = measure_galois_consistency_gap(boxes, embeddings, model, gap_pairs, n_sample=len(gap_pairs))["mean"]
    box_size_scale = mean_box_size(boxes)

    baseline_stats = compute_pair_stats(boxes, boxes, constraints)
    print(f"  FULL constraint set: {len(constraints)} constraints (ALL)")
    print(f"  Original violations: {baseline_stats['original_violations']}/{baseline_stats['total_pairs']}")
    print(f"  Base Galois gap: {base_gap:.6f}")
    print(f"  Mean box size: {box_size_scale:.6f}")

    context = {
        "embeddings": embeddings,
        "model": model,
        "gap_pairs": gap_pairs,
        "box_size_scale": box_size_scale,
    }

    # ---- Run the two Chunk-1 winners + baseline ----
    print("\n[3/5] Running Chunk-1 winners on full constraint set...")
    results: list[dict[str, object]] = []

    def record_run(method: str, hparams: dict[str, object], boxes_after: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
        row = evaluate_run(method, hparams, boxes, boxes_after, constraints, embeddings, model, gap_pairs, base_gap, box_size_scale)
        results.append(row)
        print(
            f"  {method:24s} {format_hparams(hparams):55s} "
            f"fix={row['fixed_count']:4d}/{row['original_violations']:4d} ({100 * row['fix_rate']:5.1f}%) "
            f"new={row['new_violations']:4d} net={row['net_improvement']:+5d} "
            f"gap={row['Galois_gap_change']:+.4f} disp={row['mean_box_displacement']:.4f} "
            f"S={row['aggregate_score']:+.4f}"
        )
        return row

    # Baseline
    record_run("No refinement", {}, copy_boxes(boxes))

    # Winner 1: Coordinate descent (step_size=0.3, iterations=15)
    t0 = time.time()
    boxes_coord, _ = run_coord_descent(boxes, constraints, 15, 0.3, False, **context)
    t_coord = time.time() - t0
    row_coord = record_run("Coordinate descent", {"step_size": 0.3, "iterations": 15}, boxes_coord)
    print(f"    -> wall time: {t_coord:.1f}s")

    # Also try iterations=20 to see if more iterations help at full scale
    t0 = time.time()
    boxes_coord20, _ = run_coord_descent(boxes, constraints, 20, 0.3, False, **context)
    t_coord20 = time.time() - t0
    row_coord20 = record_run("Coordinate descent", {"step_size": 0.3, "iterations": 20}, boxes_coord20)
    print(f"    -> wall time: {t_coord20:.1f}s")

    # Winner 2: Hybrid warm-start (step_size=0.3, iterations=15, warm_start_iters=5, lambda_p=1.0, lambda_a=0.0)
    t0 = time.time()
    boxes_hybrid, _ = run_hybrid(
        boxes, constraints, 15, 0.3, 5, 1.0, 0.0, TOP_FRACTION, False, **context,
    )
    t_hybrid = time.time() - t0
    row_hybrid = record_run(
        "Hybrid warm-start",
        {"step_size": 0.3, "iterations": 15, "warm_start_iters": 5,
         "lambda_p": 1.0, "lambda_a": 0.0, "top_fraction": TOP_FRACTION},
        boxes_hybrid,
    )
    print(f"    -> wall time: {t_hybrid:.1f}s")

    # Also try Hybrid with iterations=20
    t0 = time.time()
    boxes_hybrid20, _ = run_hybrid(
        boxes, constraints, 20, 0.3, 5, 1.0, 0.0, TOP_FRACTION, False, **context,
    )
    t_hybrid20 = time.time() - t0
    row_hybrid20 = record_run(
        "Hybrid warm-start",
        {"step_size": 0.3, "iterations": 20, "warm_start_iters": 5,
         "lambda_p": 1.0, "lambda_a": 0.0, "top_fraction": TOP_FRACTION},
        boxes_hybrid20,
    )
    print(f"    -> wall time: {t_hybrid20:.1f}s")

    # ---- Convergence traces for top runs ----
    print("\n[4/5] Generating convergence traces and outputs...")
    ranked = sorted(results, key=lambda row: row["aggregate_score"], reverse=True)
    write_csv(output_paths["root"] / "main_table.csv", ranked)

    convergence_logs = {}
    for index, row in enumerate(ranked, start=1):
        method = row["method"]
        hparams = row["hyperparameters"]
        if method == "No refinement":
            final_boxes, history = run_no_refinement(boxes)
        elif method == "Coordinate descent":
            final_boxes, history = run_coord_descent(
                boxes, constraints, int(hparams["iterations"]), float(hparams["step_size"]), True, **context,
            )
        elif method == "Hybrid warm-start":
            final_boxes, history = run_hybrid(
                boxes, constraints, int(hparams["iterations"]), float(hparams["step_size"]),
                int(hparams["warm_start_iters"]), float(hparams["lambda_p"]),
                float(hparams["lambda_a"]), TOP_FRACTION, True, **context,
            )
        else:
            final_boxes, history = run_no_refinement(boxes)

        for item in history:
            item["galois_gap_change"] = item["galois_gap"] - base_gap

        label = f"{index}_{method}_{format_hparams(hparams).replace(',', '_').replace('=', '-')}"
        convergence_logs[label] = history
        (output_paths["logs"] / f"{label}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        pair_rows = compute_pair_stats(boxes, final_boxes, constraints)["pair_rows"]
        save_pair_rows(output_paths["logs"] / f"{label}_pairs.csv", pair_rows)

    plot_convergence(output_paths["plots"] / "convergence_top3.png", convergence_logs)
    plot_fix_vs_new(output_paths["plots"] / "fix_vs_newviolations.png", ranked)
    plot_iteration_details(output_paths["plots"] / "convergence_details.png", convergence_logs)

    # ---- Summary ----
    best_fix = max(ranked, key=lambda row: row["fix_rate"])
    best_net = max(ranked, key=lambda row: row["net_improvement"])
    best_geom = min(ranked, key=lambda row: max(row["Galois_gap_change"], 0.0) + row["mean_box_displacement"])
    best_overall = ranked[0]

    best_coord_row = max(
        [row for row in ranked if row["method"] == "Coordinate descent"],
        key=lambda row: row["aggregate_score"], default=None,
    )
    best_hybrid_row = max(
        [row for row in ranked if row["method"] == "Hybrid warm-start"],
        key=lambda row: row["aggregate_score"], default=None,
    )

    coord_beats_hybrid = (
        best_coord_row is not None
        and best_hybrid_row is not None
        and best_coord_row["aggregate_score"] > best_hybrid_row["aggregate_score"]
    )
    hybrid_fewer_new = (
        best_hybrid_row is not None
        and best_coord_row is not None
        and best_hybrid_row["new_violations"] < best_coord_row["new_violations"]
    )
    hybrid_less_displacement = (
        best_hybrid_row is not None
        and best_coord_row is not None
        and best_hybrid_row["mean_box_displacement"] < best_coord_row["mean_box_displacement"]
    )

    # Final recommendation
    if best_overall["fix_rate"] >= 0.95 and best_overall["new_violations"] <= 5:
        verdict = "STRONG: recommended for production use"
    elif best_overall["fix_rate"] >= 0.85:
        verdict = "GOOD: usable with minor caveats"
    else:
        verdict = "WEAK: needs further investigation"

    summary = f"""# Chunk 2: Full-Scale Evaluation Summary

All {len(constraints)} verb IS-A constraints used (no sub-sampling).

## Winners

- Best method on raw fix rate: {best_fix['method']} ({format_hparams(best_fix['hyperparameters'])})
- Best method on net improvement: {best_net['method']} ({format_hparams(best_net['hyperparameters'])})
- Best method on geometry preservation: {best_geom['method']} ({format_hparams(best_geom['hyperparameters'])})
- Best overall tradeoff: {best_overall['method']} ({format_hparams(best_overall['hyperparameters'])})

## Head-to-Head

- Does coordinate descent beat hybrid on aggregate score? {'Yes' if coord_beats_hybrid else 'No'}
- Does hybrid produce fewer new violations? {'Yes' if hybrid_fewer_new else 'No'}
- Does hybrid have less box displacement? {'Yes' if hybrid_less_displacement else 'No'}

## Key Numbers (Best Overall: {best_overall['method']})

- Fix rate: {100 * best_overall['fix_rate']:.1f}%
- New violations: {best_overall['new_violations']}
- Net improvement: {best_overall['net_improvement']}
- Final violation rate: {100 * best_overall['final_violation_rate']:.1f}%
- Galois gap change: {best_overall['Galois_gap_change']:+.6f}
- Mean box displacement: {best_overall['mean_box_displacement']:.6f}
- Aggregate score: {best_overall['aggregate_score']:+.4f}

## Verdict

{verdict}

## Recommended Configuration

Method: {best_overall['method']}
Hyperparameters: {format_hparams(best_overall['hyperparameters'])}
"""
    (output_paths["root"] / "summary.md").write_text(summary, encoding="utf-8")

    (output_paths["configs"] / "experiment_config.json").write_text(
        json.dumps({
            "seed": args.seed,
            "n_concepts": args.n_concepts,
            "gap_sample": args.gap_sample,
            "total_constraints": len(constraints),
            "base_gap": base_gap,
            "chunk": 2,
            "methods_tested": ["Coordinate descent", "Hybrid warm-start"],
            "wall_times": {
                "coord_15": t_coord,
                "coord_20": t_coord20,
                "hybrid_15": t_hybrid,
                "hybrid_20": t_hybrid20,
            },
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\n[5/5] Done.")
    print(f"  Main table: {output_paths['root'] / 'main_table.csv'}")
    print(f"  Summary: {output_paths['root'] / 'summary.md'}")
    print(f"  Best overall: {best_overall['method']} ({format_hparams(best_overall['hyperparameters'])})")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
