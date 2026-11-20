"""
Service helpers for the local training and inference web interface.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.box_lattice import box_containment_score, box_containment_violation
from src.dataset import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    NON_ENTAILMENT_PAIRS,
    ENTAILMENT_PAIRS,
    SENTENCES,
    build_synthetic_dataset,
    build_synthetic_embeddings,
)
from src.improved_trainer import ImprovedTrainer
from src.metrics import compute_nli_metrics


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "webui_models"

LABEL_NAMES = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def default_training_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "n_train": 3000,
        "n_val": 500,
        "n_test": 1000,
        "input_dim": 64,
        "box_dim": 64,
        "hidden_dim": 256,
        "n_epochs": 20,
        "batch_size": 64,
        "lr": 1e-3,
        "noise": 0.05,
        "lambda1": 0.5,
        "lambda2": 2.0,
        "lambda3": 1.0,
        "lambda4": 0.3,
        "lambda5": 0.0,
        "margin": 1.0,
        "scale": 5.0,
        "cegar_margin": 1.5,
        "warmup_epochs": 0,
        "cegar_max_size": 2000,
        "flush_every": 0,
        # 3-class directional loss margins (active when lambda5 > 0)
        "ent_bwd_target": 10.0,
        "neu_fwd_margin": 3.0,
        "neu_bwd_ceiling": 6.0,
        "con_fwd_margin": 5.0,
        "overlap_weight": 0.1,
        "patience": 8,
    }


def _ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = default_training_config()
    if overrides:
        config.update(overrides)
    int_keys = [
        "seed",
        "n_train",
        "n_val",
        "n_test",
        "input_dim",
        "box_dim",
        "hidden_dim",
        "n_epochs",
        "batch_size",
        "warmup_epochs",
        "cegar_max_size",
        "flush_every",
        "patience",
    ]
    float_keys = [
        "lr",
        "noise",
        "lambda1",
        "lambda2",
        "lambda3",
        "lambda4",
        "lambda5",
        "margin",
        "scale",
        "cegar_margin",
        "ent_bwd_target",
        "neu_fwd_margin",
        "neu_bwd_ceiling",
        "con_fwd_margin",
        "overlap_weight",
    ]
    for key in int_keys:
        config[key] = int(config[key])
    for key in float_keys:
        config[key] = float(config[key])
    return config


def _make_trainer(config: dict[str, Any]) -> ImprovedTrainer:
    return ImprovedTrainer(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        box_dim=config["box_dim"],
        lr=config["lr"],
        lambda1=config["lambda1"],
        lambda2=config["lambda2"],
        lambda3=config["lambda3"],
        lambda4=config["lambda4"],
        lambda5=config["lambda5"],
        margin=config["margin"],
        scale=config["scale"],
        cegar_margin=config["cegar_margin"],
        warmup_epochs=config["warmup_epochs"],
        cegar_max_size=config["cegar_max_size"],
        flush_every=config["flush_every"],
        ent_bwd_target=config["ent_bwd_target"],
        neu_fwd_margin=config["neu_fwd_margin"],
        neu_bwd_ceiling=config["neu_bwd_ceiling"],
        con_fwd_margin=config["con_fwd_margin"],
        overlap_weight=config["overlap_weight"],
        rng=np.random.default_rng(config["seed"]),
    )


def _trainer_params(trainer: ImprovedTrainer) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in trainer.model.params_dict().items()}


def _model_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def save_model(
    trainer: ImprovedTrainer,
    config: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> Path:
    _ensure_output_dirs()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"galois_numpy_{stamp}.npz"
    params = _trainer_params(trainer)
    np.savez_compressed(model_path, **params)

    metadata = {
        "model_path": str(model_path),
        "config": config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _model_metadata_path(model_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def get_latest_model_path() -> Path | None:
    _ensure_output_dirs()
    candidates = sorted(MODELS_DIR.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def get_latest_model_info() -> dict[str, Any]:
    model_path = get_latest_model_path()
    if model_path is None:
        return {"available": False}
    metadata_path = _model_metadata_path(model_path)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "model_path": str(model_path),
        "metadata": metadata,
    }


def load_model(model_path: str | None = None) -> tuple[ImprovedTrainer, dict[str, Any], Path]:
    resolved = Path(model_path) if model_path else get_latest_model_path()
    if resolved is None or not resolved.exists():
        raise FileNotFoundError("No saved model was found. Train a model first.")

    metadata_path = _model_metadata_path(resolved)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file for model: {resolved}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = _coerce_config(metadata["config"])
    trainer = _make_trainer(config)

    with np.load(resolved) as weights:
        params = {name: weights[name] for name in trainer.model.params_dict()}
    trainer.model.set_params(params)
    return trainer, metadata, resolved


def train_synthetic_model(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = _coerce_config(config_overrides)
    rng = np.random.default_rng(config["seed"])
    train_data, val_data, test_data = build_synthetic_dataset(
        n_train=config["n_train"],
        n_val=config["n_val"],
        n_test=config["n_test"],
        dim=config["input_dim"],
        noise=config["noise"],
        rng=rng,
    )

    trainer = _make_trainer(config)
    started = time.time()
    history = trainer.train(
        train_data,
        val_data,
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        verbose=False,
        patience=config["patience"],
    )
    elapsed = round(time.time() - started, 2)
    val_metrics = trainer.evaluate(val_data)
    test_metrics = trainer.evaluate(test_data)
    model_path = save_model(trainer, config, val_metrics, test_metrics, history)

    label_counts = np.bincount(train_data["label"], minlength=3)
    return {
        "model_path": str(model_path),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_time_seconds": elapsed,
        "config": config,
        "dataset_summary": {
            "train_size": len(train_data["label"]),
            "val_size": len(val_data["label"]),
            "test_size": len(test_data["label"]),
            "label_counts": {
                "entailment": int(label_counts[0]),
                "neutral": int(label_counts[1]),
                "contradiction": int(label_counts[2]),
            },
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def _binary_confusion(binary_true: np.ndarray, binary_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((binary_true == 1) & (binary_pred == 1))),
        "fp": int(np.sum((binary_true == 0) & (binary_pred == 1))),
        "fn": int(np.sum((binary_true == 1) & (binary_pred == 0))),
        "tn": int(np.sum((binary_true == 0) & (binary_pred == 0))),
    }


def _preview_rows(
    dataset: dict[str, Any],
    scores: np.ndarray,
    threshold: float,
    violations: np.ndarray,
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(min(limit, len(scores))):
        true_label = int(dataset["label"][index])
        rows.append(
            {
                "index": index,
                "sentence_1": dataset["sent1"][index],
                "sentence_2": dataset["sent2"][index],
                "true_label": LABEL_NAMES[true_label],
                "entailment_score": round(float(scores[index]), 4),
                "predicted_binary_label": "entailment" if float(scores[index]) >= threshold else "non-entailment",
                "containment_violation": round(float(violations[index]), 4),
            }
        )
    return rows


def _build_matched_synthetic_split(
    n_samples: int,
    dim: int,
    noise: float,
    base_seed: int,
    sample_seed: int,
) -> dict[str, Any]:
    base_rng = np.random.default_rng(base_seed)
    sample_rng = np.random.default_rng(sample_seed)
    base_emb = build_synthetic_embeddings(dim=dim, noise=0.0, rng=base_rng)

    emb1_list, emb2_list, label_list, s1_list, s2_list = [], [], [], [], []
    n_ent = n_samples // 3
    n_neu = n_samples // 3
    n_con = n_samples - n_ent - n_neu

    ent_indices = sample_rng.integers(0, len(ENTAILMENT_PAIRS), n_ent)
    for idx in ent_indices:
        sentence_1_id, sentence_2_id = ENTAILMENT_PAIRS[idx]
        emb1_list.append(base_emb[sentence_1_id] + noise * sample_rng.standard_normal(dim))
        emb2_list.append(base_emb[sentence_2_id] + noise * sample_rng.standard_normal(dim))
        label_list.append(ENTAILMENT)
        s1_list.append(SENTENCES[sentence_1_id])
        s2_list.append(SENTENCES[sentence_2_id])

    sentence_ids = list(SENTENCES.keys())
    for _ in range(n_neu):
        sentence_1_id = sample_rng.choice(sentence_ids)
        sentence_2_id = sample_rng.choice(sentence_ids)
        while (sentence_1_id, sentence_2_id) in ENTAILMENT_PAIRS or sentence_1_id == sentence_2_id:
            sentence_2_id = sample_rng.choice(sentence_ids)
        emb1_list.append(base_emb[sentence_1_id] + noise * sample_rng.standard_normal(dim))
        emb2_list.append(base_emb[sentence_2_id] + noise * sample_rng.standard_normal(dim))
        label_list.append(NEUTRAL)
        s1_list.append(SENTENCES[sentence_1_id])
        s2_list.append(SENTENCES[sentence_2_id])

    non_ent_indices = sample_rng.integers(0, len(NON_ENTAILMENT_PAIRS), n_con)
    for idx in non_ent_indices:
        sentence_1_id, sentence_2_id = NON_ENTAILMENT_PAIRS[idx]
        emb1_list.append(base_emb[sentence_1_id] + noise * sample_rng.standard_normal(dim))
        emb2_list.append(-base_emb[sentence_2_id] + noise * sample_rng.standard_normal(dim))
        label_list.append(CONTRADICTION)
        s1_list.append(SENTENCES[sentence_1_id])
        s2_list.append(SENTENCES[sentence_2_id])

    order = sample_rng.permutation(len(label_list))
    return {
        "emb1": np.array(emb1_list)[order],
        "emb2": np.array(emb2_list)[order],
        "label": np.array(label_list)[order],
        "sent1": [s1_list[index] for index in order],
        "sent2": [s2_list[index] for index in order],
    }


def infer_synthetic_dataset(
    model_path: str | None = None,
    n_samples: int = 120,
    seed: int = 123,
    noise: float = 0.05,
) -> dict[str, Any]:
    trainer, metadata, resolved = load_model(model_path)
    config = _coerce_config(metadata["config"])
    dataset = _build_matched_synthetic_split(
        n_samples=int(n_samples),
        dim=config["input_dim"],
        noise=float(noise),
        base_seed=config["seed"],
        sample_seed=int(seed),
    )

    l1, u1, _ = trainer.model.encode(dataset["emb1"])
    l2, u2, _ = trainer.model.encode(dataset["emb2"])
    scores = box_containment_score(l1, u1, l2, u2, scale=trainer.scale)
    violations = box_containment_violation(l1, u1, l2, u2)

    threshold = float(metadata.get("val_metrics", {}).get("best_thr", 0.5))
    metrics = compute_nli_metrics(scores, dataset["label"], threshold=threshold)
    binary_true = (dataset["label"] == 0).astype(int)
    binary_pred = (scores >= threshold).astype(int)

    return {
        "model_path": str(resolved),
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": _binary_confusion(binary_true, binary_pred),
        "samples": _preview_rows(dataset, scores, threshold, violations),
    }


def list_sentence_catalog() -> list[dict[str, str]]:
    return [{"id": key, "text": value} for key, value in SENTENCES.items()]


def infer_custom_pair(
    sentence_1_id: str,
    sentence_2_id: str,
    model_path: str | None = None,
) -> dict[str, Any]:
    trainer, metadata, resolved = load_model(model_path)
    config = _coerce_config(metadata["config"])
    rng = np.random.default_rng(config["seed"])
    embeddings = build_synthetic_embeddings(dim=config["input_dim"], noise=0.0, rng=rng)

    if sentence_1_id not in embeddings or sentence_2_id not in embeddings:
        raise KeyError("Unknown synthetic sentence identifier.")

    emb1 = embeddings[sentence_1_id][None, :]
    emb2 = embeddings[sentence_2_id][None, :]

    l1, u1, _ = trainer.model.encode(emb1)
    l2, u2, _ = trainer.model.encode(emb2)
    score = float(box_containment_score(l1, u1, l2, u2, scale=trainer.scale)[0])
    violation = float(box_containment_violation(l1, u1, l2, u2)[0])
    threshold = float(metadata.get("val_metrics", {}).get("best_thr", 0.5))

    return {
        "model_path": str(resolved),
        "threshold": threshold,
        "sentence_1": {"id": sentence_1_id, "text": SENTENCES[sentence_1_id]},
        "sentence_2": {"id": sentence_2_id, "text": SENTENCES[sentence_2_id]},
        "entailment_score": score,
        "predicted_binary_label": "entailment" if score >= threshold else "non-entailment",
        "containment_violation": violation,
        "box_width_mean_sentence_1": float(np.mean(u1 - l1)),
        "box_width_mean_sentence_2": float(np.mean(u2 - l2)),
    }
