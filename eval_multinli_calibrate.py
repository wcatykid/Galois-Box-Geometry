"""
Threshold-only MultiNLI recalibration for the already-trained Galois v2 model.

Workflow:
1. Load MultiNLI matched dev and hold out 100 calibration examples.
2. Load the frozen Galois v2 checkpoint and encode calibration + remainder.
3. Search binary entailment thresholds from 0.05 to 0.95 by 0.05.
4. Evaluate the remainder with the original SNLI threshold and the new MultiNLI threshold.

3-class metrics are reported with the same fixed SNLI-trained probe on the
remainder split. Because the prompt asks for threshold search only, these
3-class metrics are identical across the two threshold columns.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.metrics import ThreeClassProbe, compute_3class_features


HF_DATASET_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"
MULTINLI_ARROW = HF_DATASET_CACHE / "multi_nli" / "default" / "0.0.0" / "da70db2af9d09693783c3320c4249840212ee221" / "multi_nli-validation_matched.arrow"
SNLI_ARROW_DIR = HF_DATASET_CACHE / "snli" / "plain_text" / "0.0.0" / "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"


@dataclass
class Config:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    box_dim: int = 128
    hidden_dim: int = 512
    freeze_layers: int = 3
    max_len: int = 32
    scale: float = 5.0
    seed: int = 42
    batch_size: int = 64


class BoxEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, box_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.center_head = nn.Linear(hidden_dim, box_dim)
        self.width_head = nn.Linear(hidden_dim, box_dim)
        nn.init.xavier_uniform_(self.center_head.weight, gain=0.1)
        nn.init.xavier_uniform_(self.width_head.weight, gain=0.1)
        nn.init.constant_(self.width_head.bias, -1.0)

    def forward(self, x):
        h = self.net(x)
        c = self.center_head(h)
        hw = F.softplus(self.width_head(h))
        return c - hw, c + hw


class GaloisSBERTModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model_name)
        self.box_enc = BoxEncoder(self.bert.config.hidden_size, cfg.hidden_dim, cfg.box_dim)
        self.scale = cfg.scale
        self._apply_freeze(cfg.freeze_layers)

    def _apply_freeze(self, n_layers: int) -> None:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for index, layer in enumerate(self.bert.encoder.layer):
            if index < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    @staticmethod
    def _mean_pool(last_hidden, attn_mask):
        mask_exp = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return torch.sum(last_hidden * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)

    def encode_sentence(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        h = self._mean_pool(out.last_hidden_state, mask)
        return self.box_enc(h)

    def forward(self, ids1, mask1, ids2, mask2):
        l1, u1 = self.encode_sentence(ids1, mask1)
        l2, u2 = self.encode_sentence(ids2, mask2)
        return l1, u1, l2, u2


class NLIPairDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_len: int):
        self.labels = np.asarray(labels, dtype=np.int64)
        enc1 = tokenizer(
            list(premises),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="np",
        )
        enc2 = tokenizer(
            list(hypotheses),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="np",
        )
        self.ids1 = torch.from_numpy(enc1["input_ids"].astype(np.int64))
        self.mask1 = torch.from_numpy(enc1["attention_mask"].astype(np.int64))
        self.ids2 = torch.from_numpy(enc2["input_ids"].astype(np.int64))
        self.mask2 = torch.from_numpy(enc2["attention_mask"].astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids1[index], self.mask1[index], self.ids2[index], self.mask2[index], self.labels[index]


@torch.no_grad()
def encode_all(model, loader, device):
    l1_all, u1_all, l2_all, u2_all, labels_all = [], [], [], [], []
    for ids1, mask1, ids2, mask2, labels in loader:
        l1, u1, l2, u2 = model(ids1.to(device), mask1.to(device), ids2.to(device), mask2.to(device))
        l1_all.append(l1.cpu().numpy())
        u1_all.append(u1.cpu().numpy())
        l2_all.append(l2.cpu().numpy())
        u2_all.append(u2.cpu().numpy())
        labels_all.append(labels.numpy())
    return (
        np.concatenate(l1_all),
        np.concatenate(u1_all),
        np.concatenate(l2_all),
        np.concatenate(u2_all),
        np.concatenate(labels_all),
    )


def compute_p_entail(l1, u1, l2, u2, scale: float = 5.0):
    viol = (np.maximum(0.0, l2 - l1) + np.maximum(0.0, u1 - u2)).sum(axis=-1)
    return 1.0 / (1.0 + scale * viol)


def eval_binary_at_threshold(p_ent, labels, threshold: float):
    binary_labels = (labels == 0).astype(int)
    pred_labels = (p_ent >= threshold).astype(int)

    tp = np.sum((pred_labels == 1) & (binary_labels == 1))
    fp = np.sum((pred_labels == 1) & (binary_labels == 0))
    fn = np.sum((pred_labels == 0) & (binary_labels == 1))
    tn = np.sum((pred_labels == 0) & (binary_labels == 0))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(p_ent, labels, lo: float = 0.05, hi: float = 0.95, step: float = 0.05):
    binary_labels = (labels == 0).astype(int)
    best_thr = 0.5
    best_f1 = -1.0
    search_rows = []

    for thr in np.arange(lo, hi + 1e-9, step):
        pred = (p_ent >= thr).astype(int)
        tp = np.sum((pred == 1) & (binary_labels == 1))
        fp = np.sum((pred == 1) & (binary_labels == 0))
        fn = np.sum((pred == 0) & (binary_labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        row = {
            "threshold": round(float(thr), 2),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
        search_rows.append(row)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1, search_rows


def fit_fixed_snli_probe(model, tokenizer, device, cfg: Config):
    raw = load_snli_train_cached(limit=12000)
    ds = NLIPairDataset(raw["premise"], raw["hypothesis"], raw["label"].tolist(), tokenizer, cfg.max_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    l1, u1, l2, u2, labels = encode_all(model, loader, device)
    features = compute_3class_features(l1, u1, l2, u2)
    probe = ThreeClassProbe(n_epochs=400, lr=0.05)
    probe.fit(features, labels)
    return probe


def build_stratified_calibration_indices(labels_all: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    targets = {0: 34, 1: 33, 2: 33}
    calibration = []
    for label in [0, 1, 2]:
        class_indices = np.where(labels_all == label)[0]
        chosen = rng.choice(class_indices, size=targets[label], replace=False)
        calibration.extend(chosen.tolist())
    calibration = sorted(calibration)
    evaluation = sorted(set(range(len(labels_all))) - set(calibration))
    return calibration, evaluation


def load_arrow_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        return reader.read_all()


def load_multinli_matched_cached():
    table = load_arrow_table(MULTINLI_ARROW)
    data = table.to_pydict()
    labels = np.asarray(data["label"])
    keep = labels != -1
    return {
        "premise": [value for value, flag in zip(data["premise"], keep) if flag],
        "hypothesis": [value for value, flag in zip(data["hypothesis"], keep) if flag],
        "label": labels[keep].astype(np.int64),
        "genre": [value for value, flag in zip(data["genre"], keep) if flag],
    }


def load_snli_train_cached(limit: int = 12000):
    table = load_arrow_table(SNLI_ARROW_DIR / "snli-train.arrow").slice(0, limit)
    data = table.to_pydict()
    labels = np.asarray(data["label"])
    keep = labels != -1
    return {
        "premise": [value for value, flag in zip(data["premise"], keep) if flag],
        "hypothesis": [value for value, flag in zip(data["hypothesis"], keep) if flag],
        "label": labels[keep].astype(np.int64),
    }


def print_table(snli_bin, mnli_bin, fixed_3class):
    print(f"\n  {'Metric':<18} {'SNLI threshold':>16} {'MultiNLI threshold':>20} {'Delta':>10}")
    print(f"  {'-' * 18} {'-' * 16} {'-' * 20} {'-' * 10}")
    print(
        f"  {'Bin-Acc':<18} {snli_bin['accuracy']:>16.3f} {mnli_bin['accuracy']:>20.3f} "
        f"{mnli_bin['accuracy'] - snli_bin['accuracy']:>+10.3f}"
    )
    print(
        f"  {'Bin-F1':<18} {snli_bin['f1']:>16.3f} {mnli_bin['f1']:>20.3f} "
        f"{mnli_bin['f1'] - snli_bin['f1']:>+10.3f}"
    )
    print(
        f"  {'3-class Acc':<18} {fixed_3class['acc_3class']:>16.3f} {fixed_3class['acc_3class']:>20.3f} "
        f"{0.0:>+10.3f}"
    )
    print(
        f"  {'Entailment recall':<18} {fixed_3class['per_class']['entail']:>16.3f} {fixed_3class['per_class']['entail']:>20.3f} "
        f"{0.0:>+10.3f}"
    )
    print(
        f"  {'Neutral recall':<18} {fixed_3class['per_class']['neutral']:>16.3f} {fixed_3class['per_class']['neutral']:>20.3f} "
        f"{0.0:>+10.3f}"
    )
    print(
        f"  {'Contra recall':<18} {fixed_3class['per_class']['contra']:>16.3f} {fixed_3class['per_class']['contra']:>20.3f} "
        f"{0.0:>+10.3f}"
    )


def main():
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n{'=' * 74}")
    print("  MultiNLI Threshold Calibration - Galois v2")
    print(f"{'=' * 74}")
    print(f"  Device: {device}")

    results_dir = Path("results")
    with (results_dir / "torch_models_v2" / "results.json").open(encoding="utf-8") as handle:
        snli_results = json.load(handle)
    with (results_dir / "multinli_evaluation.json").open(encoding="utf-8") as handle:
        previous_multinli = json.load(handle)

    print("\n[1/6] Loading MultiNLI matched dev set from local cache ...")
    raw = load_multinli_matched_cached()
    print(f"      Total examples after filtering label=-1: {len(raw['label'])}")

    labels_all = np.asarray(raw["label"])
    cal_indices, eval_indices = build_stratified_calibration_indices(labels_all, cfg.seed)
    cal_labels = labels_all[cal_indices]
    eval_labels = labels_all[eval_indices]

    print("\n[2/6] Building calibration and evaluation splits ...")
    print(f"      Calibration set: {len(cal_indices)} examples")
    print(f"        entail={int(np.sum(cal_labels == 0))} neutral={int(np.sum(cal_labels == 1))} contra={int(np.sum(cal_labels == 2))}")
    print(f"      Evaluation set:  {len(eval_indices)} examples")

    print("\n[3/6] Loading trained Galois v2 model ...")
    model = GaloisSBERTModel(cfg)
    checkpoint = torch.load(results_dir / "torch_models_v2" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    snli_best_thr = float(snli_results["test_binary"]["best_thr"])
    print(f"      Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"      Original SNLI-calibrated threshold: {snli_best_thr:.4f}")

    print("\n[4/6] Encoding calibration and evaluation examples ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    premises = raw["premise"]
    hypotheses = raw["hypothesis"]

    t0 = time.time()
    cal_ds = NLIPairDataset(
        [premises[i] for i in cal_indices],
        [hypotheses[i] for i in cal_indices],
        cal_labels.tolist(),
        tokenizer,
        cfg.max_len,
    )
    eval_ds = NLIPairDataset(
        [premises[i] for i in eval_indices],
        [hypotheses[i] for i in eval_indices],
        eval_labels.tolist(),
        tokenizer,
        cfg.max_len,
    )
    cal_loader = DataLoader(cal_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    l1_cal, u1_cal, l2_cal, u2_cal, labs_cal = encode_all(model, cal_loader, device)
    l1_eval, u1_eval, l2_eval, u2_eval, labs_eval = encode_all(model, eval_loader, device)
    print(f"      Encoding done in {time.time() - t0:.1f}s")

    print("\n[5/6] Threshold search on the 100-example calibration set ...")
    p_ent_cal = compute_p_entail(l1_cal, u1_cal, l2_cal, u2_cal, cfg.scale)
    mnli_thr, cal_best_f1, search_rows = find_optimal_threshold(p_ent_cal, labs_cal)
    print(f"      Best threshold: {mnli_thr:.2f}")
    print(f"      Best calibration F1: {cal_best_f1:.4f}")
    print(f"      Calibration examples needed: {len(cal_indices)}")

    p_ent_eval = compute_p_entail(l1_eval, u1_eval, l2_eval, u2_eval, cfg.scale)
    snli_binary = eval_binary_at_threshold(p_ent_eval, labs_eval, snli_best_thr)
    mnli_binary = eval_binary_at_threshold(p_ent_eval, labs_eval, mnli_thr)

    print("\n[6/6] Recreating the fixed SNLI-trained 3-class probe for the remainder split ...")
    probe = fit_fixed_snli_probe(model, tokenizer, device, cfg)
    eval_features = compute_3class_features(l1_eval, u1_eval, l2_eval, u2_eval)
    fixed_3class = probe.evaluate(eval_features, labs_eval)

    print(f"\n{'=' * 74}")
    print("  Comparison Table")
    print(f"{'=' * 74}")
    print(f"  Calibrated threshold value: {mnli_thr:.2f}")
    print(f"  Calibration examples used:  {len(cal_indices)}")
    print(f"  Evaluation remainder size:  {len(eval_indices)}")
    print("  3-class metrics use the same fixed SNLI-trained probe under both thresholds.")
    print_table(snli_binary, mnli_binary, fixed_3class)

    output = {
        "calibration": {
            "n_calibration": len(cal_indices),
            "n_evaluation": len(eval_indices),
            "class_counts": {
                "entail": int(np.sum(cal_labels == 0)),
                "neutral": int(np.sum(cal_labels == 1)),
                "contra": int(np.sum(cal_labels == 2)),
            },
            "snli_threshold": snli_best_thr,
            "mnli_threshold": mnli_thr,
            "best_calibration_f1": cal_best_f1,
            "threshold_search": search_rows,
        },
        "evaluation": {
            "snli_threshold": {
                "binary": snli_binary,
                "three_class": {
                    "acc_3class": float(fixed_3class["acc_3class"]),
                    "per_class": {key: float(value) for key, value in fixed_3class["per_class"].items()},
                    "confusion": fixed_3class["confusion"].tolist(),
                },
            },
            "mnli_threshold": {
                "binary": mnli_binary,
                "three_class": {
                    "acc_3class": float(fixed_3class["acc_3class"]),
                    "per_class": {key: float(value) for key, value in fixed_3class["per_class"].items()},
                    "confusion": fixed_3class["confusion"].tolist(),
                },
            },
        },
        "delta": {
            "bin_acc": float(mnli_binary["accuracy"] - snli_binary["accuracy"]),
            "bin_f1": float(mnli_binary["f1"] - snli_binary["f1"]),
            "acc_3class": 0.0,
            "entail_recall": 0.0,
            "neutral_recall": 0.0,
            "contra_recall": 0.0,
        },
        "provenance": {
            "model_checkpoint": "results/torch_models_v2/best_model.pt",
            "previous_multinli_results": "results/multinli_evaluation.json",
            "previous_full_multinli_3class": previous_multinli["galois_v2"]["multinli"],
        },
    }

    save_path = results_dir / "multinli_calibrated.json"
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    main()
