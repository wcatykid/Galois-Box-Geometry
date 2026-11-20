"""
eval_multinli.py — Zero-shot MultiNLI evaluation for Galois v2 and Linear baseline
===================================================================================
Pure inference. No training. No gradients (except ThreeClassProbe re-fit on SNLI train).

Evaluates:
  1. Galois v2 (fine-tuned) on MultiNLI matched dev
  2. Linear baseline (fine-tuned) on MultiNLI matched dev
  3. Comparison with SNLI test results (loaded from saved results)
  4. Genre breakdown for Galois v2 on MultiNLI
"""

import sys, os, time, json
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)

from src.metrics import compute_nli_metrics, compute_3class_features, ThreeClassProbe

# ── Reproduce model classes (must match training code exactly) ────────────────

class BoxEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, box_dim):
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
        self.width_head  = nn.Linear(hidden_dim, box_dim)
        nn.init.xavier_uniform_(self.center_head.weight, gain=0.1)
        nn.init.xavier_uniform_(self.width_head.weight,  gain=0.1)
        nn.init.constant_(self.width_head.bias, -1.0)

    def forward(self, x):
        h  = self.net(x)
        c  = self.center_head(h)
        hw = F.softplus(self.width_head(h))
        return c - hw, c + hw


@dataclass
class GaloisConfig:
    model_name:    str = 'sentence-transformers/all-MiniLM-L6-v2'
    box_dim:       int = 128
    hidden_dim:    int = 512
    freeze_layers: int = 3
    scale:         float = 5.0


class GaloisSBERTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(cfg.model_name)
        self.box_enc = BoxEncoder(
            self.bert.config.hidden_size, cfg.hidden_dim, cfg.box_dim)
        self.scale = cfg.scale
        # No freezing needed for inference

    @staticmethod
    def _mean_pool(last_hidden, attn_mask):
        mask_exp = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (torch.sum(last_hidden * mask_exp, dim=1) /
                torch.clamp(mask_exp.sum(dim=1), min=1e-9))

    def encode_sentence(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        h   = self._mean_pool(out.last_hidden_state, mask)
        return self.box_enc(h)

    def forward(self, ids1, mask1, ids2, mask2):
        l1, u1 = self.encode_sentence(ids1, mask1)
        l2, u2 = self.encode_sentence(ids2, mask2)
        return l1, u1, l2, u2


@dataclass
class BaselineConfig:
    model_name:    str = 'sentence-transformers/all-MiniLM-L6-v2'
    freeze_layers: int = 3


class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model_name)
        hidden    = self.bert.config.hidden_size
        self.head = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _mean_pool(last_hidden, attn_mask):
        mask_exp = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (torch.sum(last_hidden * mask_exp, dim=1) /
                torch.clamp(mask_exp.sum(dim=1), min=1e-9))

    def encode(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self._mean_pool(out.last_hidden_state, mask)

    def forward(self, ids1, mask1, ids2, mask2):
        h1 = self.encode(ids1, mask1)
        h2 = self.encode(ids2, mask2)
        pair = h1 - h2
        return self.head(pair)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NLIPairDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_len=32):
        premises   = list(premises)
        hypotheses = list(hypotheses)
        labels     = list(labels)
        enc1 = tokenizer(premises,   padding='max_length', truncation=True,
                         max_length=max_len, return_tensors='pt')
        enc2 = tokenizer(hypotheses, padding='max_length', truncation=True,
                         max_length=max_len, return_tensors='pt')
        self.ids1   = enc1['input_ids']
        self.mask1  = enc1['attention_mask']
        self.ids2   = enc2['input_ids']
        self.mask2  = enc2['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        return (self.ids1[i], self.mask1[i],
                self.ids2[i], self.mask2[i], self.labels[i])


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def galois_encode_all(model, loader, device):
    L1, U1, L2, U2, LABS = [], [], [], [], []
    for ids1, mask1, ids2, mask2, labels in loader:
        l1, u1, l2, u2 = model(ids1.to(device), mask1.to(device),
                                ids2.to(device), mask2.to(device))
        L1.append(l1.cpu().numpy()); U1.append(u1.cpu().numpy())
        L2.append(l2.cpu().numpy()); U2.append(u2.cpu().numpy())
        LABS.append(labels.numpy())
    return (np.concatenate(L1), np.concatenate(U1),
            np.concatenate(L2), np.concatenate(U2),
            np.concatenate(LABS))


@torch.no_grad()
def baseline_predict_all(model, loader, device):
    PROBS, LABS = [], []
    for ids1, mask1, ids2, mask2, labels in loader:
        logits = model(ids1.to(device), mask1.to(device),
                       ids2.to(device), mask2.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        PROBS.append(probs)
        LABS.append(labels.numpy())
    return np.concatenate(PROBS), np.concatenate(LABS)


def eval_galois_metrics(l1, u1, l2, u2, labels, probe, scale=5.0):
    """Compute all metrics for Galois model given boxes and fitted probe."""
    # Binary metrics from containment violation
    viol  = (np.maximum(0, l2 - l1) + np.maximum(0, u1 - u2)).sum(axis=-1)
    p_ent = 1.0 / (1.0 + scale * viol)
    binary_true = (labels == 0).astype(int)
    bin_pred    = (p_ent >= 0.5).astype(int)
    bin_acc = accuracy_score(binary_true, bin_pred)
    bin_f1  = f1_score(binary_true, bin_pred, zero_division=0)
    try:
        bin_auc = roc_auc_score(binary_true, p_ent)
    except:
        bin_auc = float('nan')

    # 3-class via probe
    feats = compute_3class_features(l1, u1, l2, u2)
    preds_3 = probe.predict(feats)
    acc_3   = accuracy_score(labels, preds_3)
    cm      = confusion_matrix(labels, preds_3, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    recalls  = (cm / row_sums).diagonal()

    return {
        'bin_acc':    float(bin_acc),
        'bin_f1':     float(bin_f1),
        'bin_auc':    float(bin_auc),
        'acc_3class': float(acc_3),
        'per_class':  {'entail': float(recalls[0]),
                       'neutral': float(recalls[1]),
                       'contra': float(recalls[2])},
        'confusion':  cm.tolist(),
    }


def eval_baseline_metrics(probs, labels):
    """Compute all metrics for linear baseline given softmax probs."""
    preds = np.argmax(probs, axis=1)
    binary_true = (labels == 0).astype(int)
    bin_pred    = (preds == 0).astype(int)

    bin_acc = accuracy_score(binary_true, bin_pred)
    bin_f1  = f1_score(binary_true, bin_pred, zero_division=0)
    try:
        bin_auc = roc_auc_score(binary_true, probs[:, 0])
    except:
        bin_auc = float('nan')

    acc_3 = accuracy_score(labels, preds)
    cm    = confusion_matrix(labels, preds, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    recalls  = (cm / row_sums).diagonal()

    return {
        'bin_acc':    float(bin_acc),
        'bin_f1':     float(bin_f1),
        'bin_auc':    float(bin_auc),
        'acc_3class': float(acc_3),
        'per_class':  {'entail': float(recalls[0]),
                       'neutral': float(recalls[1]),
                       'contra': float(recalls[2])},
        'confusion':  cm.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    BASE = Path(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = BASE / 'results'
    GALOIS_CKPT = RESULTS_DIR / 'torch_models_v2' / 'best_model.pt'
    BASELINE_CKPT = RESULTS_DIR / 'torch_baseline' / 'best_model.pt'

    print(f"\n{'='*90}")
    print(f"  Zero-Shot MultiNLI Evaluation — Galois v2 & Linear Baseline")
    print(f"{'='*90}")
    print(f"  Device: {device}")
    print(f"  Galois checkpoint:   {GALOIS_CKPT}")
    print(f"  Baseline checkpoint: {BASELINE_CKPT}")

    # ── 1. Load tokenizer ────────────────────────────────────────────────────
    print(f"\n[1/8] Loading tokenizer ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    print(f"      Done in {time.time()-t0:.1f}s")

    # ── 2. Load MultiNLI ─────────────────────────────────────────────────────
    print(f"\n[2/8] Loading MultiNLI matched dev set ...")
    t0 = time.time()
    mnli_raw = load_dataset('multi_nli', split='validation_matched')
    # Filter out label=-1
    mnli_raw = mnli_raw.filter(lambda x: x['label'] != -1)
    n_mnli = len(mnli_raw)
    print(f"      Loaded {n_mnli} examples in {time.time()-t0:.1f}s")
    c = np.bincount(mnli_raw['label'], minlength=3)
    print(f"      Label distribution: ent={c[0]}  neu={c[1]}  con={c[2]}")

    # Extract genres
    genres = np.array(mnli_raw['genre'])
    unique_genres = sorted(set(genres))
    print(f"      Genres: {unique_genres}")
    for g in unique_genres:
        gc = np.bincount(np.array(mnli_raw['label'])[genres == g], minlength=3)
        print(f"        {g:>12}: n={gc.sum()}, ent={gc[0]}, neu={gc[1]}, con={gc[2]}")

    # Check if we need to sample
    MAX_N = 9900  # use full set if under ~10k
    if n_mnli > MAX_N:
        print(f"\n      Sampling {MAX_N} examples stratified by label ...")
        rng = np.random.default_rng(42)
        labels_arr = np.array(mnli_raw['label'])
        sample_idx = []
        per_class = MAX_N // 3
        for cls in range(3):
            cls_idx = np.where(labels_arr == cls)[0]
            chosen = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
            sample_idx.extend(chosen.tolist())
        rng.shuffle(sample_idx)
        mnli_raw = mnli_raw.select(sample_idx)
        genres = np.array(mnli_raw['genre'])
        n_mnli = len(mnli_raw)
        print(f"      Sampled {n_mnli} examples")

    # ── 3. Load SNLI train (for probe fitting) + test ────────────────────────
    print(f"\n[3/8] Loading SNLI (train=12000 for probe, test=2000) ...")
    t0 = time.time()
    snli_train = load_dataset('snli', split='train[:12000]').filter(
        lambda x: x['label'] != -1)
    snli_test = load_dataset('snli', split='test[:2000]').filter(
        lambda x: x['label'] != -1)
    print(f"      SNLI train={len(snli_train)}, test={len(snli_test)}  ({time.time()-t0:.1f}s)")

    # ── 4. Tokenize all datasets ─────────────────────────────────────────────
    print(f"\n[4/8] Tokenizing ...")
    t0 = time.time()
    mnli_ds = NLIPairDataset(mnli_raw['premise'], mnli_raw['hypothesis'],
                              mnli_raw['label'], tok, max_len=32)
    snli_train_ds = NLIPairDataset(snli_train['premise'], snli_train['hypothesis'],
                                    snli_train['label'], tok, max_len=32)
    snli_test_ds = NLIPairDataset(snli_test['premise'], snli_test['hypothesis'],
                                   snli_test['label'], tok, max_len=32)
    print(f"      Done in {time.time()-t0:.1f}s")

    eval_bs = 64
    mnli_loader = DataLoader(mnli_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
    snli_train_loader = DataLoader(snli_train_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
    snli_test_loader = DataLoader(snli_test_ds, batch_size=eval_bs, shuffle=False, num_workers=0)

    # ── 5. Load & run Galois v2 model ────────────────────────────────────────
    print(f"\n[5/8] Loading Galois v2 model ...")
    t0 = time.time()
    galois_cfg = GaloisConfig()
    galois_model = GaloisSBERTModel(galois_cfg).to(device)
    ckpt = torch.load(GALOIS_CKPT, map_location=device, weights_only=False)
    galois_model.load_state_dict(ckpt['state_dict'])
    galois_model.eval()
    print(f"      Loaded (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # Encode SNLI train for probe fitting
    print(f"      Encoding SNLI train for probe fitting ...")
    l1_tr, u1_tr, l2_tr, u2_tr, labs_tr = galois_encode_all(
        galois_model, snli_train_loader, device)
    feats_tr = compute_3class_features(l1_tr, u1_tr, l2_tr, u2_tr)

    # Fit 3-class probe on SNLI train
    print(f"      Fitting ThreeClassProbe on SNLI train ({len(labs_tr)} examples) ...")
    probe = ThreeClassProbe(n_epochs=400, lr=0.05)
    probe.fit(feats_tr, labs_tr)

    # Encode SNLI test
    print(f"      Encoding SNLI test ...")
    l1_te, u1_te, l2_te, u2_te, labs_te = galois_encode_all(
        galois_model, snli_test_loader, device)
    galois_snli = eval_galois_metrics(l1_te, u1_te, l2_te, u2_te, labs_te, probe)
    print(f"      SNLI test: bin_acc={galois_snli['bin_acc']:.4f}, "
          f"3-acc={galois_snli['acc_3class']:.4f}")

    # Encode MultiNLI
    print(f"      Encoding MultiNLI ({n_mnli} examples) ...")
    t1 = time.time()
    l1_mn, u1_mn, l2_mn, u2_mn, labs_mn = galois_encode_all(
        galois_model, mnli_loader, device)
    galois_mnli = eval_galois_metrics(l1_mn, u1_mn, l2_mn, u2_mn, labs_mn, probe)
    print(f"      MultiNLI: bin_acc={galois_mnli['bin_acc']:.4f}, "
          f"3-acc={galois_mnli['acc_3class']:.4f}  ({time.time()-t1:.1f}s)")

    # Genre breakdown for Galois v2
    print(f"\n      Genre breakdown (Galois v2, 3-class) ...")
    feats_mn = compute_3class_features(l1_mn, u1_mn, l2_mn, u2_mn)
    preds_mn = probe.predict(feats_mn)
    genre_results = {}
    for g in unique_genres:
        g_mask = genres == g
        g_labs = labs_mn[g_mask]
        g_preds = preds_mn[g_mask]
        g_acc = accuracy_score(g_labs, g_preds)
        g_cm = confusion_matrix(g_labs, g_preds, labels=[0, 1, 2])
        row_sums = g_cm.sum(axis=1, keepdims=True).clip(min=1)
        g_recalls = (g_cm / row_sums).diagonal()
        genre_results[g] = {
            'n': int(g_mask.sum()),
            'acc_3class': float(g_acc),
            'per_class': {'entail': float(g_recalls[0]),
                          'neutral': float(g_recalls[1]),
                          'contra': float(g_recalls[2])},
        }
        print(f"        {g:>12}: n={genre_results[g]['n']:>5}, "
              f"3-acc={g_acc:.4f}, "
              f"ent={g_recalls[0]:.3f}, neu={g_recalls[1]:.3f}, con={g_recalls[2]:.3f}")

    # Free Galois model memory
    del galois_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 6. Load & run Linear baseline ────────────────────────────────────────
    print(f"\n[6/8] Loading Linear baseline model ...")
    t0 = time.time()
    baseline_cfg = BaselineConfig()
    baseline_model = LinearClassifier(baseline_cfg).to(device)
    ckpt_bl = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_bl['state_dict'])
    baseline_model.eval()
    print(f"      Loaded (epoch={ckpt_bl['epoch']}, val_acc={ckpt_bl['val_acc']:.4f})")

    # SNLI test
    print(f"      Running on SNLI test ...")
    bl_probs_te, bl_labs_te = baseline_predict_all(baseline_model, snli_test_loader, device)
    baseline_snli = eval_baseline_metrics(bl_probs_te, bl_labs_te)
    print(f"      SNLI test: bin_acc={baseline_snli['bin_acc']:.4f}, "
          f"3-acc={baseline_snli['acc_3class']:.4f}")

    # MultiNLI
    print(f"      Running on MultiNLI ({n_mnli} examples) ...")
    t1 = time.time()
    bl_probs_mn, bl_labs_mn = baseline_predict_all(baseline_model, mnli_loader, device)
    baseline_mnli = eval_baseline_metrics(bl_probs_mn, bl_labs_mn)
    print(f"      MultiNLI: bin_acc={baseline_mnli['bin_acc']:.4f}, "
          f"3-acc={baseline_mnli['acc_3class']:.4f}  ({time.time()-t1:.1f}s)")

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 7. Print all tables ──────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  TABLE 1: Cross-Dataset Comparison (SNLI → MultiNLI)")
    print(f"{'='*100}")
    print(f"\n  {'Model':<28} {'Dataset':<14} {'Bin-Acc':>8} {'Bin-F1':>8} "
          f"{'Bin-AUC':>8} {'3-Acc':>8} {'Ent':>8} {'Neu':>8} {'Con':>8}")
    print(f"  {'-'*94}")

    rows = [
        ("Galois v2 (fine-tuned)", "SNLI test", galois_snli),
        ("Galois v2 (fine-tuned)", "MultiNLI", galois_mnli),
        ("Linear baseline", "SNLI test", baseline_snli),
        ("Linear baseline", "MultiNLI", baseline_mnli),
    ]
    for name, dataset, m in rows:
        pc = m['per_class']
        print(f"  {name:<28} {dataset:<14} {m['bin_acc']:>8.4f} {m['bin_f1']:>8.4f} "
              f"{m['bin_auc']:>8.4f} {m['acc_3class']:>8.4f} "
              f"{pc['entail']:>8.3f} {pc['neutral']:>8.3f} {pc['contra']:>8.3f}")

    # ── Deltas ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  TABLE 2: Generalization Gap (SNLI → MultiNLI)")
    print(f"{'='*100}")
    print(f"\n  {'Model':<28} {'Δ Bin-Acc':>10} {'Δ Bin-F1':>10} {'Δ Bin-AUC':>10} "
          f"{'Δ 3-Acc':>10} {'Δ Ent':>8} {'Δ Neu':>8} {'Δ Con':>8}")
    print(f"  {'-'*86}")

    for name, snli_m, mnli_m in [
        ("Galois v2 (fine-tuned)", galois_snli, galois_mnli),
        ("Linear baseline", baseline_snli, baseline_mnli),
    ]:
        d = {}
        for k in ['bin_acc', 'bin_f1', 'bin_auc', 'acc_3class']:
            d[k] = mnli_m[k] - snli_m[k]
        d_pc = {}
        for k in ['entail', 'neutral', 'contra']:
            d_pc[k] = mnli_m['per_class'][k] - snli_m['per_class'][k]
        print(f"  {name:<28} {d['bin_acc']:>+10.4f} {d['bin_f1']:>+10.4f} "
              f"{d['bin_auc']:>+10.4f} {d['acc_3class']:>+10.4f} "
              f"{d_pc['entail']:>+8.3f} {d_pc['neutral']:>+8.3f} {d_pc['contra']:>+8.3f}")

    # ── Confusion matrices ───────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  TABLE 3: Confusion Matrices (rows=true, cols=pred)")
    print(f"{'='*100}")

    for name, dataset, m in rows:
        cm = np.array(m['confusion'])
        print(f"\n  {name} — {dataset}")
        print(f"  {'':>12}  pred_ent  pred_neu  pred_con")
        for i, nm in enumerate(['true_ent', 'true_neu', 'true_con']):
            print(f"  {nm:>12}  {cm[i,0]:>8d}  {cm[i,1]:>8d}  {cm[i,2]:>8d}")

    # ── Genre breakdown ──────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  TABLE 4: Genre Breakdown — Galois v2 on MultiNLI")
    print(f"{'='*100}")
    print(f"\n  {'Genre':<14} {'N':>6} {'3-Acc':>8} {'Ent':>8} {'Neu':>8} {'Con':>8}")
    print(f"  {'-'*54}")
    for g in unique_genres:
        gr = genre_results[g]
        pc = gr['per_class']
        print(f"  {g:<14} {gr['n']:>6} {gr['acc_3class']:>8.4f} "
              f"{pc['entail']:>8.3f} {pc['neutral']:>8.3f} {pc['contra']:>8.3f}")
    print(f"  {'-'*54}")
    print(f"  {'ALL':<14} {n_mnli:>6} {galois_mnli['acc_3class']:>8.4f} "
          f"{galois_mnli['per_class']['entail']:>8.3f} "
          f"{galois_mnli['per_class']['neutral']:>8.3f} "
          f"{galois_mnli['per_class']['contra']:>8.3f}")

    # ── 8. Save results ──────────────────────────────────────────────────────
    results = {
        'n_multinli': n_mnli,
        'n_snli_test': len(snli_test),
        'galois_v2': {
            'snli_test': galois_snli,
            'multinli':  galois_mnli,
            'genre_breakdown': genre_results,
        },
        'linear_baseline': {
            'snli_test': baseline_snli,
            'multinli':  baseline_mnli,
        },
        'deltas': {
            'galois_v2': {
                k: galois_mnli[k] - galois_snli[k]
                for k in ['bin_acc', 'bin_f1', 'bin_auc', 'acc_3class']
            },
            'linear_baseline': {
                k: baseline_mnli[k] - baseline_snli[k]
                for k in ['bin_acc', 'bin_f1', 'bin_auc', 'acc_3class']
            },
        },
    }

    out_path = RESULTS_DIR / 'multinli_evaluation.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\n  Results saved to {out_path}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
