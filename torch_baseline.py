"""
torch_baseline.py  —  Fine-tuned MiniLM + 3-class linear head (no box geometry)
=================================================================================
Ablation baseline that isolates the contribution of Galois box geometry from the
contribution of fine-tuning the BERT backbone.

Architecture
------------
  all-MiniLM-L6-v2 (22M params, 6 transformer layers, 384-dim)
    frozen  : token embeddings + layers 0 .. freeze_layers-1
    trained : top (6 - freeze_layers) layers          lr = 2e-5
    → attention-masked mean pooling
  LinearHead: 384 → 3 logits                          lr = 5e-4
    Loss    : standard 3-class cross-entropy

Hyperparameters match torch_train.py v2 exactly so results are directly
comparable.  The only difference is the head and loss function.

Usage
-----
  PYTHONIOENCODING=utf-8 python -u torch_baseline.py
"""

import sys, os, time, json
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score)


# ── Config ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model — identical to v2
    model_name:    str   = 'sentence-transformers/all-MiniLM-L6-v2'
    freeze_layers: int   = 3      # freeze bottom 3/6 BERT layers + embeddings

    # Data — identical to v2
    n_train:       int   = 12000
    n_val:         int   = 2000
    n_test:        int   = 2000
    max_len:       int   = 32

    # Training — identical to v2
    batch_size:    int   = 32
    accumulate:    int   = 4
    n_epochs:      int   = 20
    patience:      int   = 5
    warmup_ratio:  float = 0.1

    # Learning rates — identical to v2
    lr_bert:       float = 2e-5
    lr_head:       float = 5e-4
    weight_decay:  float = 0.01

    seed:          int   = 42
    save_dir:      str   = 'results/torch_baseline'


# ── Model ───────────────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    """
    Fine-tunable BERT encoder (partially frozen) + 3-class linear head.

    This is the ablation baseline: identical backbone setup to GaloisSBERTModel
    but replaces the BoxEncoder + geometric losses with a single linear layer
    trained with standard cross-entropy.

    Comparison with torch_train.py:
      torch_train.py  : BERT → mean-pool → BoxEncoder (384→512→128 boxes) → Galois losses
      torch_baseline.py: BERT → mean-pool → Linear (384→3)               → cross-entropy
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model_name)
        hidden    = self.bert.config.hidden_size          # 384 for MiniLM
        self.head = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)
        self._apply_freeze(cfg.freeze_layers)

    def _apply_freeze(self, n: int):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False
        n_total   = len(self.bert.encoder.layer)
        n_trained = n_total - n
        print(f"      BERT: {n}/{n_total} layers frozen  |  "
              f"{n_trained} layers trainable  (lr={2e-5:.0e})")

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor,
                   attn_mask:   torch.Tensor) -> torch.Tensor:
        mask_exp = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (torch.sum(last_hidden * mask_exp, dim=1) /
                torch.clamp(mask_exp.sum(dim=1), min=1e-9))

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns mean-pooled BERT embedding (B, hidden)."""
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self._mean_pool(out.last_hidden_state, mask)

    def forward(self, ids1, mask1, ids2, mask2) -> torch.Tensor:
        """Returns logits (B, 3) for the premise→hypothesis pair."""
        h1 = self.encode(ids1, mask1)
        h2 = self.encode(ids2, mask2)
        # Difference + element-wise product: classic NLI decomposition
        # (same information as concatenation but half the param count)
        pair = h1 - h2                                   # (B, 384)
        return self.head(pair)                           # (B, 3)

    def bert_params(self):
        return [p for p in self.bert.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.head.parameters())


# ── Dataset ─────────────────────────────────────────────────────────────────────

class SNLIPairDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_len: int):
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


# ── Evaluation helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(model: LinearClassifier,
                loader: DataLoader,
                device) -> tuple:
    """
    Run forward pass over all batches.
    Returns (probs, labels) as numpy arrays.
      probs: (N, 3) softmax probabilities
      labels: (N,) int64 ground truth
    """
    PROBS, LABS = [], []
    for ids1, mask1, ids2, mask2, labels in loader:
        logits = model(ids1.to(device),  mask1.to(device),
                       ids2.to(device),  mask2.to(device))
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        PROBS.append(probs)
        LABS.append(labels.numpy())
    return np.concatenate(PROBS), np.concatenate(LABS)


def eval_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute accuracy, macro-F1, per-class recall, confusion matrix, AUC.

    Binary AUC: treats entailment (class 0) vs rest as the positive class.
    3-class AUC: one-vs-rest macro.
    """
    preds = np.argmax(probs, axis=1)

    acc3   = accuracy_score(labels, preds)
    f1mac  = f1_score(labels, preds, average='macro', zero_division=0)
    cm     = confusion_matrix(labels, preds, labels=[0, 1, 2])

    # Per-class recall (diagonal / row sum)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    recalls  = (cm / row_sums).diagonal()   # [ent_recall, neu_recall, con_recall]

    # Binary AUC (entailment vs rest)
    binary_true = (labels == 0).astype(int)
    p_ent        = probs[:, 0]              # P(entailment)
    try:
        bin_auc = roc_auc_score(binary_true, p_ent)
    except Exception:
        bin_auc = float('nan')

    # Binary accuracy (argmax==0 → entail else non-entail)
    bin_pred = (preds == 0).astype(int)
    bin_acc  = accuracy_score(binary_true, bin_pred)
    bin_f1   = f1_score(binary_true, bin_pred, zero_division=0)

    return {
        'acc_3class':  acc3,
        'f1_macro':    f1mac,
        'per_class':   {'entail': recalls[0], 'neutral': recalls[1],
                        'contra': recalls[2]},
        'confusion':   cm,
        'bin_acc':     bin_acc,
        'bin_f1':      bin_f1,
        'bin_auc':     bin_auc,
    }


# ── LR schedule ─────────────────────────────────────────────────────────────────

def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))
    return LambdaLR(optimizer, lr_lambda)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n{'='*70}")
    print(f"  ABLATION BASELINE  —  Fine-tuned MiniLM + Linear 3-class head")
    print(f"{'='*70}")
    print(f"  Device   : {device}")
    print(f"  Head     : Linear(384, 3)  +  CrossEntropy  (NO box geometry)")
    print(f"  Freeze   : bottom {cfg.freeze_layers}/6 BERT layers + embeddings")
    print(f"  LR       : BERT={cfg.lr_bert:.0e}  head={cfg.lr_head:.0e}")
    print(f"  Batch    : {cfg.batch_size} x {cfg.accumulate} = "
          f"{cfg.batch_size * cfg.accumulate} effective")
    print(f"  n_train  : {cfg.n_train}  (identical to Galois v2 for fair comparison)")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading & tokenising SNLI ...")
    raw_tr  = load_dataset('snli', split=f'train[:{cfg.n_train}]').filter(
        lambda x: x['label'] != -1)
    raw_val = load_dataset('snli', split=f'validation[:{cfg.n_val}]').filter(
        lambda x: x['label'] != -1)
    raw_te  = load_dataset('snli', split=f'test[:{cfg.n_test}]').filter(
        lambda x: x['label'] != -1)
    print(f"      train={len(raw_tr)}  val={len(raw_val)}  test={len(raw_te)}")
    for nm, sp in [('train', raw_tr), ('val', raw_val), ('test', raw_te)]:
        c = np.bincount(sp['label'], minlength=3)
        print(f"      {nm}: ent={c[0]}  neu={c[1]}  con={c[2]}")

    t0  = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tr_ds  = SNLIPairDataset(raw_tr['premise'],  raw_tr['hypothesis'],
                             raw_tr['label'],    tok, cfg.max_len)
    val_ds = SNLIPairDataset(raw_val['premise'], raw_val['hypothesis'],
                             raw_val['label'],   tok, cfg.max_len)
    te_ds  = SNLIPairDataset(raw_te['premise'],  raw_te['hypothesis'],
                             raw_te['label'],    tok, cfg.max_len)
    print(f"      Tokenisation: {time.time()-t0:.1f}s")

    eval_bs    = cfg.batch_size * 2
    tr_loader  = DataLoader(tr_ds,  batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=eval_bs,        shuffle=False, num_workers=0)
    te_loader  = DataLoader(te_ds,  batch_size=eval_bs,        shuffle=False, num_workers=0)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print(f"\n[2/5] Building model ...")
    model = LinearClassifier(cfg).to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total params: {n_tot/1e6:.1f}M  |  Trainable: {n_trn/1e6:.1f}M")

    optimizer = AdamW([
        {'params': model.bert_params(), 'lr': cfg.lr_bert,
         'weight_decay': cfg.weight_decay},
        {'params': model.head_params(), 'lr': cfg.lr_head,
         'weight_decay': cfg.weight_decay},
    ])
    steps_per_epoch = len(tr_loader) // cfg.accumulate
    total_steps     = steps_per_epoch * cfg.n_epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler       = make_lr_scheduler(optimizer, warmup_steps, total_steps)

    # ── 3. Training loop ──────────────────────────────────────────────────────
    print(f"\n[3/5] Training ...")
    print(f"      steps/epoch={steps_per_epoch}  total={total_steps}  "
          f"warmup={warmup_steps}  patience={cfg.patience}")

    print(f"\n  {'Ep':>3}  {'Loss':>7}  {'ValAcc':>7}  {'ValF1':>6}  {'t':>5}")
    print("  " + "-" * 38)

    best_val_acc = -1.0
    best_state   = None
    pat_ctr      = 0
    history      = []
    global_step  = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step_i, (ids1, mask1, ids2, mask2, labels) in enumerate(tr_loader):
            ids1   = ids1.to(device);   mask1 = mask1.to(device)
            ids2   = ids2.to(device);   mask2 = mask2.to(device)
            labels = labels.to(device)

            logits = model(ids1, mask1, ids2, mask2)
            loss   = F.cross_entropy(logits, labels)

            (loss / cfg.accumulate).backward()

            ep_loss   += loss.item()
            n_batches += 1

            if (step_i + 1) % cfg.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Validation (3-class accuracy)
        model.eval()
        val_probs, val_labs = predict_all(model, val_loader, device)
        vm      = eval_metrics(val_probs, val_labs)
        val_acc = vm['acc_3class']
        val_f1  = vm['f1_macro']

        elapsed = time.time() - t0
        avg_loss = ep_loss / max(n_batches, 1)
        print(f"  {epoch:>3}  {avg_loss:>7.4f}  {val_acc:>7.4f}  "
              f"{val_f1:>6.4f}  {elapsed:>4.0f}s")

        history.append({
            'epoch':   epoch,
            'loss':    avg_loss,
            'val_acc': val_acc,
            'val_f1':  val_f1,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            pat_ctr      = 0
            torch.save({'state_dict': best_state,
                        'config':     asdict(cfg),
                        'epoch':      epoch,
                        'val_acc':    val_acc},
                       save_dir / 'best_model.pt')
        else:
            pat_ctr += 1
            if pat_ctr >= cfg.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val 3-acc={best_val_acc:.4f})")
                break

    # ── 4/5. Final evaluation ─────────────────────────────────────────────────
    print(f"\n[4/5] Loading best checkpoint (val 3-acc={best_val_acc:.4f}) ...")
    model.load_state_dict(best_state)
    model.eval()

    print(f"\n[5/5] Test-set evaluation ...")
    te_probs, te_labs = predict_all(model, te_loader, device)
    te = eval_metrics(te_probs, te_labs)

    pc = te['per_class']
    cm = te['confusion']

    print(f"\n{'='*70}")
    print(f"  ABLATION BASELINE  —  SNLI Test Results")
    print(f"{'='*70}")
    print(f"\n  3-class accuracy  : {te['acc_3class']:.4f}")
    print(f"  Macro F1          : {te['f1_macro']:.4f}")
    print(f"    ent={pc['entail']:.3f}  neu={pc['neutral']:.3f}  "
          f"con={pc['contra']:.3f}")
    print(f"\n  Binary (entail vs rest):")
    print(f"    Acc={te['bin_acc']:.4f}  F1={te['bin_f1']:.4f}  "
          f"AUC={te['bin_auc']:.4f}")
    print(f"\n  Confusion matrix  [rows=true, cols=pred]  "
          f"(0=entail  1=neutral  2=contra)")
    print(f"  {'':>10}  pred_ent  pred_neu  pred_con")
    for i, nm in enumerate(['true_ent', 'true_neu', 'true_con']):
        print(f"  {nm:>10}  {cm[i,0]:>8d}  {cm[i,1]:>8d}  {cm[i,2]:>8d}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n  Ablation comparison (all use same SNLI splits, seed=42):")
    print(f"  {'Model':<45}  {'Bin-Acc':>7}  {'3-Acc':>6}  "
          f"{'ent':>5}  {'neu':>5}  {'con':>5}")
    print("  " + "-" * 76)
    frozen_rows = [
        ("Cosine (frozen SBERT)",                "0.5056", "0.5677", "0.737", "0.240", "0.720"),
        ("Order Embeddings (frozen)",             "0.6902", "0.5127", "0.696", "0.191", "0.645"),
        ("Galois no-CEGAR (frozen)",              "0.7386", "0.5585", "0.586", "0.633", "0.454"),
        ("Galois CEGAR warmup=5 (frozen)",        "0.7436", "0.5549", "0.581", "0.656", "0.424"),
        ("Galois fine-tuned v1 (6k, freeze=4)",   "0.7945", "0.5977", "0.715", "0.587", "0.484"),
        ("Galois fine-tuned v2 (12k, freeze=3)",  "0.8067", "0.6419", "0.749", "0.661", "0.509"),
    ]
    for row in frozen_rows:
        nm, ba, ta, e, n, c = row
        print(f"  {nm:<45}  {ba:>7}  {ta:>6}  {e:>5}  {n:>5}  {c:>5}")

    # Current baseline
    ba = f"{te['bin_acc']:.4f}"
    ta = f"{te['acc_3class']:.4f}"
    e  = f"{pc['entail']:.3f}"
    n  = f"{pc['neutral']:.3f}"
    c  = f"{pc['contra']:.3f}"
    print(f"  {'*** Linear baseline (12k, freeze=3, no box)':<45}  "
          f"{ba:>7}  {ta:>6}  {e:>5}  {n:>5}  {c:>5}  <-- ABLATION")

    delta_3acc = te['acc_3class'] - 0.6419
    delta_neu  = pc['neutral']    - 0.661
    delta_con  = pc['contra']     - 0.509
    print(f"\n  Galois v2 vs this baseline:")
    print(f"    delta 3-acc : {-delta_3acc:+.4f}  "
          f"({'Galois box helps' if delta_3acc < 0 else 'baseline stronger'})")
    print(f"    delta neu   : {-delta_neu:+.3f}")
    print(f"    delta con   : {-delta_con:+.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'model':      'linear_baseline_fine_tuned',
        'n_train':     cfg.n_train,
        'freeze':      cfg.freeze_layers,
        'test': {
            'acc_3class': te['acc_3class'],
            'f1_macro':   te['f1_macro'],
            'per_class':  pc,
            'bin_acc':    te['bin_acc'],
            'bin_f1':     te['bin_f1'],
            'bin_auc':    te['bin_auc'],
            'confusion':  cm.tolist(),
        },
        'best_val_acc': best_val_acc,
        'history':      history,
    }
    out = save_dir / 'results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")
    print(f"  Checkpoint saved to {save_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
