"""
torch_train.py  —  End-to-end fine-tuning: SBERT backbone + Galois box head
=============================================================================
Fine-tunes all-MiniLM-L6-v2 jointly with a Galois box encoder under the
3-class directional hinge loss.

Why fine-tune?
  Frozen SBERT is capped at 55.5% 3-class accuracy because contradiction and
  neutral share nearly identical cosine-similarity geometry (asymmetry ≈ −1.5
  for both).  Training the top 2/6 BERT layers under the directional loss
  reshapes the embedding space so contradiction pairs become geometrically
  disjoint from neutral (large viol_sym, near-zero overlap).

Architecture
------------
  all-MiniLM-L6-v2 (22M params, 6 transformer layers, 384-dim)
    frozen  : token embeddings + layers 0-3      (general language rep)
    trained : layers 4-5                         lr = 2e-5
    → attention-masked mean pooling (SentenceTransformer convention)
  BoxEncoder: 384 → LayerNorm+ReLU(512) → (center, half-width) → (l, u) ∈ R^128
    trained : lr = 5e-4

Loss
----
  λ2 * galois_hinge   binary ent/non-ent margin hinge (correct direction)
  λ3 * task_bce       binary cross-entropy on p_entail = σ(−viol · scale)
  λ5 * galois_3class  3-class directional hinge (margins from violation analysis)

Gradient accumulation
---------------------
  accumulate=4, batch_size=32  →  effective batch = 128
  Linear warmup (10% of steps) → cosine decay

CPU runtime on this machine
---------------------------
  ~4-5 min/epoch × 20 epochs ≈ 80-100 min total
  (all-MiniLM is 6 layers / 22M params — specifically designed for CPU)

Usage
-----
  python torch_train.py               # run with defaults
  PYTHONIOENCODING=utf-8 python torch_train.py   # Windows (unicode output)
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

from src.metrics import compute_nli_metrics, compute_3class_features, ThreeClassProbe


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    model_name:      str   = 'sentence-transformers/all-MiniLM-L6-v2'
    box_dim:         int   = 128
    hidden_dim:      int   = 512
    freeze_layers:   int   = 4      # freeze bottom N of 6 BERT layers + embeddings

    # Data
    n_train:         int   = 6000
    n_val:           int   = 2000
    n_test:          int   = 2000
    max_len:         int   = 32     # token max-length per sentence
    # SNLI premises avg ~14 words, hypotheses ~8 words → ~20-22 BPE tokens.
    # max_len=32 covers >90% of examples.  BERT attention is O(L²) so
    # 32 vs 64 tokens is 4× faster per batch on CPU.

    # Training
    batch_size:      int   = 32
    accumulate:      int   = 4      # effective_batch = batch_size × accumulate = 128
    n_epochs:        int   = 20
    patience:        int   = 5
    warmup_ratio:    float = 0.1    # fraction of total steps for linear LR warmup

    # Layer-wise learning rates
    # BERT pretrained weights need a much smaller LR to avoid catastrophic forgetting.
    lr_bert:         float = 2e-5
    lr_head:         float = 5e-4   # BoxEncoder
    weight_decay:    float = 0.01

    # Loss weights
    lambda2:         float = 2.0    # binary galois hinge
    lambda3:         float = 1.0    # binary BCE
    lambda5:         float = 0.5    # 3-class directional hinge

    # Binary loss params
    margin:          float = 1.0    # non-entailment viol must exceed this
    scale:           float = 5.0    # p_ent = σ(−viol × scale)

    # 3-class margins — calibrated from frozen-SBERT violation magnitude analysis:
    #   viol_fwd medians: ent=0.65  neu=5.65  con=7.31
    #   viol_bwd medians: ent=12.7  neu=4.09  con=5.86
    #   asymmetry (bwd-fwd) medians: ent=+12.0  neu=−1.6  con=−1.5
    ent_bwd_target:  float = 10.0   # push ent viol_bwd above this (box asymmetry)
    neu_fwd_margin:  float = 3.0    # push neu viol_fwd above this
    neu_bwd_ceiling: float = 6.0    # push neu viol_bwd below this
    con_fwd_margin:  float = 5.0    # push con viol_fwd above this
    overlap_weight:  float = 0.1    # push con box overlap toward 0

    seed:            int   = 42
    save_dir:        str   = 'results/torch_models'


# ── Model ──────────────────────────────────────────────────────────────────────

class BoxEncoder(nn.Module):
    """
    Maps a BERT mean-pool embedding → axis-aligned box (l, u) ∈ R^box_dim.

    Parameterisation:
        h   = LayerNorm+ReLU MLP(x)           (two hidden layers)
        c   = center_head(h)                   (unconstrained center)
        hw  = softplus(width_head(h))          (half-width > 0)
        l, u = c − hw,  c + hw                (valid box: l ≤ u always)

    Width bias initialised to −1 so softplus starts at ≈ ln(2) ≈ 0.69,
    giving compact initial boxes that expand as training proceeds.
    """

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
        self.width_head  = nn.Linear(hidden_dim, box_dim)
        nn.init.xavier_uniform_(self.center_head.weight, gain=0.1)
        nn.init.xavier_uniform_(self.width_head.weight,  gain=0.1)
        nn.init.constant_(self.width_head.bias, -1.0)   # compact initial boxes

    def forward(self, x: torch.Tensor):
        h  = self.net(x)
        c  = self.center_head(h)
        hw = F.softplus(self.width_head(h))   # half-width > 0
        return c - hw, c + hw                 # (l, u)


class GaloisSBERTModel(nn.Module):
    """
    Fine-tunable BERT encoder (partially frozen) + Galois box head.

    Freezing strategy:
      - Token/position embeddings:   frozen  (task-agnostic)
      - BERT layers 0 .. freeze_layers−1:  frozen  (general syntax/semantics)
      - BERT layers freeze_layers .. 5:    trained  (task-specific geometry)
      - BoxEncoder:                  trained  (box parameterisation)

    The frozen layers preserve pretrained language knowledge (SentenceTransformer
    training on 1B+ sentence pairs).  The top layers and box head learn to
    separate entailment / neutral / contradiction in box space.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(cfg.model_name)
        self.box_enc = BoxEncoder(
            self.bert.config.hidden_size, cfg.hidden_dim, cfg.box_dim)
        self.scale = cfg.scale
        self._apply_freeze(cfg.freeze_layers)

    def _apply_freeze(self, n: int):
        """Freeze token embeddings and bottom n transformer layers."""
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
                   attn_mask: torch.Tensor) -> torch.Tensor:
        """Attention-masked mean pooling — SentenceTransformer convention."""
        mask_exp = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (torch.sum(last_hidden * mask_exp, dim=1) /
                torch.clamp(mask_exp.sum(dim=1), min=1e-9))

    def encode_sentence(self, ids: torch.Tensor,
                        mask: torch.Tensor) -> tuple:
        """Returns (l, u) for a batch of single sentences."""
        out = self.bert(input_ids=ids, attention_mask=mask)
        h   = self._mean_pool(out.last_hidden_state, mask)
        return self.box_enc(h)

    def forward(self, ids1, mask1, ids2, mask2):
        l1, u1 = self.encode_sentence(ids1, mask1)
        l2, u2 = self.encode_sentence(ids2, mask2)
        return l1, u1, l2, u2

    def bert_params(self):
        """Trainable BERT parameters (top layers only)."""
        return [p for p in self.bert.parameters() if p.requires_grad]

    def head_params(self):
        """BoxEncoder parameters."""
        return list(self.box_enc.parameters())


# ── Loss functions (PyTorch, fully differentiable) ─────────────────────────────

def box_viol(l1: torch.Tensor, u1: torch.Tensor,
             l2: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """
    Axis-aligned box containment violation:
        I([l1,u1] ⊆ [l2,u2]) = Σ_k relu(l2_k−l1_k) + relu(u1_k−u2_k)
    Returns (N,).  Zero iff [l1,u1] ⊆ [l2,u2].
    """
    return (F.relu(l2 - l1) + F.relu(u1 - u2)).sum(dim=-1)


def loss_galois_hinge(l1, u1, l2, u2, labels, margin: float = 1.0):
    """
    Binary directional hinge loss.
    Entailment (label=0):     push viol_fwd → 0  (penalty = relu(viol))
    Non-entailment (label≠0): push viol_fwd ≥ margin  (penalty = relu(margin−viol))
    """
    viol   = box_viol(l1, u1, l2, u2)
    binary = (labels == 0).float()
    return (binary * F.relu(viol) +
            (1 - binary) * F.relu(margin - viol)).mean()


def loss_task_bce(l1, u1, l2, u2, labels, scale: float = 5.0):
    """
    Binary cross-entropy on soft entailment probability.
    p_entail = sigmoid(−viol × scale) ∈ (0, 1)
    Target = 1 for entailment, 0 otherwise.
    """
    viol   = box_viol(l1, u1, l2, u2)
    p_ent  = torch.sigmoid(-viol * scale)
    binary = (labels == 0).float()
    return F.binary_cross_entropy(p_ent.clamp(1e-7, 1 - 1e-7), binary)


def loss_galois_3class(l1, u1, l2, u2, labels,
                       ent_bwd_target:  float = 10.0,
                       neu_fwd_margin:  float = 3.0,
                       neu_bwd_ceiling: float = 6.0,
                       con_fwd_margin:  float = 5.0,
                       overlap_weight:  float = 0.1) -> torch.Tensor:
    """
    3-class directional hinge loss.

    Margins calibrated from violation distributions on frozen SBERT + SNLI:
      entailment   viol_fwd p50 = 0.65,  viol_bwd p50 = 12.7
      neutral      viol_fwd p50 = 5.65,  viol_bwd p50 =  4.1
      contradiction viol_fwd p50 = 7.31,  viol_bwd p50 =  5.9

    Loss terms:
      Entailment   : relu(viol_fwd) + 0.5·relu(ent_bwd_target − viol_bwd)
                     push viol_fwd→0  AND  viol_bwd→high (directional asymmetry)
      Neutral      : relu(neu_fwd_margin − viol_fwd) + 0.5·relu(viol_bwd − neu_bwd_ceiling)
                     push viol_fwd≥3  AND  viol_bwd≤6  (symmetric partial overlap)
      Contradiction: relu(con_fwd_margin − viol_fwd) + overlap·overlap_weight
                     push viol_fwd≥5  AND  boxes disjoint  (no intersection)
    """
    viol_fwd = box_viol(l1, u1, l2, u2)
    viol_bwd = box_viol(l2, u2, l1, u1)
    overlap  = F.relu(torch.minimum(u1, u2) -
                      torch.maximum(l1, l2)).sum(dim=-1)   # (N,)

    ent    = (labels == 0).float()
    neu    = (labels == 1).float()
    contra = (labels == 2).float()

    l_ent = ent * (
        F.relu(viol_fwd) +
        0.5 * F.relu(ent_bwd_target - viol_bwd)
    )
    l_neu = neu * (
        F.relu(neu_fwd_margin - viol_fwd) +
        0.5 * F.relu(viol_bwd - neu_bwd_ceiling)
    )
    l_con = contra * (
        F.relu(con_fwd_margin - viol_fwd) +
        overlap * overlap_weight
    )
    return (l_ent + l_neu + l_con).mean()


# ── Dataset ────────────────────────────────────────────────────────────────────

class SNLIPairDataset(Dataset):
    """
    Pre-tokenises all sentence pairs at construction time and stores
    as tensors in memory.  With max_len=64 and 12k pairs, memory footprint
    is ~12 MB (negligible).

    Each __getitem__ returns (ids1, mask1, ids2, mask2, label).
    """

    def __init__(self, premises,
                       hypotheses,
                       labels,
                       tokenizer,
                       max_len:     int):
        # HuggingFace Dataset columns are not plain Python lists — the tokenizer
        # requires list[str], so coerce here rather than at every call site.
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.ids1[i], self.mask1[i],
                self.ids2[i], self.mask2[i], self.labels[i])


# ── Evaluation helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(model: GaloisSBERTModel,
               loader: DataLoader,
               device) -> tuple:
    """
    Run model.forward() over all batches in loader with no_grad.
    Returns numpy arrays (l1, u1, l2, u2, labels).
    """
    L1, U1, L2, U2, LABS = [], [], [], [], []
    for ids1, mask1, ids2, mask2, labels in loader:
        l1, u1, l2, u2 = model(ids1.to(device),  mask1.to(device),
                                ids2.to(device),  mask2.to(device))
        L1.append(l1.cpu().numpy());  U1.append(u1.cpu().numpy())
        L2.append(l2.cpu().numpy());  U2.append(u2.cpu().numpy())
        LABS.append(labels.numpy())
    return (np.concatenate(L1), np.concatenate(U1),
            np.concatenate(L2), np.concatenate(U2),
            np.concatenate(LABS))


def eval_binary(l1, u1, l2, u2, labels, scale=5.0) -> dict:
    """Binary entailment metrics from numpy box arrays."""
    viol  = (np.maximum(0, l2 - l1) + np.maximum(0, u1 - u2)).sum(axis=-1)
    p_ent = 1.0 / (1.0 + scale * viol)
    return compute_nli_metrics(p_ent, labels)


def print_viol_stats(l1, u1, l2, u2, labels, title="Violation geometry"):
    """
    Print viol_fwd / viol_bwd / asymmetry median per class.
    Used to track how fine-tuning reshapes the embedding geometry relative
    to the frozen-SBERT baseline.
    """
    viol_fwd = (np.maximum(0, l2-l1) + np.maximum(0, u1-u2)).sum(axis=-1)
    viol_bwd = (np.maximum(0, l1-l2) + np.maximum(0, u2-u1)).sum(axis=-1)
    print(f"\n  {title}")
    print(f"  {'class':>8}  {'N':>5}  {'fwd_p50':>8}  {'bwd_p50':>8}  "
          f"{'asym_p50':>9}  (frozen-SBERT: ent=+12.0  neu=−1.6  con=−1.5)")
    for c, nm in [(0, 'entail'), (1, 'neutral'), (2, 'contra')]:
        m = labels == c
        if m.sum() == 0:
            continue
        fwd  = float(np.median(viol_fwd[m]))
        bwd  = float(np.median(viol_bwd[m]))
        asym = float(np.median(viol_bwd[m] - viol_fwd[m]))
        print(f"  {nm:>8}  {m.sum():>5}  {fwd:>8.2f}  {bwd:>8.2f}  {asym:>+9.2f}")


# ── LR schedule: linear warmup → cosine decay ─────────────────────────────────

def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))
    return LambdaLR(optimizer, lr_lambda)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Force line-buffered stdout so progress prints appear immediately even
    # when output is redirected to a file (default is block-buffered on CPU).
    sys.stdout.reconfigure(line_buffering=True)

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n{'='*70}")
    print(f"  Galois-NLP  ×  Fine-tuning all-MiniLM-L6-v2  ×  SNLI")
    print(f"{'='*70}")
    print(f"  Device  : {device}")
    print(f"  Boxes   : dim={cfg.box_dim}  hidden={cfg.hidden_dim}")
    print(f"  Freeze  : bottom {cfg.freeze_layers}/6 BERT layers + embeddings")
    print(f"  LR      : BERT={cfg.lr_bert:.0e}  head={cfg.lr_head:.0e}")
    print(f"  Batch   : {cfg.batch_size} × {cfg.accumulate} = "
          f"{cfg.batch_size * cfg.accumulate} effective")
    print(f"  Losses  : λ2={cfg.lambda2} hinge  "
          f"λ3={cfg.lambda3} BCE  λ5={cfg.lambda5} 3-class")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading & tokenising SNLI …")
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

    eval_bs    = cfg.batch_size * 2     # no grad → larger batch for eval
    tr_loader  = DataLoader(tr_ds,  batch_size=cfg.batch_size,
                            shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=eval_bs,
                            shuffle=False, num_workers=0)
    te_loader  = DataLoader(te_ds,  batch_size=eval_bs,
                            shuffle=False, num_workers=0)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print(f"\n[2/5] Building model …")
    model = GaloisSBERTModel(cfg).to(device)
    n_tot   = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total params: {n_tot/1e6:.1f}M  |  Trainable: {n_train/1e6:.1f}M")

    # ── 3. Optimiser + LR schedule ────────────────────────────────────────────
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

    print(f"\n[3/5] Training …")
    print(f"      steps/epoch={steps_per_epoch}  "
          f"total={total_steps}  warmup={warmup_steps}  "
          f"patience={cfg.patience}")

    hdr = (f"\n  {'Ep':>3}  {'Loss':>7}  "
           f"{'Hinge':>6}  {'BCE':>6}  {'3cls':>6}  "
           f"{'ValAcc':>7}  {'ValF1':>6}  {'t':>5}")
    print(hdr)
    print("  " + "─" * 58)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val_acc = -1.0
    best_state   = None
    pat_ctr      = 0
    history      = []
    global_step  = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        ep_loss = ep_h = ep_b = ep_3 = 0.0
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad()    # clear any leftover grads from previous epoch

        for step_i, (ids1, mask1, ids2, mask2, labels) in enumerate(tr_loader):
            ids1   = ids1.to(device);   mask1 = mask1.to(device)
            ids2   = ids2.to(device);   mask2 = mask2.to(device)
            labels = labels.to(device)

            l1, u1, l2, u2 = model(ids1, mask1, ids2, mask2)

            l_h = loss_galois_hinge(l1, u1, l2, u2, labels, cfg.margin)
            l_b = loss_task_bce(l1, u1, l2, u2, labels, cfg.scale)
            l_3 = loss_galois_3class(
                l1, u1, l2, u2, labels,
                ent_bwd_target  = cfg.ent_bwd_target,
                neu_fwd_margin  = cfg.neu_fwd_margin,
                neu_bwd_ceiling = cfg.neu_bwd_ceiling,
                con_fwd_margin  = cfg.con_fwd_margin,
                overlap_weight  = cfg.overlap_weight,
            )
            loss = cfg.lambda2 * l_h + cfg.lambda3 * l_b + cfg.lambda5 * l_3

            # Divide by accumulate so the effective gradient magnitude is
            # the same as computing the loss over the full effective batch.
            (loss / cfg.accumulate).backward()

            ep_loss += loss.item()
            ep_h    += l_h.item()
            ep_b    += l_b.item()
            ep_3    += l_3.item()
            n_batches += 1

            if (step_i + 1) % cfg.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Validation: binary metrics only (fast, ~10-15s)
        model.eval()
        l1v, u1v, l2v, u2v, labv = encode_all(model, val_loader, device)
        vm      = eval_binary(l1v, u1v, l2v, u2v, labv, cfg.scale)
        val_acc = vm['accuracy']
        val_f1  = vm['f1']

        elapsed = time.time() - t0
        m = lambda x: x / max(n_batches, 1)
        print(f"  {epoch:>3}  {m(ep_loss):>7.4f}  "
              f"{m(ep_h):>6.3f}  {m(ep_b):>6.3f}  {m(ep_3):>6.3f}  "
              f"{val_acc:>7.4f}  {val_f1:>6.4f}  {elapsed:>4.0f}s")

        history.append({
            'epoch':   epoch,
            'loss':    m(ep_loss),
            'hinge':   m(ep_h),
            'bce':     m(ep_b),
            '3class':  m(ep_3),
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
                      f"(best val acc={best_val_acc:.4f})")
                break

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    print(f"\n[4/5] Loading best checkpoint (val acc={best_val_acc:.4f}) …")
    model.load_state_dict(best_state)
    model.eval()

    print(f"\n[5/5] Test-set evaluation …")
    # Encode full training set (for probe fitting) and test set
    print(f"      Encoding training set for probe fitting …")
    l1_tr, u1_tr, l2_tr, u2_tr, lab_tr = encode_all(model, tr_loader, device)
    print(f"      Encoding test set …")
    l1_te, u1_te, l2_te, u2_te, lab_te = encode_all(model, te_loader, device)

    # Binary metrics
    te_bin = eval_binary(l1_te, u1_te, l2_te, u2_te, lab_te, cfg.scale)

    # 3-class linear probe (9 features, 400-epoch SGD)
    print(f"      Fitting 3-class linear probe (9-D features) …")
    feat_tr = compute_3class_features(l1_tr, u1_tr, l2_tr, u2_tr)
    feat_te = compute_3class_features(l1_te, u1_te, l2_te, u2_te)
    probe   = ThreeClassProbe(n_epochs=400, lr=0.05).fit(feat_tr, lab_tr)
    te_3    = probe.evaluate(feat_te, lab_te)

    # Violation geometry — key diagnostic for whether fine-tuning worked
    print_viol_stats(l1_te, u1_te, l2_te, u2_te, lab_te,
                     title="Violation geometry after fine-tuning (test set)")

    # ── Results ──────────────────────────────────────────────────────────────
    pc = te_3['per_class']
    cm = te_3['confusion']

    print(f"\n{'='*70}")
    print(f"  Fine-tuned SBERT + Galois-NLP  —  SNLI Test Results")
    print(f"{'='*70}")
    print(f"\n  Binary entailment vs rest:")
    print(f"    Acc={te_bin['accuracy']:.4f}  F1={te_bin['f1']:.4f}  "
          f"AUC={te_bin['auc']:.4f}")
    print(f"\n  3-class (9-feature linear probe):  {te_3['acc_3class']:.4f}")
    print(f"    ent={pc['entail']:.3f}  neu={pc['neutral']:.3f}  "
          f"con={pc['contra']:.3f}")
    print(f"\n  Confusion matrix  [rows=true, cols=pred]  "
          f"(0=entail  1=neutral  2=contra)")
    print(f"  {'':>10}  pred_ent  pred_neu  pred_con")
    for i, nm in enumerate(['true_ent', 'true_neu', 'true_con']):
        print(f"  {nm:>10}  {cm[i,0]:>8d}  {cm[i,1]:>8d}  {cm[i,2]:>8d}")

    print(f"\n  Comparison with frozen-SBERT run:")
    print(f"  {'Model':<42}  {'Bin-Acc':>7}  {'3-Acc':>6}  "
          f"{'ent':>5}  {'neu':>5}  {'con':>5}")
    print("  " + "─" * 70)
    # Frozen-SBERT baseline numbers from run_sbert_snli.py
    frozen_rows = [
        ("Cosine (frozen)",              "0.5056", "0.5677", "0.737", "0.240", "0.720"),
        ("Order Embeddings (frozen)",    "0.6902", "0.5127", "0.696", "0.191", "0.645"),
        ("Galois no-CEGAR (frozen)",     "0.7386", "0.5585", "0.586", "0.633", "0.454"),
        ("Galois CEGAR warmup=5 (frozen)","0.7436","0.5549", "0.581", "0.656", "0.424"),
    ]
    for nm, ba, ta, e, n, c in frozen_rows:
        print(f"  {nm:<42}  {ba:>7}  {ta:>6}  {e:>5}  {n:>5}  {c:>5}")
    print(f"  {'Fine-tuned SBERT + Galois':<42}  "
          f"{te_bin['accuracy']:>7.4f}  {te_3['acc_3class']:>6.4f}  "
          f"{pc['entail']:>5.3f}  {pc['neutral']:>5.3f}  {pc['contra']:>5.3f}")

    # Save
    results = {
        'config':       asdict(cfg),
        'best_val_acc': float(best_val_acc),
        'test_binary':  {k: float(v) for k, v in te_bin.items()},
        'test_3class': {
            'acc_3class': float(te_3['acc_3class']),
            'per_class':  {k: float(v) for k, v in pc.items()},
            'confusion':  cm.tolist(),
        },
        'history': history,
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Model   → {save_dir}/best_model.pt")
    print(f"  Results → {save_dir}/results.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
