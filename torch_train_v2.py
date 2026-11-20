"""
torch_train_v2.py  —  Improved fine-tuning: wider margins, more data, 3 trainable layers
=========================================================================================
Version 2 improvements over v1 (based on v1 error analysis):

  Problem diagnosed in v1:
    • 139/639 true-contra predicted as entailment (viol_fwd too low after training)
    • neutral (fwd=2.47, bwd=3.66) and contra (fwd=4.15, bwd=5.62) too close in
      violation space — only 1.68-pt fwd gap and 1.96-pt bwd gap.

  Fixes applied:
    1. con_fwd_margin  5.0 → 8.0   (force contra viol_fwd much higher)
    2. con_bwd_floor   NEW  = 6.0   (also push contra viol_bwd ≥ 6.0, separating
                                     from neutral bwd~3.7; entailment bwd stays ~11)
    3. neu_fwd_margin  3.0 → 4.0   (push neutral fwd up to widen gap with entailment)
    4. neu_bwd_ceiling 6.0 → 5.0   (tighten neutral bwd ceiling for cleaner region)
    5. lambda5         0.5 → 1.0   (give 3-class loss equal weight with BCE)
    6. n_train         6k  → 12k   (double training data for better margin coverage)
    7. freeze_layers   4   → 3     (unfreeze 3 BERT layers for more geometric freedom)
    8. patience        5   → 7     (allow more training epochs)

  Expected effect on probe geometry:
    After v2:  ent fwd~0, neu fwd~4-5, con fwd~8+     (gap: 4 pts vs v1's 1.7)
               ent bwd~11, neu bwd~3-4, con bwd~6-8   (gap: 2-3 pts vs v1's 2)
    → Linear probe should separate all 3 classes cleanly.

Architecture (same as v1)
--------------------------
  all-MiniLM-L6-v2  (freeze layers 0-2; train layers 3-5 + BoxEncoder)
  BoxEncoder: 384 → LayerNorm+ReLU(512) → (center, half-width) → (l,u) ∈ R^128

v1 test results (baseline to beat):
  Binary Acc=0.7945  F1=0.6694  AUC=0.8746
  3-class=0.5977   ent=0.715  neu=0.587  con=0.484
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
    freeze_layers:   int   = 3      # v2: 3 frozen (was 4), train layers 3-5

    # Data
    n_train:         int   = 12000  # v2: doubled (was 6000)
    n_val:           int   = 2000
    n_test:          int   = 2000
    max_len:         int   = 32

    # Training
    batch_size:      int   = 32
    accumulate:      int   = 4      # effective_batch = 128
    n_epochs:        int   = 20
    patience:        int   = 7      # v2: increased from 5
    warmup_ratio:    float = 0.1

    # Layer-wise LR
    lr_bert:         float = 2e-5
    lr_head:         float = 5e-4
    weight_decay:    float = 0.01

    # Loss weights
    lambda2:         float = 2.0    # binary galois hinge
    lambda3:         float = 1.0    # binary BCE
    lambda5:         float = 1.0    # v2: 3-class directional hinge (was 0.5)

    # Binary loss params
    margin:          float = 1.0
    scale:           float = 5.0

    # 3-class margins — v2: recalibrated from v1 error analysis
    #   v1 geometry (fine-tuned):  ent fwd=0.08, neu fwd=2.47, con fwd=4.15
    #                              ent bwd=11.26, neu bwd=3.66, con bwd=5.62
    #   Goal: ent fwd~0, neu fwd~4-5, con fwd~8+
    #          ent bwd~11, neu bwd~3-4, con bwd~6-8
    ent_bwd_target:  float = 10.0
    neu_fwd_margin:  float = 4.0    # v2: was 3.0
    neu_bwd_ceiling: float = 5.0    # v2: was 6.0  (tighter ceiling)
    con_fwd_margin:  float = 8.0    # v2: was 5.0  (force contra fwd much higher)
    con_bwd_floor:   float = 6.0    # v2: NEW — push contra bwd ≥ 6.0
    overlap_weight:  float = 0.1

    seed:            int   = 42
    save_dir:        str   = 'results/torch_models_v2'


# ── Model ──────────────────────────────────────────────────────────────────────

class BoxEncoder(nn.Module):
    """384 → LayerNorm+ReLU(512) → (center, half-width) → box (l, u) ∈ R^box_dim."""

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
        nn.init.constant_(self.width_head.bias, -1.0)

    def forward(self, x: torch.Tensor):
        h  = self.net(x)
        c  = self.center_head(h)
        hw = F.softplus(self.width_head(h))
        return c - hw, c + hw


class GaloisSBERTModel(nn.Module):
    """Partially frozen BERT + Galois box head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(cfg.model_name)
        self.box_enc = BoxEncoder(
            self.bert.config.hidden_size, cfg.hidden_dim, cfg.box_dim)
        self.scale = cfg.scale
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
                   attn_mask: torch.Tensor) -> torch.Tensor:
        mask_exp = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (torch.sum(last_hidden * mask_exp, dim=1) /
                torch.clamp(mask_exp.sum(dim=1), min=1e-9))

    def encode_sentence(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple:
        out = self.bert(input_ids=ids, attention_mask=mask)
        h   = self._mean_pool(out.last_hidden_state, mask)
        return self.box_enc(h)

    def forward(self, ids1, mask1, ids2, mask2):
        l1, u1 = self.encode_sentence(ids1, mask1)
        l2, u2 = self.encode_sentence(ids2, mask2)
        return l1, u1, l2, u2

    def bert_params(self):
        return [p for p in self.bert.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.box_enc.parameters())


# ── Loss functions ──────────────────────────────────────────────────────────────

def box_viol(l1: torch.Tensor, u1: torch.Tensor,
             l2: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    return (F.relu(l2 - l1) + F.relu(u1 - u2)).sum(dim=-1)


def loss_galois_hinge(l1, u1, l2, u2, labels, margin: float = 1.0):
    viol   = box_viol(l1, u1, l2, u2)
    binary = (labels == 0).float()
    return (binary * F.relu(viol) +
            (1 - binary) * F.relu(margin - viol)).mean()


def loss_task_bce(l1, u1, l2, u2, labels, scale: float = 5.0):
    viol   = box_viol(l1, u1, l2, u2)
    p_ent  = torch.sigmoid(-viol * scale)
    binary = (labels == 0).float()
    return F.binary_cross_entropy(p_ent.clamp(1e-7, 1 - 1e-7), binary)


def loss_galois_3class(l1, u1, l2, u2, labels,
                       ent_bwd_target:  float = 10.0,
                       neu_fwd_margin:  float = 4.0,
                       neu_bwd_ceiling: float = 5.0,
                       con_fwd_margin:  float = 8.0,
                       con_bwd_floor:   float = 6.0,
                       overlap_weight:  float = 0.1) -> torch.Tensor:
    """
    3-class directional hinge loss — v2 with wider margins and con_bwd_floor.

    v2 margin targets (based on v1 fine-tuned geometry + gap widening):
      Entailment   : viol_fwd → 0    viol_bwd → ≥10  (strong asymmetry)
      Neutral      : viol_fwd ≥ 4.0  viol_bwd ≤  5.0  (moderate symmetric)
      Contradiction: viol_fwd ≥ 8.0  viol_bwd ≥  6.0  (both directions violated)

    The new con_bwd_floor pushes contradiction viol_bwd above neutral's ceiling,
    creating a clean separation gap in both fwd and bwd dimensions.
    """
    viol_fwd = box_viol(l1, u1, l2, u2)
    viol_bwd = box_viol(l2, u2, l1, u1)
    overlap  = F.relu(torch.minimum(u1, u2) -
                      torch.maximum(l1, l2)).sum(dim=-1)

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
        0.5 * F.relu(con_bwd_floor - viol_bwd) +   # v2: NEW — push contra bwd ≥ floor
        overlap * overlap_weight
    )
    return (l_ent + l_neu + l_con).mean()


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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.ids1[i], self.mask1[i],
                self.ids2[i], self.mask2[i], self.labels[i])


# ── Evaluation helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(model: GaloisSBERTModel, loader: DataLoader, device) -> tuple:
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
    viol  = (np.maximum(0, l2 - l1) + np.maximum(0, u1 - u2)).sum(axis=-1)
    p_ent = 1.0 / (1.0 + scale * viol)
    return compute_nli_metrics(p_ent, labels)


def print_viol_stats(l1, u1, l2, u2, labels, title="Violation geometry"):
    viol_fwd = (np.maximum(0, l2-l1) + np.maximum(0, u1-u2)).sum(axis=-1)
    viol_bwd = (np.maximum(0, l1-l2) + np.maximum(0, u2-u1)).sum(axis=-1)
    print(f"\n  {title}")
    print(f"  {'class':>8}  {'N':>5}  {'fwd_p50':>8}  {'bwd_p50':>8}  "
          f"{'asym_p50':>9}  (v1: ent fwd=0.08 bwd=11.26  neu fwd=2.47 bwd=3.66  con fwd=4.15 bwd=5.62)")
    for c, nm in [(0, 'entail'), (1, 'neutral'), (2, 'contra')]:
        m = labels == c
        if m.sum() == 0:
            continue
        fwd  = float(np.median(viol_fwd[m]))
        bwd  = float(np.median(viol_bwd[m]))
        asym = float(np.median(viol_bwd[m] - viol_fwd[m]))
        print(f"  {nm:>8}  {m.sum():>5}  {fwd:>8.2f}  {bwd:>8.2f}  {asym:>+9.2f}")


def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))
    return LambdaLR(optimizer, lr_lambda)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n{'='*70}")
    print(f"  Galois-NLP v2  ×  Fine-tuning all-MiniLM-L6-v2  ×  SNLI")
    print(f"{'='*70}")
    print(f"  Device      : {device}")
    print(f"  Boxes       : dim={cfg.box_dim}  hidden={cfg.hidden_dim}")
    print(f"  Freeze      : bottom {cfg.freeze_layers}/6 BERT layers + embeddings")
    print(f"  LR          : BERT={cfg.lr_bert:.0e}  head={cfg.lr_head:.0e}")
    print(f"  Batch       : {cfg.batch_size} × {cfg.accumulate} = "
          f"{cfg.batch_size * cfg.accumulate} effective")
    print(f"  Losses      : λ2={cfg.lambda2} hinge  "
          f"λ3={cfg.lambda3} BCE  λ5={cfg.lambda5} 3-class")
    print(f"  3-cls margins: neu_fwd≥{cfg.neu_fwd_margin}  neu_bwd≤{cfg.neu_bwd_ceiling}  "
          f"con_fwd≥{cfg.con_fwd_margin}  con_bwd≥{cfg.con_bwd_floor}")

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

    eval_bs    = cfg.batch_size * 2
    tr_loader  = DataLoader(tr_ds,  batch_size=cfg.batch_size,
                            shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=eval_bs,
                            shuffle=False, num_workers=0)
    te_loader  = DataLoader(te_ds,  batch_size=eval_bs,
                            shuffle=False, num_workers=0)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print(f"\n[2/5] Building model …")
    model   = GaloisSBERTModel(cfg).to(device)
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
        optimizer.zero_grad()

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
                con_bwd_floor   = cfg.con_bwd_floor,
                overlap_weight  = cfg.overlap_weight,
            )
            loss = cfg.lambda2 * l_h + cfg.lambda3 * l_b + cfg.lambda5 * l_3

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
    print(f"      Encoding training set for probe fitting …")
    l1_tr, u1_tr, l2_tr, u2_tr, lab_tr = encode_all(model, tr_loader, device)
    print(f"      Encoding test set …")
    l1_te, u1_te, l2_te, u2_te, lab_te = encode_all(model, te_loader, device)

    te_bin = eval_binary(l1_te, u1_te, l2_te, u2_te, lab_te, cfg.scale)

    print(f"      Fitting 3-class linear probe (9-D features) …")
    feat_tr = compute_3class_features(l1_tr, u1_tr, l2_tr, u2_tr)
    feat_te = compute_3class_features(l1_te, u1_te, l2_te, u2_te)
    probe   = ThreeClassProbe(n_epochs=400, lr=0.05).fit(feat_tr, lab_tr)
    te_3    = probe.evaluate(feat_te, lab_te)

    print_viol_stats(l1_te, u1_te, l2_te, u2_te, lab_te,
                     title="Violation geometry after fine-tuning v2 (test set)")

    pc = te_3['per_class']
    cm = te_3['confusion']

    print(f"\n{'='*70}")
    print(f"  Fine-tuned SBERT + Galois-NLP v2  —  SNLI Test Results")
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

    print(f"\n  Full comparison table:")
    print(f"  {'Model':<44}  {'Bin-Acc':>7}  {'3-Acc':>6}  "
          f"{'ent':>5}  {'neu':>5}  {'con':>5}")
    print("  " + "─" * 72)
    all_rows = [
        ("Cosine (frozen SBERT)",          "0.5056", "0.5677", "0.737", "0.240", "0.720"),
        ("Order Embeddings (frozen)",       "0.6902", "0.5127", "0.696", "0.191", "0.645"),
        ("Galois no-CEGAR (frozen SBERT)",  "0.7386", "0.5585", "0.586", "0.633", "0.454"),
        ("Galois CEGAR warmup=5 (frozen)",  "0.7436", "0.5549", "0.581", "0.656", "0.424"),
        ("Fine-tuned SBERT+Galois v1",      "0.7945", "0.5977", "0.715", "0.587", "0.484"),
    ]
    for nm, ba, ta, e, n, c in all_rows:
        print(f"  {nm:<44}  {ba:>7}  {ta:>6}  {e:>5}  {n:>5}  {c:>5}")
    print(f"  {'Fine-tuned SBERT+Galois v2':<44}  "
          f"{te_bin['accuracy']:>7.4f}  {te_3['acc_3class']:>6.4f}  "
          f"{pc['entail']:>5.3f}  {pc['neutral']:>5.3f}  {pc['contra']:>5.3f}")

    results = {
        'version':      'v2',
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
