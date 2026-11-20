"""
cegar_corrected.py — Corrected CEGAR with Spurious Counterexample Filtering
============================================================================
Three changes from original CEGAR:
  1. Confidence threshold raised to 0.90 (filters boundary noise)
  2. Spurious counterexample detection via calibrated viol_fwd threshold
  3. Asymmetric loss weighting (FP=1.0, FN=0.3)

Two models:
  Model A: full WordNet GloVe, no CEGAR        (trained fresh, same seed)
  Model B: full WordNet GloVe, corrected CEGAR  (trained with corrected buffer)

Usage
-----
  PYTHONIOENCODING=utf-8 python -u cegar_corrected.py
"""

import sys, os, time, json, copy
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import kendalltau, spearmanr

import gensim.downloader as api

from src.improved_trainer import (
    ImprovedTrainer, GaloisModel, Adam,
    relu, relu_grad, softplus, softplus_grad,
    layer_norm, layer_norm_backward,
)
from src.box_lattice import box_containment_violation, box_containment_score
from src.losses import ENTAILMENT, CEGARBuffer
from src.dataset import NLIDataLoader
from src.metrics import compute_nli_metrics, compute_box_stats

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from eval_hyperlex_embeddings import (
    download_hyperlex, load_hyperlex_splits, filter_nouns,
    get_wv_embedding, DATA_DIR, BOX_DIM, HIDDEN_DIM,
)
from cegar_experiment import (
    get_full_wordnet_pairs_with_hops, evaluate_per_bucket,
    bootstrap_delta_tau, get_wordnet_hop,
)

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Corrected CEGAR Buffer
# ══════════════════════════════════════════════════════════════════════════════

class CorrectedCEGARBuffer:
    """
    CEGAR buffer with three fixes:
    1. Confidence filter (threshold=0.90) — removes boundary noise
    2. Spurious counterexample filter — removes geometrically correct predictions
    3. Moving window expiry — removes stale counterexamples

    Only genuine geometric failures enter the buffer.
    """

    def __init__(self, max_size=2000, expiry_steps=300):
        self.max_size      = max_size
        self.expiry_steps  = expiry_steps
        self.pairs         = []  # (l1, u1, l2, u2, label, step, error_type)
        self.current_step  = 0

        # Diagnostics
        self.n_added_total       = 0
        self.n_rejected_conf     = 0
        self.n_rejected_spurious = 0
        self.n_fp_added          = 0
        self.n_fn_added          = 0

    def step(self):
        self.current_step += 1

    def add(self, l1, u1, l2, u2, labels, pred_scores,
            confidence_threshold, spurious_threshold):
        """
        Add genuine geometric failures to buffer.

        A pair is added only if:
        1. Model was confidently wrong (score > conf_threshold or < 1-conf_threshold)
        2. The error is not spurious (geometry genuinely wrong, not boundary noise)
        """
        binary      = (labels == ENTAILMENT).astype(int)  # 1 if IS-A
        pred_labels = (pred_scores > 0.5).astype(int)
        wrong       = (pred_labels != binary)

        # Filter 1: confidence
        confident = (
            (pred_scores > confidence_threshold) |
            (pred_scores < 1.0 - confidence_threshold)
        )

        self.n_rejected_conf += int(np.sum(wrong & ~confident))
        candidates = np.where(wrong & confident)[0]

        for i in candidates:
            viol_fwd = float(box_containment_violation(
                l1[i:i+1], u1[i:i+1], l2[i:i+1], u2[i:i+1]))
            is_entailment = binary[i] == 1

            # Filter 2: spurious detection
            if is_entailment:
                # False negative: model says no IS-A, truth says yes
                # Spurious if viol_fwd already low (geometry approximately right)
                if viol_fwd < spurious_threshold:
                    self.n_rejected_spurious += 1
                    continue
                error_type = 'FN'
                self.n_fn_added += 1
            else:
                # False positive: model says IS-A, truth says no
                # Spurious if viol_fwd already high (geometry approximately right)
                if viol_fwd > spurious_threshold * 3:  # 3x for non-entailment
                    self.n_rejected_spurious += 1
                    continue
                error_type = 'FP'
                self.n_fp_added += 1

            self.pairs.append((
                l1[i].copy(), u1[i].copy(),
                l2[i].copy(), u2[i].copy(),
                labels[i], self.current_step, error_type
            ))
            self.n_added_total += 1

        # Enforce max size (keep most recent)
        if len(self.pairs) > self.max_size:
            self.pairs = self.pairs[-self.max_size:]

    def sample(self, n):
        # Expire stale entries
        self.pairs = [
            p for p in self.pairs
            if self.current_step - p[5] <= self.expiry_steps
        ]

        if len(self.pairs) == 0:
            return None

        idx = np.random.choice(
            len(self.pairs),
            size=min(n, len(self.pairs)),
            replace=False)
        batch = [self.pairs[i] for i in idx]

        return (
            np.stack([b[0] for b in batch]),
            np.stack([b[1] for b in batch]),
            np.stack([b[2] for b in batch]),
            np.stack([b[3] for b in batch]),
            np.array([b[4] for b in batch]),
            [b[6] for b in batch],   # error types for diagnostics
        )

    def diagnostics(self):
        active = [p for p in self.pairs
                  if self.current_step - p[5] <= self.expiry_steps]
        fp_active = sum(1 for p in active if p[6] == 'FP')
        fn_active = sum(1 for p in active if p[6] == 'FN')
        n_active  = len(active)

        total_considered = (self.n_added_total + self.n_rejected_conf
                            + self.n_rejected_spurious)
        rej_rate = ((self.n_rejected_conf + self.n_rejected_spurious)
                    / max(total_considered, 1))

        print(f"      CEGAR Buffer (step {self.current_step}): "
              f"active={n_active} "
              f"(FP={fp_active} [{100*fp_active/max(n_active,1):.1f}%], "
              f"FN={fn_active} [{100*fn_active/max(n_active,1):.1f}%])  "
              f"rej_rate={rej_rate:.1%} "
              f"(conf={self.n_rejected_conf}, spur={self.n_rejected_spurious})")
        return {
            'active': n_active,
            'fp_active': fp_active,
            'fn_active': fn_active,
            'rejection_rate': rej_rate,
        }

    def __len__(self):
        return len(self.pairs)


# ══════════════════════════════════════════════════════════════════════════════
#  Corrected CEGAR Loss
# ══════════════════════════════════════════════════════════════════════════════

def cegar_loss_corrected(buffer_sample, fn_weight=0.3, fp_weight=1.0,
                         margin=1.5):
    """
    Asymmetric CEGAR loss based on error type.
    FP: full weight — model confidently puts non-IS-A into containment.
    FN: downweighted — harder to fix, noisier signal.
    """
    if buffer_sample is None:
        return 0.0, np.zeros(0), np.zeros(0)

    l1, u1, l2, u2, labels, error_types = buffer_sample
    viol   = box_containment_violation(l1, u1, l2, u2)
    binary = (labels == ENTAILMENT).astype(float)  # 1 if IS-A

    # Per-sample weights
    weights = np.array([
        fp_weight if et == 'FP' else fn_weight
        for et in error_types
    ])

    # FN loss: push toward containment (reduce viol)
    fn_loss = binary * np.maximum(0.0, viol)
    # FP loss: push away from containment (increase viol)
    fp_loss = (1 - binary) * np.maximum(0.0, margin - viol)

    weighted = weights * (fn_loss + fp_loss)
    return float(np.mean(weighted)), viol, binary


# ══════════════════════════════════════════════════════════════════════════════
#  Calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_spurious_threshold(model, val_emb1, val_emb2, val_labels,
                                  percentile=75):
    """
    Set spurious threshold at pXX of viol_fwd for true entailment pairs.
    p75: keeps top 25% of entailment violations as genuine failures.
    """
    ent_mask = val_labels == ENTAILMENT
    if ent_mask.sum() == 0:
        return 0.5  # fallback

    l1, u1, _ = model.encode(val_emb1[ent_mask])
    l2, u2, _ = model.encode(val_emb2[ent_mask])
    viols = box_containment_violation(l1, u1, l2, u2)

    threshold = float(np.percentile(viols, percentile))
    print(f"\n      Spurious threshold calibration (p{percentile} of entailment viols):")
    print(f"        Threshold = {threshold:.4f}")
    for p in [25, 50, 75, 90, 95]:
        print(f"        p{p}: {np.percentile(viols, p):.4f}")
    return threshold


# ══════════════════════════════════════════════════════════════════════════════
#  Precision Tracking
# ══════════════════════════════════════════════════════════════════════════════

def measure_abstraction_precision(model, emb1, emb2, labels, scale=5.0):
    """
    Precision = median(viol_fwd for non-IS-A) - median(viol_fwd for IS-A)
    Higher = better separation = more precise abstraction.
    """
    l1, u1, _ = model.encode(emb1)
    l2, u2, _ = model.encode(emb2)
    viols = box_containment_violation(l1, u1, l2, u2)

    ent_mask  = labels == ENTAILMENT
    nent_mask = ~ent_mask

    med_ent  = float(np.median(viols[ent_mask]))  if ent_mask.any()  else 0.0
    med_nent = float(np.median(viols[nent_mask])) if nent_mask.any() else 0.0
    precision = med_nent - med_ent
    return precision, med_ent, med_nent


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Trainer with Corrected CEGAR
# ══════════════════════════════════════════════════════════════════════════════

class CorrectedCEGARTrainer:
    """
    Wraps GaloisModel + Adam with corrected CEGAR buffer.
    Identical to ImprovedTrainer except for CEGAR logic.
    """

    def __init__(self, input_dim, hidden_dim=256, box_dim=64, lr=1e-3,
                 lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
                 margin=1.0, scale=5.0, cegar_margin=1.5,
                 warmup_epochs=3,
                 confidence_threshold=0.90,
                 spurious_threshold=0.5,
                 fn_weight=0.3, fp_weight=1.0,
                 cegar_max_size=2000, cegar_expiry=300,
                 rng=None):
        self.rng    = rng or np.random.default_rng(42)
        self.model  = GaloisModel(input_dim, hidden_dim, box_dim, self.rng)
        self.opt    = Adam(lr=lr, wd=1e-5)
        self.scale  = scale
        self.margin = margin
        self.cegar_margin  = cegar_margin
        self.warmup_epochs = warmup_epochs

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

        self.conf_threshold    = confidence_threshold
        self.spurious_threshold = spurious_threshold
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

        self.cegar = CorrectedCEGARBuffer(
            max_size=cegar_max_size, expiry_steps=cegar_expiry)

        self._epoch = 0
        self.history = []

    def _step(self, emb1, emb2, labels):
        """Single training step — identical to ImprovedTrainer except CEGAR."""
        λ1 = self.lambda1
        λ2 = self.lambda2
        λ3 = self.lambda3
        λ4 = self.lambda4 if self._epoch > self.warmup_epochs else 0.0
        N  = len(labels)
        binary = (labels == ENTAILMENT).astype(float)

        # ── Forward ──
        l1, u1, h1, cache1 = self.model._encode_fwd(emb1)
        l2, u2, h2, cache2 = self.model._encode_fwd(emb2)
        x1hat, dcache1     = self.model._decode_fwd(l1, u1)
        x2hat, dcache2     = self.model._decode_fwd(l2, u2)

        viol12 = box_containment_violation(l1, u1, l2, u2)
        p_ent  = 1.0 / (1.0 + self.scale * viol12)

        # ── Losses ──
        # 1. Reconstruction
        diff1  = x1hat - emb1; diff2 = x2hat - emb2
        l_rec  = 0.5 * (np.mean(diff1**2) + np.mean(diff2**2))

        # 2. Galois hinge
        ent_loss  = binary       * np.maximum(0.0, viol12)
        nent_loss = (1 - binary) * np.maximum(0.0, self.margin - viol12)
        l_gal     = float(np.mean(ent_loss + nent_loss))

        # 3. Task BCE
        p_task = np.clip(p_ent, 1e-7, 1-1e-7)
        l_tsk  = -float(np.mean(
            binary * np.log(p_task) + (1-binary) * np.log(1-p_task)))

        # 4. Corrected CEGAR
        # Add to buffer with confidence + spurious filtering
        self.cegar.add(l1, u1, l2, u2, labels, p_ent,
                       self.conf_threshold, self.spurious_threshold)
        self.cegar.step()

        l_ceg = 0.0
        dceg_dviol12 = np.zeros(N)

        if len(self.cegar) > 32 and λ4 > 0:
            sample = self.cegar.sample(64)
            if sample is not None:
                l_ceg, _, _ = cegar_loss_corrected(
                    sample,
                    fn_weight=self.fn_weight,
                    fp_weight=self.fp_weight,
                    margin=self.cegar_margin)

                # Proxy gradient on current batch
                dceg_dviol12 = (
                    binary       * (viol12 > 0).astype(float)
                    - (1-binary) * (viol12 < self.cegar_margin).astype(float)
                ) / N

        l_total = λ1*l_rec + λ2*l_gal + λ3*l_tsk + λ4*l_ceg

        # ── Backward ──
        gal_active_ent  = binary       * (viol12 > 0).astype(float)
        gal_active_nent = (1 - binary) * (viol12 < self.margin).astype(float)
        dL_dviol = (λ2 * (gal_active_ent - gal_active_nent)) / N

        dp_dviol  = -self.scale / ((1 + self.scale * viol12) ** 2)
        dBCE_dp   = -(binary / p_task - (1-binary) / (1-p_task)) / N
        dL_dviol += λ3 * dBCE_dp * dp_dviol

        # CEGAR proxy gradient
        dL_dviol += λ4 * dceg_dviol12

        indicator_l = (l2 > l1).astype(float)
        indicator_u = (u1 > u2).astype(float)

        dl1_task = -dL_dviol[:, None] * indicator_l
        du1_task =  dL_dviol[:, None] * indicator_u
        dl2_task =  dL_dviol[:, None] * indicator_l
        du2_task = -dL_dviol[:, None] * indicator_u

        dec_grads1, dl1_rec, du1_rec = self.model._backprop_decode(emb1, l1, u1, dcache1)
        dec_grads2, dl2_rec, du2_rec = self.model._backprop_decode(emb2, l2, u2, dcache2)

        dl1 = dl1_task + λ1 * dl1_rec
        du1 = du1_task + λ1 * du1_rec
        dl2 = dl2_task + λ1 * dl2_rec
        du2 = du2_task + λ1 * du2_rec

        enc_grads1 = self.model._backprop_encode(dl1, du1, cache1)
        enc_grads2 = self.model._backprop_encode(dl2, du2, cache2)
        enc_grads = {k: 0.5*(enc_grads1[k] + enc_grads2[k]) for k in enc_grads1}
        dec_grads = {k: 0.5*(dec_grads1[k] + dec_grads2[k])
                     for k in ['Wd1','bd1','Wd2','bd2']}

        all_grads = {**enc_grads, **dec_grads}
        params = self.model.params_dict()
        self.opt.update(params, all_grads)
        self.model.set_params(params)

        return {
            'total': l_total, 'recon': l_rec,
            'galois': l_gal,  'task': l_tsk,
            'cegar': λ4 * l_ceg,
        }

    def train(self, train_data, val_data, n_epochs=50, batch_size=256,
              verbose=True, patience=7, precision_baseline=None):
        loader     = NLIDataLoader(train_data, batch_size=batch_size,
                                    shuffle=True, rng=self.rng)
        best_acc   = -1
        best_state = None
        pat_ctr    = 0

        for epoch in range(1, n_epochs + 1):
            self._epoch = epoch

            # Clear buffer after warmup
            if epoch == self.warmup_epochs + 1 and self.warmup_epochs > 0:
                self.cegar = CorrectedCEGARBuffer(
                    max_size=self.cegar.max_size,
                    expiry_steps=self.cegar.expiry_steps)

            t0 = time.time()
            ep_losses = []

            for batch in loader:
                loss_d = self._step(batch['emb1'], batch['emb2'], batch['label'])
                ep_losses.append(loss_d)

            mean = {k: float(np.mean([l[k] for l in ep_losses])) for k in ep_losses[0]}
            val_m = self.evaluate(val_data)
            elapsed = time.time() - t0

            # Precision tracking
            prec, med_ent, med_nent = measure_abstraction_precision(
                self.model, val_data['emb1'], val_data['emb2'],
                val_data['label'], self.scale)

            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Loss={mean['total']:.4f} "
                      f"(R={mean['recon']:.3f} G={mean['galois']:.3f} "
                      f"T={mean['task']:.3f} C={mean['cegar']:.3f}) | "
                      f"Val Acc={val_m['accuracy']:.4f} F1={val_m['f1']:.4f} | "
                      f"Prec={prec:.4f} "
                      f"(ent={med_ent:.3f} nent={med_nent:.3f})",
                      end="")

                if precision_baseline is not None:
                    delta = prec - precision_baseline
                    print(f" Δ={delta:+.4f}", end="")

                print(f" | {elapsed:.1f}s")

                # Buffer diagnostics every 3 epochs
                if epoch % 3 == 0 or epoch == self.warmup_epochs + 1:
                    self.cegar.diagnostics()

            self.history.append({
                'epoch': epoch, **mean,
                'val_acc': val_m['accuracy'],
                'val_f1': val_m['f1'],
                'precision': prec,
                'med_ent': med_ent,
                'med_nent': med_nent,
            })

            # Early stopping: precision collapse check
            if (precision_baseline is not None and
                    epoch > self.warmup_epochs + 2 and
                    prec < precision_baseline - 0.5):
                print(f"  WARNING: precision dropped {prec:.4f} < baseline "
                      f"{precision_baseline:.4f} - 0.5. Reducing lambda4.")
                self.lambda4 *= 0.5
                if self.lambda4 < 0.01:
                    print(f"  CEGAR DISABLED — lambda4 reduced to {self.lambda4:.4f}")
                    self.lambda4 = 0.0
                    if epoch < 5:
                        print(f"  EARLY TERMINATION: CEGAR disabled before epoch 5.")
                        if best_state:
                            self.model.set_params(best_state)
                        return self.history

            if val_m['accuracy'] > best_acc:
                best_acc   = val_m['accuracy']
                best_state = {k: getattr(self.model, k).copy()
                              for k in self.model.params_dict()}
                pat_ctr = 0
            else:
                pat_ctr += 1
                if pat_ctr >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best Val Acc={best_acc:.4f})")
                    if best_state:
                        self.model.set_params(best_state)
                    break

        return self.history

    def evaluate(self, dataset):
        l1, u1, _ = self.model.encode(dataset['emb1'])
        l2, u2, _ = self.model.encode(dataset['emb2'])
        p_ent     = box_containment_score(l1, u1, l2, u2, scale=self.scale)
        metrics   = compute_nli_metrics(p_ent, dataset['label'])
        box_st    = compute_box_stats(l1, u1, l2, u2)
        return {**metrics, **box_st}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*90}")
    print(f"  Corrected CEGAR Experiment")
    print(f"{'='*90}")
    print(f"  Input: GloVe-wiki-gigaword-300")
    print(f"  Training: Full WordNet noun hypernymy (~74k edges)")
    print(f"  Model A: lambda4=0.0 (no CEGAR)")
    print(f"  Model C: lambda4=0.3 (corrected CEGAR: conf=0.90, spurious filter, "
          f"FP=1.0/FN=0.3)")
    print(f"  Seed: {SEED}")

    # ── 1. Load data ─────────────────────────────────────────────────────
    print(f"\n[1/7] Loading data...")
    download_hyperlex()
    _, _, test_pairs = load_hyperlex_splits()
    te_n = filter_nouns(test_pairs)
    gold_te = np.array([p['score_10'] for p in te_n])

    hop_map = {
        'hyp-1': '1', 'hyp-2': '2', 'hyp-3': '3', 'hyp-4': '4+',
    }

    print(f"\n      HyperLex test noun bucket sizes:")
    for bucket in ['1', '2', '3', '4+']:
        types_for_bucket = [k for k, v in hop_map.items() if v == bucket]
        n = sum(1 for p in te_n if p['type'] in types_for_bucket)
        print(f"        Hop {bucket}: {n} positive pairs")
    n_neg = sum(1 for p in te_n
                if not p['type'].startswith('hyp-') and not p['type'].startswith('r-hyp-'))
    print(f"        Negatives (cohyp/no-rel/syn/ant/mero): {n_neg}")

    # ── 2. Load GloVe ────────────────────────────────────────────────────
    print(f"\n[2/7] Loading GloVe...")
    t0 = time.time()
    glove = api.load('glove-wiki-gigaword-300')
    print(f"      Loaded in {time.time()-t0:.0f}s  vocab={len(glove)}")

    print(f"      Building GloVe lookup...")
    wn_words = set()
    for s in wn.all_synsets('n'):
        wn_words.add(s.lemmas()[0].name())
        for h in s.hypernyms():
            wn_words.add(h.lemmas()[0].name())
    hl_words = {p['word1'] for p in te_n} | {p['word2'] for p in te_n}
    all_words = wn_words | hl_words

    emb_lookup = {}
    oov = 0
    for w in all_words:
        v = get_wv_embedding(w, glove)
        if v is not None:
            emb_lookup[w] = v
        else:
            oov += 1
    print(f"      Words: {len(all_words)}  Embedded: {len(emb_lookup)}  OOV: {oov}")

    # ── 3. Generate training data ────────────────────────────────────────
    print(f"\n[3/7] Generating full WordNet training pairs...")
    rng = np.random.default_rng(SEED)
    wn_pairs, wn_hops = get_full_wordnet_pairs_with_hops(rng)

    # Build embedding arrays
    emb1_list, emb2_list, lab_list = [], [], []
    for w1, w2, lab in wn_pairs:
        v1 = emb_lookup.get(w1)
        v2 = emb_lookup.get(w2)
        if v1 is not None and v2 is not None:
            emb1_list.append(v1)
            emb2_list.append(v2)
            lab_list.append(lab)

    emb1 = np.stack(emb1_list)
    emb2 = np.stack(emb2_list)
    labs  = np.array(lab_list)
    input_dim = emb1.shape[1]
    print(f"      {len(labs)} pairs with embeddings (dim={input_dim})")

    # Train/val split (same as cegar_experiment.py)
    n_val = max(512, int(len(labs) * 0.1))
    split_rng = np.random.default_rng(SEED)
    idx = split_rng.permutation(len(labs))
    tr_idx, vl_idx = idx[n_val:], idx[:n_val]

    train_data = {'emb1': emb1[tr_idx], 'emb2': emb2[tr_idx], 'label': labs[tr_idx]}
    val_data   = {'emb1': emb1[vl_idx], 'emb2': emb2[vl_idx], 'label': labs[vl_idx]}
    print(f"      Train: {len(tr_idx)}  Val: {len(vl_idx)}")

    # ── 4. Train Model A (no CEGAR) ─────────────────────────────────────
    print(f"\n[4/7] Training Model A (no CEGAR)...")
    trainer_a = ImprovedTrainer(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        box_dim=BOX_DIM,
        lr=5e-4,
        lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
        margin=1.0, scale=5.0,
        cegar_margin=1.5,
        warmup_epochs=3,
        cegar_max_size=4000,
        rng=np.random.default_rng(SEED),
    )
    t0 = time.time()
    trainer_a.train(train_data, val_data, n_epochs=50, batch_size=256,
                    verbose=True, patience=7)
    print(f"      Done in {time.time()-t0:.1f}s")

    # ── 5. Calibrate spurious threshold ─────────────────────────────────
    print(f"\n[5/7] Calibrating spurious threshold from Model A...")
    spurious_threshold = calibrate_spurious_threshold(
        trainer_a.model, val_data['emb1'], val_data['emb2'],
        val_data['label'], percentile=75)

    # Measure baseline precision
    prec_baseline, med_ent_0, med_nent_0 = measure_abstraction_precision(
        trainer_a.model, val_data['emb1'], val_data['emb2'],
        val_data['label'], trainer_a.scale)
    print(f"\n      Baseline precision (Model A): {prec_baseline:.4f}")
    print(f"        ent median viol:  {med_ent_0:.4f}")
    print(f"        nent median viol: {med_nent_0:.4f}")

    # ── 6. Train Model C (corrected CEGAR) ──────────────────────────────
    print(f"\n[6/7] Training Model C (corrected CEGAR)...")
    print(f"      Hyperparameters:")
    print(f"        confidence_threshold = 0.90")
    print(f"        spurious_threshold   = {spurious_threshold:.4f}")
    print(f"        fn_weight = 0.3, fp_weight = 1.0")
    print(f"        lambda4 = 0.3, warmup = 3, buf_size = 2000, expiry = 300")

    trainer_c = CorrectedCEGARTrainer(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        box_dim=BOX_DIM,
        lr=5e-4,
        lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.3,
        margin=1.0, scale=5.0,
        cegar_margin=1.5,
        warmup_epochs=3,
        confidence_threshold=0.90,
        spurious_threshold=spurious_threshold,
        fn_weight=0.3, fp_weight=1.0,
        cegar_max_size=2000,
        cegar_expiry=300,
        rng=np.random.default_rng(SEED),
    )
    t0 = time.time()
    trainer_c.train(train_data, val_data, n_epochs=50, batch_size=256,
                    verbose=True, patience=7,
                    precision_baseline=prec_baseline)
    print(f"      Done in {time.time()-t0:.1f}s")

    # Final buffer diagnostics
    print(f"\n      Final buffer diagnostics (Model C):")
    buf_diag = trainer_c.cegar.diagnostics()

    # ── 7. Evaluate both models ─────────────────────────────────────────
    print(f"\n[7/7] Evaluating on HyperLex test (stratified by hop)...")
    results_a = evaluate_per_bucket(trainer_a, te_n, gold_te, emb_lookup, hop_map)
    results_c = evaluate_per_bucket(trainer_c, te_n, gold_te, emb_lookup, hop_map)

    # Bootstrap CIs
    print(f"\n      Computing bootstrap CIs (1000 resamples per bucket)...")
    bootstrap_rng = np.random.default_rng(999)
    ci_results = {}
    for bucket in ['1', '2', '3', '4+', 'ALL']:
        ra = results_a.get(bucket)
        rc = results_c.get(bucket)
        if ra is None or rc is None:
            ci_results[bucket] = None
            continue
        mean_d, ci_lo, ci_hi = bootstrap_delta_tau(
            ra['scores'], rc['scores'], ra['labels'],
            n_bootstrap=1000, rng=bootstrap_rng)
        ci_results[bucket] = {
            'mean_delta': mean_d, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'sig': ci_lo > 0 or ci_hi < 0,
        }

    # Final precision for Model C
    prec_c, med_ent_c, med_nent_c = measure_abstraction_precision(
        trainer_c.model, val_data['emb1'], val_data['emb2'],
        val_data['label'], trainer_c.scale)

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════════════

    # Also load original CEGAR results for comparison
    orig_cegar_path = Path('results/cegar_experiment.json')
    orig_cegar = None
    if orig_cegar_path.exists():
        with open(orig_cegar_path) as f:
            orig_cegar = json.load(f)

    print(f"\n\n{'='*90}")
    print(f"  TABLE 1: Overall Comparison")
    print(f"{'='*90}")
    print(f"\n  {'Model':<40} {'τ(all)':>8} {'Spearman':>10} {'Dir%':>8} {'Precision':>10}")
    print(f"  {'-'*76}")

    ra_all = results_a.get('ALL', {})
    rc_all = results_c.get('ALL', {})

    print(f"  {'Model A (no CEGAR)':<40} "
          f"{ra_all.get('tau', 0):>+8.4f} "
          f"{ra_all.get('rho', 0):>+10.4f} "
          f"{ra_all.get('dir_acc', 0):>7.1%} "
          f"{prec_baseline:>10.4f}")

    if orig_cegar:
        d_orig = orig_cegar.get('delta_tau_all', 0)
        print(f"  {'Model B (original CEGAR)':<40} "
              f"{ra_all.get('tau', 0) + d_orig:>+8.4f} "
              f"{'—':>10} "
              f"{'—':>8} "
              f"{'—':>10}  (Δτ={d_orig:+.3f})")

    print(f"  {'Model C (corrected CEGAR)':<40} "
          f"{rc_all.get('tau', 0):>+8.4f} "
          f"{rc_all.get('rho', 0):>+10.4f} "
          f"{rc_all.get('dir_acc', 0):>7.1%} "
          f"{prec_c:>10.4f}")

    delta_prec = prec_c - prec_baseline
    print(f"\n  Precision delta (C vs A): {delta_prec:+.4f}")

    # Table 2: Hop-stratified
    print(f"\n\n{'='*90}")
    print(f"  TABLE 2: Hop-Stratified (Model A vs Model C)")
    print(f"{'='*90}")

    hdr = (f"  {'Hop':>4}  {'N_pos':>6}  "
           f"{'τ_A':>8}  {'τ_C':>8}  {'Δτ':>8}  "
           f"{'Dir_A':>8}  {'Dir_C':>8}  {'ΔDir':>8}  "
           f"{'95% CI':>20}")
    print(f"\n{hdr}")
    print(f"  {'-'*86}")

    for bucket in ['1', '2', '3', '4+', 'ALL']:
        ra = results_a.get(bucket)
        rc = results_c.get(bucket)
        ci = ci_results.get(bucket)

        if ra is None or rc is None:
            print(f"  {bucket:>4}  (insufficient data)")
            continue

        d_tau = rc['tau'] - ra['tau']
        d_dir = (rc['dir_acc'] - ra['dir_acc']) * 100

        ci_str = ""
        if ci is not None:
            sig = "*" if ci['sig'] else ""
            ci_str = f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]{sig}"

        if bucket == 'ALL':
            print(f"  {'-'*86}")

        print(f"  {bucket:>4}  {ra['n_pos']:>6}  "
              f"{ra['tau']:>+8.4f}  {rc['tau']:>+8.4f}  {d_tau:>+8.4f}  "
              f"{ra['dir_acc']:>7.1%}  {rc['dir_acc']:>7.1%}  {d_dir:>+7.1f}%  "
              f"{ci_str:>20}")

    # Table 3: Buffer composition comparison
    print(f"\n\n{'='*90}")
    print(f"  TABLE 3: Buffer Composition Comparison")
    print(f"{'='*90}")

    buf = trainer_c.cegar
    total_considered = (buf.n_added_total + buf.n_rejected_conf
                        + buf.n_rejected_spurious)
    rej_rate = ((buf.n_rejected_conf + buf.n_rejected_spurious)
                / max(total_considered, 1))

    active_pairs = [p for p in buf.pairs
                    if buf.current_step - p[5] <= buf.expiry_steps]
    fp_active = sum(1 for p in active_pairs if p[6] == 'FP')
    fn_active = sum(1 for p in active_pairs if p[6] == 'FN')
    n_active = len(active_pairs)

    fp_rate_c = fp_active / max(n_active, 1)
    fn_rate_c = fn_active / max(n_active, 1)

    orig_fp = orig_cegar.get('buf_stats', {}).get('fp_rate', 'N/A') if orig_cegar else 'N/A'
    orig_fn = orig_cegar.get('buf_stats', {}).get('fn_rate', 'N/A') if orig_cegar else 'N/A'
    orig_buf = orig_cegar.get('buf_stats', {}).get('buf_size', 'N/A') if orig_cegar else 'N/A'

    print(f"\n  {'':>28} {'Original CEGAR':>16} {'Corrected CEGAR':>16}")
    print(f"  {'-'*60}")
    print(f"  {'Buffer size':<28} {str(orig_buf):>16} {n_active:>16}")
    if isinstance(orig_fp, float):
        print(f"  {'FP rate':<28} {orig_fp:>15.1%} {fp_rate_c:>15.1%}  (target: >40%)")
        print(f"  {'FN rate':<28} {orig_fn:>15.1%} {fn_rate_c:>15.1%}  (target: <50%)")
    else:
        print(f"  {'FP rate':<28} {str(orig_fp):>16} {fp_rate_c:>15.1%}  (target: >40%)")
        print(f"  {'FN rate':<28} {str(orig_fn):>16} {fn_rate_c:>15.1%}  (target: <50%)")
    print(f"  {'Rejection rate':<28} {'0%':>16} {rej_rate:>15.1%}")
    print(f"  {'Precision Δ vs baseline':<28} "
          f"{orig_cegar.get('delta_tau_all', 0) if orig_cegar else 'N/A':>16} "
          f"{delta_prec:>+15.4f}  (target: positive)")

    # ── Decision criteria ────────────────────────────────────────────────
    r4_a = results_a.get('4+')
    r4_c = results_c.get('4+')
    r1_a = results_a.get('1')
    r1_c = results_c.get('1')
    ci_4 = ci_results.get('4+')

    d_tau_4 = (r4_c['tau'] - r4_a['tau']) if (r4_a and r4_c) else 0.0
    d_tau_1 = (r1_c['tau'] - r1_a['tau']) if (r1_a and r1_c) else 0.0
    d_tau_all = (rc_all.get('tau', 0) - ra_all.get('tau', 0))

    print(f"\n  Decision criteria:")
    c1 = d_tau_4 > 0.02
    c2 = d_tau_4 > d_tau_1
    c3 = delta_prec > 0  # precision increased
    c4 = fp_rate_c > 0.40
    c5 = d_tau_all >= 0

    print(f"    [{'PASS' if c1 else 'FAIL'}] Δτ(4+) = {d_tau_4:+.4f} > +0.02")
    print(f"    [{'PASS' if c2 else 'FAIL'}] Δτ(4+) = {d_tau_4:+.4f} > Δτ(1) = {d_tau_1:+.4f}")
    print(f"    [{'PASS' if c3 else 'FAIL'}] Precision increased: {delta_prec:+.4f}")
    print(f"    [{'PASS' if c4 else 'FAIL'}] FP rate in buffer > 40%: {fp_rate_c:.1%}")
    print(f"    [{'PASS' if c5 else 'FAIL'}] Δτ(ALL) = {d_tau_all:+.4f} >= 0")

    validated = c1 and c3 and c5
    not_validated = (d_tau_4 <= 0) or (delta_prec < 0) or (d_tau_all < -0.01)

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    ci_str = (f"[{ci_4['ci_lo']:+.4f}, {ci_4['ci_hi']:+.4f}]"
              if ci_4 else "N/A")

    if validated:
        print(f"""
VERDICT A -- CORRECTED CEGAR VALIDATED:

Corrected CEGAR is validated as a contribution.
Evidence: Δτ(4+) = {d_tau_4:+.4f} {ci_str}
          Precision Δ = {delta_prec:+.4f} (positive = geometry improved)
          FP rate in buffer: {fp_rate_c:.1%}
Interpretation: Confidence filtering + spurious detection + asymmetric
weighting produce a CEGAR buffer that improves box geometry.
Action: Include corrected CEGAR as contribution. Document the three
fixes as part of the methodology.
""")
    else:
        print(f"""
VERDICT B -- CORRECTED CEGAR NOT VALIDATED:

Corrected CEGAR does not pass validation criteria.
Evidence: Δτ(4+) = {d_tau_4:+.4f} {ci_str}
          Precision Δ = {delta_prec:+.4f}
          Δτ(ALL) = {d_tau_all:+.4f}
Interpretation: Even with confidence filtering, spurious detection, and
asymmetric weighting, CEGAR does not improve on this benchmark.
The counterexample refinement paradigm may require fundamentally
different training settings (hard negatives, graded objectives).
Action: Remove CEGAR from claimed contributions. Move to Section 6 (Future Work).
""")
    print(f"{'='*90}")

    # ── Save ─────────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    save_data = {
        'seed': SEED,
        'experiment': 'corrected_cegar',
        'verdict': 'VALIDATED' if validated else 'NOT_VALIDATED',
        'delta_tau_4plus': d_tau_4,
        'delta_tau_1': d_tau_1,
        'delta_tau_all': d_tau_all,
        'precision_baseline': prec_baseline,
        'precision_corrected': prec_c,
        'precision_delta': delta_prec,
        'spurious_threshold': spurious_threshold,
        'ci_4plus': ci_results.get('4+'),
        'buffer_stats': {
            'active': n_active,
            'fp_active': fp_active,
            'fn_active': fn_active,
            'fp_rate': fp_rate_c,
            'fn_rate': fn_rate_c,
            'rejection_rate': rej_rate,
            'total_added': buf.n_added_total,
            'rejected_conf': buf.n_rejected_conf,
            'rejected_spurious': buf.n_rejected_spurious,
        },
        'per_bucket': {
            bucket: {
                'tau_a': results_a[bucket]['tau'] if results_a.get(bucket) else None,
                'tau_c': results_c[bucket]['tau'] if results_c.get(bucket) else None,
                'dir_a': results_a[bucket]['dir_acc'] if results_a.get(bucket) else None,
                'dir_c': results_c[bucket]['dir_acc'] if results_c.get(bucket) else None,
                'ci': ci_results.get(bucket),
            }
            for bucket in ['1', '2', '3', '4+', 'ALL']
        },
        'precision_history': [
            {'epoch': h['epoch'], 'precision': h['precision']}
            for h in trainer_c.history
        ],
    }
    with open('results/cegar_corrected.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"  Results saved to results/cegar_corrected.json")


if __name__ == '__main__':
    main()
