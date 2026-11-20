"""
Loss Functions — FIXED
======================
Corrected from original version. Key fixes:

  BUG 1 FIXED: galois_loss() hinge directions were swapped.
    - Entailment loss = max(0, I) -- penalize violation (push I → 0)
    - Non-entailment loss = max(0, margin - I) -- penalize closeness (push I ≥ margin)
  
  BUG 2 FIXED: label convention unified to SNLI standard:
    - 0 = entailment, 1 = neutral, 2 = contradiction (throughout)
    - galois_loss() and task_loss() now both use 0=entailment
"""

import numpy as np
from src.box_lattice import box_containment_violation, box_containment_score

ENTAILMENT    = 0
NEUTRAL       = 1
CONTRADICTION = 2

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _bce(y_pred, y_true, eps=1e-7):
    p = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def reconstruction_loss(x_orig, x_recon):
    """MSE between original and reconstructed embeddings."""
    diff = x_orig - x_recon
    return float(np.mean(diff ** 2))


def galois_loss(l1, u1, l2, u2, labels, margin=1.0):
    """
    Hinge loss enforcing:
      Entailment   (label=0): α(x1) ⊆ α(x2) → penalise violation I
      Non-entailment (label≠0): push I ≥ margin → penalise max(0, margin-I)

    Args:
        labels: SNLI labels (0=entailment, 1=neutral, 2=contradiction)
    """
    viol = box_containment_violation(l1, u1, l2, u2)        # (N,)
    binary = (labels == ENTAILMENT).astype(float)            # 1 if entailment

    ent_loss  = binary       * np.maximum(0.0, viol)         # push viol → 0
    nent_loss = (1 - binary) * np.maximum(0.0, margin - viol)  # push viol ≥ margin

    return float(np.mean(ent_loss + nent_loss))


def task_loss(l1, u1, l2, u2, labels, scale=10.0):
    """
    Binary cross-entropy on entailment classification.
    p_entail = σ(−I · scale).   label=0 (entailment) → target=1.

    Args:
        labels: SNLI labels (0=entailment, 1=neutral, 2=contradiction)
    Returns:
        (scalar loss, predicted probabilities)
    """
    viol = box_containment_violation(l1, u1, l2, u2)
    p_entail = _sigmoid(-viol * scale)
    binary_target = (labels == ENTAILMENT).astype(float)     # 1 if entailment
    loss = _bce(p_entail, binary_target)
    return float(np.mean(loss)), p_entail


class CEGARBuffer:
    """
    Vectorised ring-buffer of misclassified counterexample pairs.

    Stores l1/u1/l2/u2/labels as contiguous NumPy arrays so that both
    add() and sample() are fully vectorised — no Python-level loops.

    Label convention: stored labels use the same binary encoding passed
    in by the caller (1 = entailment, 0 = non-entailment).
    """

    def __init__(self, max_size=2000):
        self.max_size = max_size
        # Arrays are lazily allocated on the first add() call once D is known.
        self._l1 = self._u1 = self._l2 = self._u2 = self._lab = None
        self._head = 0   # next write position (ring pointer)
        self._size = 0   # number of valid entries currently stored

    def _init_arrays(self, D):
        self._l1  = np.empty((self.max_size, D), dtype=np.float64)
        self._u1  = np.empty((self.max_size, D), dtype=np.float64)
        self._l2  = np.empty((self.max_size, D), dtype=np.float64)
        self._u2  = np.empty((self.max_size, D), dtype=np.float64)
        self._lab = np.empty(self.max_size,       dtype=np.int32)

    def add(self, l1, u1, l2, u2, labels, pred_labels):
        """Add misclassified pairs — fully vectorised, no Python loop."""
        mask = (labels != pred_labels)
        if not np.any(mask):
            return
        nl1  = l1[mask];  nu1  = u1[mask]
        nl2  = l2[mask];  nu2  = u2[mask]
        nlab = labels[mask].astype(np.int32)
        n, D = nl1.shape

        if self._l1 is None:
            self._init_arrays(D)

        # If the new batch is larger than the buffer, keep only the tail.
        if n >= self.max_size:
            nl1 = nl1[-self.max_size:]; nu1 = nu1[-self.max_size:]
            nl2 = nl2[-self.max_size:]; nu2 = nu2[-self.max_size:]
            nlab = nlab[-self.max_size:]
            n = self.max_size
            self._head = 0
            self._size = 0

        end = self._head + n
        if end <= self.max_size:
            self._l1[self._head:end]  = nl1
            self._u1[self._head:end]  = nu1
            self._l2[self._head:end]  = nl2
            self._u2[self._head:end]  = nu2
            self._lab[self._head:end] = nlab
        else:
            # Wrap around the ring boundary.
            split = self.max_size - self._head
            self._l1[self._head:] = nl1[:split];  self._l1[:n-split] = nl1[split:]
            self._u1[self._head:] = nu1[:split];  self._u1[:n-split] = nu1[split:]
            self._l2[self._head:] = nl2[:split];  self._l2[:n-split] = nl2[split:]
            self._u2[self._head:] = nu2[:split];  self._u2[:n-split] = nu2[split:]
            self._lab[self._head:]= nlab[:split]; self._lab[:n-split]= nlab[split:]

        self._head = end % self.max_size
        self._size = min(self._size + n, self.max_size)

    def __len__(self):
        return self._size

    def clear(self):
        """Discard all stored entries (keeps allocated arrays for reuse)."""
        self._head = 0
        self._size = 0

    def sample(self, n):
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return (self._l1[idx], self._u1[idx],
                self._l2[idx], self._u2[idx],
                self._lab[idx])


def cegar_loss(buffer, n_sample=64, margin=1.5):
    """
    Refinement loss on counterexample buffer.
    False positives (predicted entail, truth=non-entail): push violation UP.
    False negatives (predicted non-entail, truth=entail): push violation DOWN.
    """
    if len(buffer) == 0:
        return 0.0
    result = buffer.sample(n_sample)
    if result is None:
        return 0.0
    l1, u1, l2, u2, labels = result
    viol = box_containment_violation(l1, u1, l2, u2)
    # Buffer stores binary labels: 1 = entailment, 0 = non-entailment
    binary = labels.astype(float)
    fn_loss = binary       * np.maximum(0.0, viol)               # false neg: push viol down
    fp_loss = (1 - binary) * np.maximum(0.0, margin - viol)      # false pos: push viol up
    return float(np.mean(fn_loss + fp_loss))


def total_loss(l_recon, l_galois, l_task, l_cegar,
               lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=0.5):
    return lambda1*l_recon + lambda2*l_galois + lambda3*l_task + lambda4*l_cegar


def galois_loss_3class(l1, u1, l2, u2, labels,
                       # Calibrated from violation magnitude analysis on SBERT+SNLI
                       # (128-dim boxes, 10-epoch model):
                       #   viol_fwd medians: ent=0.65, neu=5.65, con=7.31
                       #   viol_bwd medians: ent=12.7,  neu=4.09, con=5.86
                       #   asymmetry (viol_bwd−viol_fwd): +12 ent, −1.5 neu/con
                       ent_fwd_margin   = 3.0,   # push ent viol_fwd → 0
                       ent_bwd_target   = 10.0,  # push ent viol_bwd high
                       neu_fwd_margin   = 3.0,   # push neu viol_fwd ≥ 3
                       neu_bwd_ceiling  = 6.0,   # push neu viol_bwd ≤ 6
                       con_fwd_margin   = 5.0,   # push con viol_fwd ≥ 5
                       con_bwd_ceiling  = 8.0,   # reserved (con viol_bwd unconstrained)
                       overlap_weight   = 0.1):  # contradiction overlap penalty
    """
    Three-class directional hinge loss over learned box representations.

    Entailment   : viol_fwd → 0  AND  viol_bwd → ent_bwd_target
                   (premise box contains hypothesis → backward violation is high)
    Neutral      : viol_fwd ≥ neu_fwd_margin  AND  viol_bwd ≤ neu_bwd_ceiling
                   (partial overlap — neither box contains the other)
    Contradiction: viol_fwd ≥ con_fwd_margin  AND  overlap → 0
                   (disjoint boxes — separation in forward direction)

    The viol_bwd signal for entailment is the key improvement over binary loss:
    the model was accidentally learning viol_bwd ≈ 12.7 for entailment as a
    side-effect.  Making it explicit sharpens the entailment/neutral boundary.

    Args:
        labels       : (N,) SNLI labels — 0=entailment, 1=neutral, 2=contradiction
        con_bwd_ceiling: reserved for future use; currently unused
    Returns:
        scalar float loss
    """
    viol_fwd = box_containment_violation(l1, u1, l2, u2)   # (N,)
    viol_bwd = box_containment_violation(l2, u2, l1, u1)   # (N,)

    meet_l  = np.maximum(l1, l2)
    meet_u  = np.minimum(u1, u2)
    overlap = np.sum(np.maximum(0.0, meet_u - meet_l), axis=-1)   # (N,)

    ent    = (labels == ENTAILMENT).astype(float)
    neu    = (labels == NEUTRAL).astype(float)
    contra = (labels == CONTRADICTION).astype(float)

    # Entailment: viol_fwd → 0,  viol_bwd → high (directional box asymmetry)
    l_ent = ent * (
        np.maximum(0.0, viol_fwd) +
        0.5 * np.maximum(0.0, ent_bwd_target - viol_bwd)
    )

    # Neutral: viol_fwd ≥ margin,  viol_bwd ≤ ceiling (symmetric partial overlap)
    l_neu = neu * (
        np.maximum(0.0, neu_fwd_margin - viol_fwd) +
        0.5 * np.maximum(0.0, viol_bwd - neu_bwd_ceiling)
    )

    # Contradiction: viol_fwd ≥ margin,  push overlap → 0 (disjoint boxes)
    l_con = contra * (
        np.maximum(0.0, con_fwd_margin - viol_fwd) +
        overlap * overlap_weight
    )

    return float(np.mean(l_ent + l_neu + l_con))
