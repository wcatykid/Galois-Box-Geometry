"""
Metrics & Baselines
====================
Implements:
  1. NLI evaluation: accuracy, precision, recall, F1, AUC-ROC
  2. STS evaluation: Pearson r, Spearman ρ
  3. Box statistics: mean volume, width distribution, lattice axiom checks
  4. Baseline models: cosine similarity, random, majority class, SVM on raw embeddings
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# NLI metrics
# ---------------------------------------------------------------------------

def compute_nli_metrics(p_entail, labels, threshold=0.5):
    """
    Compute binary NLI metrics (entailment vs. non-entailment).

    Args:
        p_entail: (N,) predicted probability of entailment
        labels:   (N,) ground-truth SNLI labels (0=entail, 1=neutral, 2=contra)
        threshold: decision threshold

    Returns:
        dict with accuracy, precision, recall, f1, auc, threshold
    """
    binary_labels = (labels == 0).astype(int)   # 0=entail in SNLI
    pred_labels   = (p_entail >= threshold).astype(int)

    tp = np.sum((pred_labels == 1) & (binary_labels == 1))
    fp = np.sum((pred_labels == 1) & (binary_labels == 0))
    fn = np.sum((pred_labels == 0) & (binary_labels == 1))
    tn = np.sum((pred_labels == 0) & (binary_labels == 0))

    accuracy  = (tp + tn) / len(labels)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # AUC via trapezoidal rule
    auc = _compute_auc(p_entail, binary_labels)

    # Optimal threshold search — scan percentiles of the actual score
    # distribution so the search is scale-invariant (works when scores
    # are clustered near 0 or near 1, not just in [0.1, 0.9]).
    candidates = np.unique(np.percentile(p_entail, np.linspace(1, 99, 49)))
    best_f1, best_thr = 0.0, threshold
    for thr in candidates:
        pred_t = (p_entail >= thr).astype(int)
        tp_t = np.sum((pred_t == 1) & (binary_labels == 1))
        fp_t = np.sum((pred_t == 1) & (binary_labels == 0))
        fn_t = np.sum((pred_t == 0) & (binary_labels == 1))
        pr_t = tp_t / (tp_t + fp_t + 1e-8)
        rc_t = tp_t / (tp_t + fn_t + 1e-8)
        f1_t = 2 * pr_t * rc_t / (pr_t + rc_t + 1e-8)
        if f1_t > best_f1:
            best_f1, best_thr = f1_t, thr

    return {
        'accuracy':  float(accuracy),
        'precision': float(precision),
        'recall':    float(recall),
        'f1':        float(f1),
        'auc':       float(auc),
        'best_f1':   float(best_f1),
        'best_thr':  float(best_thr),
    }


def _compute_auc(scores, labels):
    """Compute AUC-ROC using the trapezoidal rule."""
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = labels.sum()
    neg = len(labels) - pos
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tpr = np.sum((pred == 1) & (labels == 1)) / (pos + 1e-8)
        fpr = np.sum((pred == 1) & (labels == 0)) / (neg + 1e-8)
        tprs.append(tpr); fprs.append(fpr)
    tprs.append(1.0); fprs.append(1.0)
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    return float(trapz(tprs, fprs))


def compute_3class_accuracy(pred_classes, true_labels):
    """3-way classification accuracy."""
    return float(np.mean(pred_classes == true_labels))


# ---------------------------------------------------------------------------
# 3-class box features & probe
# ---------------------------------------------------------------------------

def extract_box_features(l1, u1, l2, u2):
    """
    Extract 4 scalar features per (premise, hypothesis) box pair.
    Legacy 4-feature version kept for backward compatibility.
    Prefer compute_3class_features() for new code (9 features, better discrimination).

    Features:
      0  viol_fwd   : violation of "premise contains hypothesis"
      1  viol_bwd   : violation of "hypothesis contains premise"
      2  overlap    : mean dimension-wise intersection width
      3  width_asym : mean_width(premise) − mean_width(hypothesis)

    Returns: (N, 4) float64
    """
    from src.box_lattice import box_containment_violation

    viol_fwd = box_containment_violation(l1, u1, l2, u2)
    viol_bwd = box_containment_violation(l2, u2, l1, u1)

    inter_lo = np.maximum(l1, l2)
    inter_hi = np.minimum(u1, u2)
    overlap  = np.mean(np.maximum(0.0, inter_hi - inter_lo), axis=-1)

    w1 = np.mean(u1 - l1, axis=-1)
    w2 = np.mean(u2 - l2, axis=-1)
    width_asym = w1 - w2

    return np.stack([viol_fwd, viol_bwd, overlap, width_asym], axis=-1)


def compute_3class_features(l1, u1, l2, u2):
    """
    Extract 9 discriminative features per (premise, hypothesis) box pair for
    3-class NLI classification.

    Calibrated from violation magnitude analysis on SBERT+SNLI (128-dim boxes):
      viol_fwd medians : ent=0.65,  neu=5.65,  con=7.31
      viol_bwd medians : ent=12.7,  neu=4.09,  con=5.86
      asymmetry        : ent=+12.0, neu=−1.6,  con=−1.5  ← strongest discriminator
      width_ratio      : ent >> 1 (premise much wider than hypothesis)  ← 2nd strongest

    Features:
      0  viol_fwd   : violation of 'premise contains hypothesis'
                      (low → entailment; high → neutral/contradiction)
      1  viol_bwd   : violation of 'hypothesis contains premise'
                      (high → entailment; low → neutral/contradiction)
      2  asymmetry  : viol_bwd − viol_fwd
                      STRONGEST discriminator: +12 for ent, −1.5 for neu/con
      3  viol_sum   : viol_fwd + viol_bwd  (total disjointness proxy)
      4  overlap    : sum of per-dimension intersection widths
                      (high → overlapping/contained; near-zero → contradiction)
      5  w1         : mean box width of premise
      6  w2         : mean box width of hypothesis
      7  width_ratio: w1 / (w2 + ε)
                      entailment: large premise box containing small hypothesis
      8  width_diff : |w1 − w2|

    Args:
        l1, u1 : lower/upper of premise boxes   (N, d)
        l2, u2 : lower/upper of hypothesis boxes (N, d)

    Returns:
        features : (N, 9)  float64
    """
    from src.box_lattice import box_containment_violation

    viol_fwd  = box_containment_violation(l1, u1, l2, u2)   # p contains h?
    viol_bwd  = box_containment_violation(l2, u2, l1, u1)   # h contains p?
    asymmetry = viol_bwd - viol_fwd       # key: +12 for ent, −1.5 for neu/con
    viol_sum  = viol_fwd + viol_bwd

    inter_lo = np.maximum(l1, l2)
    inter_hi = np.minimum(u1, u2)
    overlap  = np.sum(np.maximum(0.0, inter_hi - inter_lo), axis=-1)

    w1          = np.mean(u1 - l1, axis=-1)
    w2          = np.mean(u2 - l2, axis=-1)
    width_ratio = w1 / (w2 + 1e-8)        # entailment: premise box >> hypothesis
    width_diff  = np.abs(w1 - w2)

    return np.stack([
        viol_fwd, viol_bwd,
        asymmetry,
        viol_sum,
        overlap,
        w1, w2,
        width_ratio,
        width_diff,
    ], axis=-1)   # (N, 9)


class ThreeClassProbe:
    """
    Softmax linear probe over box features for 3-class SNLI classification.

    Input features: compute_3class_features(l1, u1, l2, u2) — shape (N, 9)
                    or extract_box_features(l1, u1, l2, u2)  — shape (N, 4)
                    (probe is fully dimension-agnostic via W∈R^{D×3})
    Output classes: 0=entailment, 1=neutral, 2=contradiction  (SNLI standard)

    Trained with SGD + cross-entropy on normalised features.  Normalisation
    stats are fit on the training set and reused at evaluation time.

    With 9-feature input the key discriminators are:
      asymmetry = viol_bwd − viol_fwd  (+12 ent vs −1.5 neu/con)
      width_ratio = w1/(w2+ε)          (ent premise box >> hypothesis box)
    A cosine-only baseline cannot replicate these because cosine is symmetric.
    """

    name = "3-Class Linear Probe (box features)"

    def __init__(self, n_epochs=400, lr=0.05):
        self.n_epochs = n_epochs
        self.lr       = lr
        self.W    = None
        self.b    = None
        self.mu   = None
        self.sigma = None

    # ── normalisation ────────────────────────────────────────────────────────

    def _normalise(self, X):
        return (X - self.mu) / self.sigma

    # ── softmax cross-entropy ────────────────────────────────────────────────

    @staticmethod
    def _softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l   = np.exp(shifted)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, features, labels):
        """
        Args:
            features : (N, 4)  output of extract_box_features
            labels   : (N,)    SNLI labels  0/1/2
        Returns:
            self
        """
        self.mu    = features.mean(0)
        self.sigma = features.std(0) + 1e-8
        X = self._normalise(features)
        N, D = X.shape
        K = 3

        rng    = np.random.default_rng(42)
        self.W = rng.standard_normal((D, K)) * 0.01
        self.b = np.zeros(K)

        for epoch in range(self.n_epochs):
            probs   = self._softmax(X @ self.W + self.b)     # (N, K)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), labels] = 1.0
            d  = (probs - one_hot) / N                       # (N, K)

            # Cosine-decay learning-rate schedule
            lr_t    = self.lr * 0.5 * (1.0 + np.cos(np.pi * epoch / self.n_epochs))
            self.W -= lr_t * (X.T @ d)
            self.b -= lr_t * d.sum(0)

        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, features):
        """Return class indices (N,)."""
        X = self._normalise(features)
        return np.argmax(X @ self.W + self.b, axis=1)

    def predict_proba(self, features):
        """Return class probabilities (N, 3)."""
        X = self._normalise(features)
        return self._softmax(X @ self.W + self.b)

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self, features, labels):
        """
        Returns dict with:
          acc_3class  : overall 3-class accuracy
          per_class   : {'entail': float, 'neutral': float, 'contra': float}
          confusion   : (3, 3) confusion matrix (rows = true, cols = pred)
        """
        preds = self.predict(features)
        acc_3 = float(np.mean(preds == labels))

        K = 3
        confusion = np.zeros((K, K), dtype=int)
        for t in range(K):
            for p in range(K):
                confusion[t, p] = int(np.sum((labels == t) & (preds == p)))

        per_class = {}
        for c, nm in enumerate(['entail', 'neutral', 'contra']):
            mask = labels == c
            if mask.sum() > 0:
                per_class[nm] = float(np.mean(preds[mask] == c))
            else:
                per_class[nm] = float('nan')

        return {'acc_3class': acc_3, 'per_class': per_class, 'confusion': confusion}


# ---------------------------------------------------------------------------
# STS metrics
# ---------------------------------------------------------------------------

def compute_sts_metrics(pred_scores, true_scores):
    """
    Evaluate semantic textual similarity predictions.

    Args:
        pred_scores: (N,) predicted similarity in [0,5]
        true_scores: (N,) ground-truth similarity in [0,5]

    Returns:
        dict with pearson_r, spearman_rho, mse
    """
    r, _  = pearsonr(pred_scores, true_scores)
    rho, _ = spearmanr(pred_scores, true_scores)
    mse   = float(np.mean((pred_scores - true_scores) ** 2))
    return {'pearson_r': float(r), 'spearman_rho': float(rho), 'mse': mse}


# ---------------------------------------------------------------------------
# Box statistics
# ---------------------------------------------------------------------------

def compute_box_stats(l1, u1, l2, u2):
    """
    Compute statistics about the learned box representations.

    Returns:
        dict of mean/std of box volumes, widths, containment rates, etc.
    """
    from src.box_lattice import box_volume, box_containment_violation

    vol1 = box_volume(l1, u1)
    vol2 = box_volume(l2, u2)
    width1 = np.mean(u1 - l1, axis=-1)
    width2 = np.mean(u2 - l2, axis=-1)
    viol   = box_containment_violation(l1, u1, l2, u2)

    return {
        'box_vol1_mean': float(np.mean(vol1)),
        'box_vol1_std':  float(np.std(vol1)),
        'box_vol2_mean': float(np.mean(vol2)),
        'box_width_mean': float(np.mean([width1.mean(), width2.mean()])),
        'containment_viol_mean': float(np.mean(viol)),
        'zero_viol_rate': float(np.mean(viol < 1e-3)),   # fraction of "contained" pairs
    }


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class CosineSimilarityBaseline:
    """
    Cosine similarity between raw sentence embeddings as entailment proxy.
    This approximates SBERT / SimCSE behaviour on pre-computed embeddings.
    """
    name = "Cosine Similarity (SBERT-style)"

    def fit(self, train_data):
        # Find optimal threshold on training set
        scores = self._score(train_data['emb1'], train_data['emb2'])
        binary = (train_data['label'] == 0).astype(int)
        best_acc, self.threshold = 0.0, 0.5
        for thr in np.linspace(0.0, 1.0, 21):
            acc = np.mean((scores >= thr).astype(int) == binary)
            if acc > best_acc:
                best_acc = acc
                self.threshold = thr
        return self

    def _score(self, emb1, emb2):
        n1 = emb1 / (np.linalg.norm(emb1, axis=-1, keepdims=True) + 1e-8)
        n2 = emb2 / (np.linalg.norm(emb2, axis=-1, keepdims=True) + 1e-8)
        return np.sum(n1 * n2, axis=-1)

    def evaluate(self, dataset):
        scores = self._score(dataset['emb1'], dataset['emb2'])
        # Map to [0,1]
        scores = (scores + 1.0) / 2.0
        return compute_nli_metrics(scores, dataset['label'], self.threshold)

    def predict_sts(self, dataset):
        scores = self._score(dataset['emb1'], dataset['emb2'])
        pred = (scores + 1.0) * 2.5  # map [-1,1] → [0,5]
        return compute_sts_metrics(pred, dataset['score'])


class DotProductBaseline:
    """Dot product similarity (unnormalised)."""
    name = "Dot Product"

    def fit(self, train_data):
        scores = np.sum(train_data['emb1'] * train_data['emb2'], axis=-1)
        binary = (train_data['label'] == 0).astype(int)
        best_acc, self.threshold = 0.0, 0.0
        for thr in np.linspace(scores.min(), scores.max(), 21):
            acc = np.mean((scores >= thr).astype(int) == binary)
            if acc > best_acc:
                best_acc = acc
                self.threshold = float(thr)
        return self

    def evaluate(self, dataset):
        scores = np.sum(dataset['emb1'] * dataset['emb2'], axis=-1)
        s_min, s_max = scores.min(), scores.max()
        norm_scores = (scores - s_min) / (s_max - s_min + 1e-8)
        return compute_nli_metrics(norm_scores, dataset['label'], self.threshold)


class MajorityClassBaseline:
    """Always predicts the majority class."""
    name = "Majority Class"

    def fit(self, train_data):
        labels = train_data['label']
        counts = np.bincount(labels)
        self.majority = int(np.argmax(counts))
        return self

    def evaluate(self, dataset):
        N = len(dataset['label'])
        labels = dataset['label']
        binary = (labels == 0).astype(int)
        majority_binary = int(self.majority == 0)
        pred = np.full(N, majority_binary)
        p = pred.astype(float)
        return compute_nli_metrics(p, labels)


class LinearSVMBaseline:
    """
    Linear SVM on concatenated / difference embeddings.
    Implemented without sklearn (pure NumPy Pegasos SVM).
    """
    name = "Linear SVM"

    def __init__(self, C=1.0, n_epochs=100, lr=0.01):
        self.C = C
        self.n_epochs = n_epochs
        self.lr = lr
        self.w = None
        self.b = 0.0

    def _features(self, emb1, emb2):
        diff = emb1 - emb2
        prod = emb1 * emb2
        return np.concatenate([diff, prod], axis=-1)

    def fit(self, train_data):
        X = self._features(train_data['emb1'], train_data['emb2'])
        y = 2 * (train_data['label'] == 0).astype(float) - 1   # {-1, +1}
        N, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        rng = np.random.default_rng(7)

        for epoch in range(self.n_epochs):
            lr_t = self.lr / (1 + epoch)
            idx = rng.permutation(N)
            for i in idx:
                margin = y[i] * (X[i] @ self.w + self.b)
                if margin < 1.0:
                    self.w -= lr_t * (-self.C * y[i] * X[i] + self.w)
                    self.b -= lr_t * (-self.C * y[i])
                else:
                    self.w -= lr_t * self.w

        # Optimal threshold
        scores = X @ self.w + self.b
        binary = (train_data['label'] == 0).astype(int)
        best_acc, self.threshold = 0.0, 0.0
        for thr in np.linspace(scores.min(), scores.max(), 21):
            acc = np.mean((scores >= thr).astype(int) == binary)
            if acc > best_acc:
                best_acc = acc
                self.threshold = float(thr)
        return self

    def evaluate(self, dataset):
        X = self._features(dataset['emb1'], dataset['emb2'])
        scores = X @ self.w + self.b
        s_min, s_max = scores.min(), scores.max()
        norm = (scores - s_min) / (s_max - s_min + 1e-8)
        return compute_nli_metrics(norm, dataset['label'], 0.5)


class OrderEmbeddingBaseline:
    """
    Order embedding (Vendrov et al. 2016) approximation.
    Embeds sentences in R^d with partial order defined by coordinate-wise dominance.
    Energy(x1, x2) = ||max(0, f(x1) - f(x2))||^2
    Entailment when energy is low.
    """
    name = "Order Embeddings"

    def __init__(self, box_dim=32, n_epochs=50, lr=1e-3):
        self.box_dim = box_dim
        self.n_epochs = n_epochs
        self.lr = lr

    def _init_params(self, input_dim, rng):
        limit = np.sqrt(6.0 / (input_dim + self.box_dim))
        self.W = rng.uniform(-limit, limit, (input_dim, self.box_dim))
        self.b = np.zeros(self.box_dim)

    def _embed(self, x):
        return np.maximum(0.0, x @ self.W + self.b)

    def _energy(self, e1, e2):
        return np.sum(np.maximum(0.0, e1 - e2) ** 2, axis=-1)

    def fit(self, train_data):
        rng = np.random.default_rng(5)
        self._init_params(train_data['emb1'].shape[1], rng)

        emb1, emb2 = train_data['emb1'], train_data['emb2']
        binary = (train_data['label'] == 0).astype(float)
        N = len(binary)

        for epoch in range(self.n_epochs):
            e1 = self._embed(emb1)
            e2 = self._embed(emb2)
            energy = self._energy(e1, e2)

            # Normalise energy by box_dim so margins are O(1) regardless of
            # dimensionality.  Without this, at box_dim=64 raw energy is O(64)
            # and the neg_loss term (1.0 - energy) is always negative, giving
            # zero gradient for non-entailment pairs.
            energy_n = energy / self.box_dim

            # Hinge loss on normalised energy
            pos_loss = binary       * np.maximum(0.0, energy_n - 0.5)
            neg_loss = (1 - binary) * np.maximum(0.0, 1.0 - energy_n)
            loss = (pos_loss + neg_loss).mean()

            # Gradients w.r.t. raw energy, then chain through normalisation
            d_energy_e1 = 2 * binary[:, None] * np.maximum(0.0, e1 - e2)
            d_energy_e1 -= 2 * (1 - binary[:, None]) * np.maximum(0.0, e1 - e2)
            d_energy_e1 /= self.box_dim      # chain rule through energy_n
            d_energy_e2 = -d_energy_e1

            # Through relu: de/dx
            relu_mask1 = (emb1 @ self.W + self.b > 0).astype(float)
            relu_mask2 = (emb2 @ self.W + self.b > 0).astype(float)
            dW = (emb1.T @ (d_energy_e1 * relu_mask1) +
                  emb2.T @ (d_energy_e2 * relu_mask2)) / N
            db = ((d_energy_e1 * relu_mask1).mean(0) +
                  (d_energy_e2 * relu_mask2).mean(0))

            self.W -= self.lr * dW
            self.b -= self.lr * db

        # Calibrate threshold on normalised scores
        e1 = self._embed(emb1)
        e2 = self._embed(emb2)
        energy_n = self._energy(e1, e2) / self.box_dim
        scores = 1.0 / (1.0 + energy_n)
        binary_labels = (train_data['label'] == 0).astype(int)
        # Percentile-based search: scale-invariant, never collapses to
        # all-one-class regardless of the score magnitude.
        candidates = np.unique(np.percentile(scores, np.linspace(1, 99, 49)))
        best_acc, self.threshold = 0.0, float(np.median(scores))
        for thr in candidates:
            acc = np.mean((scores >= thr).astype(int) == binary_labels)
            if acc > best_acc:
                best_acc = acc
                self.threshold = float(thr)
        return self

    def evaluate(self, dataset):
        e1 = self._embed(dataset['emb1'])
        e2 = self._embed(dataset['emb2'])
        energy_n = self._energy(e1, e2) / self.box_dim
        scores = 1.0 / (1.0 + energy_n)
        return compute_nli_metrics(scores, dataset['label'], self.threshold)
