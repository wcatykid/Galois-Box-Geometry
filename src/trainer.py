"""
Trainer
=======
Implements end-to-end training of the Galois NLP model using a simple
Adam optimiser implemented in NumPy (no PyTorch dependency).

Training loop:
  For each batch:
    1. Encode both sentences → boxes [l1,u1], [l2,u2]
    2. Decode box midpoints  → reconstructed embeddings
    3. Compute all four losses
    4. Detect counterexamples → add to CEGAR buffer
    5. Backprop (finite differences or analytic gradients)
    6. Update parameters with Adam

Because NumPy has no autograd, we use:
  - Closed-form parameter-level update approximations (chain-rule sketches)
  - Finite-difference gradient verification (optional)

This gives a fully runnable training loop suitable for CPU execution.
"""

import numpy as np
import time
from src.encoder import encode, decode, init_encoder_params, init_decoder_params
from src.losses import (
    reconstruction_loss, galois_loss, task_loss,
    cegar_loss, total_loss, CEGARBuffer
)
from src.box_lattice import box_containment_score
from src.dataset import NLIDataLoader, ENTAILMENT


# ---------------------------------------------------------------------------
# Adam optimiser (parameter-dict based)
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam for a flat dict of NumPy parameter arrays."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for k in self.params:
            if k not in grads:
                continue
            g = grads[k] + self.wd * self.params[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Gradient computation via finite differences
# ---------------------------------------------------------------------------

def finite_diff_grad(loss_fn, params, key, eps=1e-4, n_sample=10):
    """
    Compute ∂loss/∂params[key] using central differences.
    Approximates gradient on a random sample of parameter entries.

    This is used as a stand-in for backprop in the NumPy environment.
    For the full differentiable version, see the PyTorch implementation.
    """
    param = params[key]
    grad = np.zeros_like(param)

    # Only compute gradient for a random subset of entries (efficiency)
    flat_idx = np.random.choice(param.size, size=min(n_sample, param.size), replace=False)

    for fi in flat_idx:
        idx = np.unravel_index(fi, param.shape)
        old = param[idx]
        param[idx] = old + eps
        loss_plus = loss_fn(params)
        param[idx] = old - eps
        loss_minus = loss_fn(params)
        param[idx] = old
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

    return grad


# ---------------------------------------------------------------------------
# Analytic gradient approximation
# ---------------------------------------------------------------------------

def compute_gradients_analytic(batch, enc_params, dec_params,
                                lambda1, lambda2, lambda3, lambda4,
                                cegar_buf, margin=1.0, scale=10.0):
    """
    Compute approximate gradients analytically for the encoder/decoder.

    Strategy:
      1. Forward pass to get boxes and losses
      2. Compute ∂L_total/∂l, ∂L_total/∂u (box gradient)
      3. Backprop through the linear output heads with chain rule

    This is an approximation; the trunk (Transformer) gradient is
    approximated by propagating through only the linear heads.
    """
    emb1, emb2, labels = batch['emb1'], batch['emb2'], batch['label']
    N = len(labels)

    # ── Forward pass ──────────────────────────────────────────────────────
    l1, u1, h1 = encode(emb1, enc_params)
    l2, u2, h2 = encode(emb2, enc_params)
    x1_hat = decode(l1, u1, dec_params)
    x2_hat = decode(l2, u2, dec_params)

    # ── Losses ────────────────────────────────────────────────────────────
    l_recon = reconstruction_loss(
        np.vstack([emb1, emb2]), np.vstack([x1_hat, x2_hat]))

    binary = (labels == ENTAILMENT).astype(float)
    l_gal  = galois_loss(l1, u1, l2, u2, binary, margin=margin)
    l_tsk, p_pred = task_loss(l1, u1, l2, u2, labels, scale=scale)
    l_ceg  = cegar_loss(cegar_buf)

    l_tot = total_loss(l_recon, l_gal, l_tsk, l_ceg,
                       lambda1, lambda2, lambda3, lambda4)

    # ── Gradient w.r.t. box bounds (∂L/∂l1, ∂L/∂u1) ─────────────────────
    # From Galois loss (hinge) for entailment pairs
    viol_12 = np.maximum(0.0, (l2 - l1)) + np.maximum(0.0, (u1 - u2))

    # ∂I/∂l1 = -1 where l2 > l1 else 0  (per dim)
    dI_dl1 = -(l2 > l1).astype(float)
    # ∂I/∂u1 = +1 where u1 > u2 else 0
    dI_du1 = (u1 > u2).astype(float)

    # Galois gradient for entailment (label=1): ∂max(0, m-I)/∂I = -1 if I<m
    gal_active = (binary[:, None]) * (viol_12.sum(-1, keepdims=True) < margin)
    dLgal_dl1 = -gal_active * dI_dl1 / N
    dLgal_du1 = -gal_active * dI_du1 / N

    # For non-entailment (label=0): ∂I/∂l1
    non_ent = (1 - binary[:, None])
    dLgal_dl1 += non_ent * dI_dl1 / N
    dLgal_du1 += non_ent * dI_du1 / N

    # Task BCE gradient w.r.t. viol → box bounds
    p_bin = (labels == ENTAILMENT).astype(float)
    dp_dviol = -scale * p_pred * (1 - p_pred)
    dBCE_dviol = (p_pred - p_bin) / N   # ∂BCE/∂p · ∂p/∂viol (simplified)
    dLtsk_dviol = dBCE_dviol[:, None]

    dLtsk_dl1 = dLtsk_dviol * dI_dl1
    dLtsk_du1 = dLtsk_dviol * dI_du1

    # Reconstruction: ∂MSE/∂l1  (through decoder midpoint z = (l+u)/2)
    diff1 = x1_hat - emb1
    dL_dz1 = 2.0 * (diff1 @ dec_params['W2'].T) @ dec_params['W1'].T
    dL_dz1 = dL_dz1 / (N * emb1.shape[1])
    dLrec_dl1 = 0.5 * dL_dz1
    dLrec_du1 = 0.5 * dL_dz1

    # Total gradient on l1, u1
    dl1 = lambda2 * dLgal_dl1 + lambda3 * dLtsk_dl1 + lambda1 * dLrec_dl1
    du1 = lambda2 * dLgal_du1 + lambda3 * dLtsk_du1 + lambda1 * dLrec_du1

    # ── Chain rule through linear heads ──────────────────────────────────
    # u = l + softplus(W_u h + b_u),  δ = softplus(...)
    # ∂loss/∂W_l  =  h.T @ dl1  (batch outer product)
    grads_enc = {}
    grads_enc['Wl'] = (h1.T @ dl1) / N
    grads_enc['bl'] = dl1.mean(axis=0)

    # ∂loss/∂W_u: through softplus(x), ∂softplus/∂x = sigmoid(x)
    pre_delta1 = h1 @ enc_params['Wu'] + enc_params['bu']
    sig_pre = 1.0 / (1.0 + np.exp(-np.clip(pre_delta1, -50, 50)))
    d_delta = du1 * sig_pre   # chain through softplus
    # Also l1 depends on W_u only through u = l + delta → both dl1 and du1
    grads_enc['Wu'] = (h1.T @ d_delta) / N
    grads_enc['bu'] = d_delta.mean(axis=0)

    # Approximate gradient of trunk layers (W2): propagate dl1 back
    dh1_from_l = dl1 @ enc_params['Wl'].T + d_delta @ enc_params['Wu'].T
    grads_enc['W2'] = (h1.T @ dh1_from_l) / N   # using h1 ≈ pre-activation
    grads_enc['b2'] = dh1_from_l.mean(axis=0)

    # W1 (further approximation)
    dh0 = dh1_from_l @ enc_params['W2'].T
    grads_enc['W1'] = (emb1.T @ dh0) / N
    grads_enc['b1'] = dh0.mean(axis=0)

    # Decoder gradients: ∂MSE/∂W_dec
    dL_dx1hat = 2.0 * diff1 / (N * emb1.shape[1])
    z1 = 0.5 * (l1 + u1)
    h_dec1 = np.maximum(0.0, z1 @ dec_params['W1'] + dec_params['b1'])
    grads_dec = {}
    grads_dec['W2'] = (h_dec1.T @ dL_dx1hat)
    grads_dec['b2'] = dL_dx1hat.mean(axis=0)
    # ReLU derivative
    drelu = (h_dec1 > 0).astype(float)
    dh_dec = (dL_dx1hat @ dec_params['W2'].T) * drelu
    grads_dec['W1'] = (z1.T @ dh_dec)
    grads_dec['b1'] = dh_dec.mean(axis=0)

    losses = {
        'total': l_tot, 'recon': l_recon,
        'galois': l_gal, 'task': l_tsk, 'cegar': l_ceg
    }
    return grads_enc, grads_dec, losses, (l1, u1, l2, u2, p_pred)


# ---------------------------------------------------------------------------
# Main Trainer class
# ---------------------------------------------------------------------------

class GaloisTrainer:

    def __init__(self, input_dim, hidden_dim=128, box_dim=32,
                 lr=3e-4, lambda1=1.0, lambda2=1.5, lambda3=1.0, lambda4=0.5,
                 margin=1.0, scale=10.0, cegar_max=2000, rng=None):

        self.rng = rng or np.random.default_rng(42)
        self.box_dim = box_dim
        self.input_dim = input_dim
        self.lambdas = (lambda1, lambda2, lambda3, lambda4)
        self.margin = margin
        self.scale = scale

        # Initialise parameters
        self.enc_params = init_encoder_params(input_dim, hidden_dim, box_dim, self.rng)
        self.dec_params = init_decoder_params(box_dim, hidden_dim, input_dim, self.rng)

        # Optimisers
        self.enc_opt = AdamOptimizer(self.enc_params, lr=lr, weight_decay=1e-4)
        self.dec_opt = AdamOptimizer(self.dec_params, lr=lr, weight_decay=1e-4)

        # CEGAR buffer
        self.cegar_buf = CEGARBuffer(max_size=cegar_max)

        # History
        self.history = []

    def encode(self, emb):
        return encode(emb, self.enc_params)

    def predict_entailment(self, emb1, emb2):
        """Return entailment probability for pairs."""
        l1, u1, _ = self.encode(emb1)
        l2, u2, _ = self.encode(emb2)
        return box_containment_score(l1, u1, l2, u2, scale=self.scale)

    def train_epoch(self, dataloader, epoch=0):
        losses_epoch = []
        for batch in dataloader:
            grads_enc, grads_dec, losses, (l1, u1, l2, u2, p_pred) = \
                compute_gradients_analytic(
                    batch, self.enc_params, self.dec_params,
                    *self.lambdas, self.cegar_buf,
                    margin=self.margin, scale=self.scale)

            # Detect counterexamples and add to buffer
            binary_labels = (batch['label'] == ENTAILMENT).astype(int)
            pred_labels   = (p_pred > 0.5).astype(int)
            self.cegar_buf.add(l1, u1, l2, u2, binary_labels, pred_labels)

            # Parameter update
            self.enc_opt.step(grads_enc)
            self.dec_opt.step(grads_dec)

            losses_epoch.append(losses)

        mean_losses = {k: np.mean([l[k] for l in losses_epoch])
                       for k in losses_epoch[0]}
        self.history.append({'epoch': epoch, **mean_losses})
        return mean_losses

    def evaluate(self, dataset):
        """Evaluate accuracy, F1, and box statistics on a dataset."""
        from src.metrics import compute_nli_metrics, compute_box_stats
        l1, u1, _ = self.encode(dataset['emb1'])
        l2, u2, _ = self.encode(dataset['emb2'])
        p_entail = box_containment_score(l1, u1, l2, u2, scale=self.scale)
        labels = dataset['label']
        metrics = compute_nli_metrics(p_entail, labels)
        box_stats = compute_box_stats(l1, u1, l2, u2)
        return {**metrics, **box_stats}

    def train(self, train_data, val_data, n_epochs=20, batch_size=64,
              verbose=True, patience=5):
        """Full training loop with early stopping."""
        loader = NLIDataLoader(train_data, batch_size=batch_size,
                               shuffle=True, rng=self.rng)
        best_val_acc = -1
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_losses = self.train_epoch(loader, epoch=epoch)
            val_metrics  = self.evaluate(val_data)
            elapsed = time.time() - t0

            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Loss={train_losses['total']:.4f} "
                      f"(R={train_losses['recon']:.3f} "
                      f"G={train_losses['galois']:.3f} "
                      f"T={train_losses['task']:.3f} "
                      f"C={train_losses['cegar']:.3f}) | "
                      f"Val Acc={val_metrics['accuracy']:.3f} "
                      f"F1={val_metrics['f1']:.3f} | "
                      f"CEGAR buf={len(self.cegar_buf)} | "
                      f"{elapsed:.1f}s")

            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(best={best_val_acc:.3f} at epoch {best_epoch})")
                    self._restore_best()
                    break

        return self.history

    def _save_best(self):
        import copy
        self._best_enc = {k: v.copy() for k, v in self.enc_params.items()}
        self._best_dec = {k: v.copy() for k, v in self.dec_params.items()}

    def _restore_best(self):
        if hasattr(self, '_best_enc'):
            for k in self.enc_params:
                self.enc_params[k] = self._best_enc[k]
            for k in self.dec_params:
                self.dec_params[k] = self._best_dec[k]
