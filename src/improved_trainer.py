"""
Improved Trainer — Full analytic backprop with correct LayerNorm backward
=========================================================================
Fixes from unit testing:
  - Correct LayerNorm backward (was approximating with 1/std only)
  - CEGAR gradient flows through backward pass (was computed but ignored)
  - Label convention unified: 0=entailment throughout (SNLI standard)
  - history logs weighted CEGAR contribution (lambda4 * l_ceg)
"""

import numpy as np
import time
from src.box_lattice import box_containment_score, box_containment_violation
from src.losses import CEGARBuffer, ENTAILMENT
from src.dataset import NLIDataLoader
from src.metrics import (compute_nli_metrics, compute_box_stats,
                         extract_box_features, compute_3class_features, ThreeClassProbe)


# ── Activations ──────────────────────────────────────────────────────────────

def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))

def softplus_grad(x):
    """d/dx softplus(x) = sigmoid(x)"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def layer_norm(x, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + eps
    return (x - mu) / std, mu, std

def layer_norm_backward(dout, x, mu, std, eps=1e-5):
    """
    Correct LayerNorm backward.
    y = (x - mu) / std
    dx = (1/std) * (dout - mean(dout) - y * mean(dout * y))
    """
    N_feat = x.shape[-1]
    y = (x - mu) / std
    mean_dout    = dout.mean(axis=-1, keepdims=True)
    mean_dout_y  = (dout * y).mean(axis=-1, keepdims=True)
    dx = (dout - mean_dout - y * mean_dout_y) / std
    return dx


# ── Adam ─────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-5):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2
        self.eps = eps; self.wd = wd; self.t = 0
        self.m = {}; self.v = {}

    def update(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            g = g + self.wd * params[k]
            g = np.clip(g, -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ── Model ─────────────────────────────────────────────────────────────────────

class GaloisModel:
    """
    αθ: x → (c, δ) → l = c−δ, u = c+δ  where δ = softplus(Ww·h + bw) > 0
    γϕ: (l,u) → midpoint z → x̂

    Network: x → LN(ReLU(W1 x+b1)) → LN(ReLU(W2 h+b2)) → center & width heads
    """

    def __init__(self, input_dim, hidden_dim, box_dim, rng):
        self.d = box_dim
        H, D = hidden_dim, input_dim

        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

        self.W1 = he(D, H);   self.b1 = np.zeros(H)
        self.W2 = he(H, H);   self.b2 = np.zeros(H)
        self.Wc = he(H, box_dim) * 0.1;  self.bc = np.zeros(box_dim)
        self.Ww = he(H, box_dim) * 0.1;  self.bw = np.full(box_dim, -1.0)

        self.Wd1 = he(box_dim, H); self.bd1 = np.zeros(H)
        self.Wd2 = he(H, D);       self.bd2 = np.zeros(D)

    # ── Encoder forward ───────────────────────────────────────────────────────

    def _encode_fwd(self, x):
        """Returns l, u, h2, cache for backprop."""
        # Layer 1
        pre1     = x  @ self.W1 + self.b1        # (N, H)
        act1     = relu(pre1)
        y1, mu1, std1 = layer_norm(act1)          # (N, H)

        # Layer 2
        pre2     = y1 @ self.W2 + self.b2        # (N, H)
        act2     = relu(pre2)
        y2, mu2, std2 = layer_norm(act2)          # (N, H)  <- h2

        # Box heads
        c_pre    = y2 @ self.Wc + self.bc        # center
        w_pre    = y2 @ self.Ww + self.bw        # pre-softplus width
        hw       = softplus(w_pre)               # half-width > 0
        l        = c_pre - hw
        u        = c_pre + hw

        cache = (x, pre1, act1, y1, mu1, std1,
                    pre2, act2, y2, mu2, std2,
                    c_pre, w_pre, hw)
        return l, u, y2, cache

    def encode(self, x):
        l, u, h, _ = self._encode_fwd(x)
        return l, u, h

    # ── Decoder forward ───────────────────────────────────────────────────────

    def _decode_fwd(self, l, u):
        z      = 0.5 * (l + u)
        h_pre  = z  @ self.Wd1 + self.bd1
        h      = relu(h_pre)
        x_hat  = h  @ self.Wd2 + self.bd2
        return x_hat, (z, h_pre, h)

    def decode(self, l, u):
        x_hat, _ = self._decode_fwd(l, u)
        return x_hat

    # ── Encoder backward ──────────────────────────────────────────────────────

    def _backprop_encode(self, dl, du, cache):
        """
        Given ∂L/∂l and ∂L/∂u, compute gradients for all encoder params.
        Uses correct LayerNorm backward.
        """
        (x, pre1, act1, y1, mu1, std1,
            pre2, act2, y2, mu2, std2,
            c_pre, w_pre, hw) = cache
        N = x.shape[0]

        # l = c - hw,  u = c + hw
        # => ∂L/∂c   = dl + du
        # => ∂L/∂hw  = -dl + du
        dc  = dl + du
        dhw = -dl + du

        # Through softplus: ∂hw/∂w_pre = sigmoid(w_pre)
        dw_pre = dhw * softplus_grad(w_pre)      # (N, box_dim)

        # Grads w.r.t. Wc, bc, Ww, bw
        dWc = (y2.T @ dc)
        dbc = dc.sum(0)
        dWw = (y2.T @ dw_pre)
        dbw = dw_pre.sum(0)

        # Back through y2 (LN output of layer 2)
        dy2 = dc @ self.Wc.T + dw_pre @ self.Ww.T   # (N, H)

        # Through LN of layer 2: correct backward
        dact2 = layer_norm_backward(dy2, act2, mu2, std2)

        # Through ReLU of layer 2
        dpre2 = dact2 * relu_grad(pre2)

        # Grads w.r.t. W2, b2
        dW2 = (y1.T @ dpre2)
        db2 = dpre2.sum(0)

        # Back through y1 (LN output of layer 1)
        dy1 = dpre2 @ self.W2.T

        # Through LN of layer 1: correct backward
        dact1 = layer_norm_backward(dy1, act1, mu1, std1)

        # Through ReLU of layer 1
        dpre1 = dact1 * relu_grad(pre1)

        # Grads w.r.t. W1, b1
        dW1 = (x.T @ dpre1)
        db1 = dpre1.sum(0)

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'Wc': dWc, 'bc': dbc,
            'Ww': dWw, 'bw': dbw,
        }

    # ── Decoder backward ──────────────────────────────────────────────────────

    def _backprop_decode(self, x_orig, l, u, dcache):
        """Decoder gradient. Returns param grads + (dl_from_dec, du_from_dec)."""
        z, h_pre, h = dcache
        N, D = x_orig.shape

        x_hat     = h @ self.Wd2 + self.bd2
        dL_dxhat  = 2.0 * (x_hat - x_orig) / (N * D)    # MSE grad

        # Through Wd2
        dWd2 = (h.T @ dL_dxhat)
        dbd2 = dL_dxhat.sum(0)

        # Through ReLU
        dh = (dL_dxhat @ self.Wd2.T) * relu_grad(h_pre)

        # Through Wd1
        dWd1 = (z.T @ dh)
        dbd1 = dh.sum(0)

        # Through z = 0.5*(l+u)
        dz          = dh @ self.Wd1.T
        dl_from_dec = 0.5 * dz
        du_from_dec = 0.5 * dz

        grads = {'Wd1': dWd1, 'bd1': dbd1, 'Wd2': dWd2, 'bd2': dbd2}
        return grads, dl_from_dec, du_from_dec

    # ── Parameter dict ────────────────────────────────────────────────────────

    def params_dict(self):
        keys = ['W1','b1','W2','b2','Wc','bc','Ww','bw','Wd1','bd1','Wd2','bd2']
        return {k: getattr(self, k) for k in keys}

    def set_params(self, d):
        for k, v in d.items():
            setattr(self, k, v)


# ── Trainer ───────────────────────────────────────────────────────────────────

class ImprovedTrainer:

    def __init__(self, input_dim, hidden_dim=256, box_dim=64, lr=1e-3,
                 lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3, lambda5=0.0,
                 margin=1.0, scale=5.0, cegar_margin=1.5,
                 warmup_epochs=0, cegar_max_size=2000, flush_every=0,
                 # 3-class loss hyperparameters — calibrated from violation analysis
                 ent_bwd_target=10.0, neu_fwd_margin=3.0, neu_bwd_ceiling=6.0,
                 con_fwd_margin=5.0, overlap_weight=0.1,
                 rng=None):
        self.rng    = rng or np.random.default_rng(42)
        self.model  = GaloisModel(input_dim, hidden_dim, box_dim, self.rng)
        self.opt    = Adam(lr=lr, wd=1e-5)
        self.cegar  = CEGARBuffer(max_size=cegar_max_size)
        self.lam    = (lambda1, lambda2, lambda3, lambda4, lambda5)
        self.margin = margin
        self.scale  = scale
        self.cegar_margin  = cegar_margin
        self.warmup_epochs = warmup_epochs
        self.flush_every   = flush_every   # 0 = disabled; N = flush every N epochs
        self._epoch        = 0   # updated by train(); read by _step()
        # 3-class directional loss margins
        self.ent_bwd_target  = ent_bwd_target
        self.neu_fwd_margin  = neu_fwd_margin
        self.neu_bwd_ceiling = neu_bwd_ceiling
        self.con_fwd_margin  = con_fwd_margin
        self.overlap_weight  = overlap_weight
        self.history = []

    # ── Single training step ──────────────────────────────────────────────────

    def _step(self, emb1, emb2, labels):
        λ1, λ2, λ3, λ4, λ5 = self.lam
        # Suppress CEGAR during warmup; model must have a base before
        # counterexamples are meaningful.
        if self._epoch <= self.warmup_epochs:
            λ4 = 0.0
        N      = len(labels)
        binary = (labels == ENTAILMENT).astype(float)   # 1 if entailment

        # ── Forward ──────────────────────────────────────────────────────────
        l1, u1, h1, cache1 = self.model._encode_fwd(emb1)
        l2, u2, h2, cache2 = self.model._encode_fwd(emb2)
        x1hat, dcache1     = self.model._decode_fwd(l1, u1)
        x2hat, dcache2     = self.model._decode_fwd(l2, u2)

        viol12 = box_containment_violation(l1, u1, l2, u2)     # (N,)
        p_ent  = 1.0 / (1.0 + self.scale * viol12)             # soft score

        # ── Losses ───────────────────────────────────────────────────────────
        # 1. Reconstruction (MSE)
        diff1  = x1hat - emb1;  diff2 = x2hat - emb2
        l_rec  = 0.5 * (np.mean(diff1**2) + np.mean(diff2**2))

        # 2. Galois hinge (CORRECTED: ent→push viol→0, nent→push viol≥margin)
        ent_loss  = binary       * np.maximum(0.0, viol12)
        nent_loss = (1 - binary) * np.maximum(0.0, self.margin - viol12)
        l_gal     = float(np.mean(ent_loss + nent_loss))

        # 3. Task BCE
        p_task = np.clip(p_ent, 1e-7, 1-1e-7)
        l_tsk  = -float(np.mean(
            binary * np.log(p_task) + (1-binary) * np.log(1-p_task)))

        # 4. CEGAR (contributes to both loss AND gradients)
        pred_bin = (p_ent > 0.5).astype(int)
        self.cegar.add(l1, u1, l2, u2, binary.astype(int), pred_bin)

        l_ceg          = 0.0
        dceg_dviol12   = np.zeros(N)   # CEGAR gradient signal on current batch

        if len(self.cegar) > 32 and λ4 > 0:
            res = self.cegar.sample(64)
            if res is not None:
                cl1, cu1, cl2, cu2, clabs = res
                cviol  = box_containment_violation(cl1, cu1, cl2, cu2)
                # Buffer stores binary labels: 1 = entailment, 0 = non-entailment
                cbin   = clabs.astype(float)
                cfn    = cbin     * np.maximum(0.0, cviol)
                cfp    = (1-cbin) * np.maximum(0.0, self.cegar_margin - cviol)
                l_ceg  = float(np.mean(cfn + cfp))

                # Proxy gradient on current batch (shapes the encoder toward fixing failures)
                # False negatives on current batch: entail pairs with high viol → push down
                # False positives on current batch: non-ent pairs with low viol → push up
                dceg_dviol12 = (
                    binary       * (viol12 > 0).astype(float)            # fn: push viol down
                    - (1-binary) * (viol12 < self.cegar_margin).astype(float)  # fp: push viol up
                ) / N

        # 5. Three-class directional hinge loss (λ5 > 0 enables)
        l_3cls = 0.0
        dl1_3  = np.zeros_like(l1); du1_3 = np.zeros_like(u1)
        dl2_3  = np.zeros_like(l2); du2_3 = np.zeros_like(u2)

        if λ5 > 0:
            viol_bwd = box_containment_violation(l2, u2, l1, u1)  # (N,)
            meet_l   = np.maximum(l1, l2)
            meet_u   = np.minimum(u1, u2)
            olap_dim = np.maximum(0.0, meet_u - meet_l)           # (N, box_dim)
            overlap  = olap_dim.sum(axis=-1)                       # (N,)

            ent    = binary                        # (labels == 0) already computed
            neu    = (labels == 1).astype(float)
            contra = (labels == 2).astype(float)

            l_ent  = ent * (
                np.maximum(0.0, viol12) +
                0.5 * np.maximum(0.0, self.ent_bwd_target - viol_bwd)
            )
            l_neu  = neu * (
                np.maximum(0.0, self.neu_fwd_margin - viol12) +
                0.5 * np.maximum(0.0, viol_bwd - self.neu_bwd_ceiling)
            )
            l_con  = contra * (
                np.maximum(0.0, self.con_fwd_margin - viol12) +
                overlap * self.overlap_weight
            )
            l_3cls = float(np.mean(l_ent + l_neu + l_con))

            # ∂L/∂viol_fwd — from all three class terms
            dL_d3fwd = (
                ent    * (viol12 > 0).astype(float)
              - neu    * (viol12 < self.neu_fwd_margin).astype(float)
              - contra * (viol12 < self.con_fwd_margin).astype(float)
            ) / N

            # ∂L/∂viol_bwd — from entailment (push high) and neutral (push low)
            dL_d3bwd = (
              - ent * 0.5 * (viol_bwd < self.ent_bwd_target).astype(float)
              + neu * 0.5 * (viol_bwd > self.neu_bwd_ceiling).astype(float)
            ) / N

            # Map viol_fwd grad → box bounds
            # viol_fwd = viol(l1,u1,l2,u2): ∂/∂l1=−1{l2>l1}, ∂/∂u1=+1{u1>u2}
            ind_lf = (l2 > l1).astype(float)
            ind_uf = (u1 > u2).astype(float)
            dl1_3 += -dL_d3fwd[:, None] * ind_lf
            du1_3 +=  dL_d3fwd[:, None] * ind_uf
            dl2_3 +=  dL_d3fwd[:, None] * ind_lf
            du2_3 += -dL_d3fwd[:, None] * ind_uf

            # Map viol_bwd grad → box bounds
            # viol_bwd = viol(l2,u2,l1,u1): ∂/∂l1=+1{l1>l2}, ∂/∂u1=−1{u2>u1}
            ind_lb = (l1 > l2).astype(float)
            ind_ub = (u2 > u1).astype(float)
            dl1_3 +=  dL_d3bwd[:, None] * ind_lb
            du1_3 += -dL_d3bwd[:, None] * ind_ub
            dl2_3 += -dL_d3bwd[:, None] * ind_lb
            du2_3 +=  dL_d3bwd[:, None] * ind_ub

            # Overlap grad: contradiction pushes overlap → 0
            # ∂overlap_k/∂l1_k = −1{l1≥l2}·active,  ∂/∂u1_k = +1{u1≤u2}·active
            dL_dov = (contra * self.overlap_weight / N)[:, None]   # (N,1)
            active = (olap_dim > 0).astype(float)
            dl1_3 += -dL_dov * (l1 >= l2).astype(float) * active
            du1_3 +=  dL_dov * (u1 <= u2).astype(float) * active
            dl2_3 += -dL_dov * (l2 >= l1).astype(float) * active
            du2_3 +=  dL_dov * (u2 <= u1).astype(float) * active

        l_total = λ1*l_rec + λ2*l_gal + λ3*l_tsk + λ4*l_ceg + λ5*l_3cls

        # ── Backward ─────────────────────────────────────────────────────────
        # ∂L/∂viol12 from all three loss terms
        # Galois: d/dviol12 = binary*(viol>0) - (1-binary)*(viol<margin)
        gal_active_ent  = binary       * (viol12 > 0).astype(float)
        gal_active_nent = (1 - binary) * (viol12 < self.margin).astype(float)
        dL_dviol = (λ2 * (gal_active_ent - gal_active_nent)) / N

        # Task BCE: dp/dviol = -scale/(1+scale*v)^2
        dp_dviol  = -self.scale / ((1 + self.scale * viol12) ** 2)
        dBCE_dp   = -(binary / p_task - (1-binary) / (1-p_task)) / N
        dL_dviol += λ3 * dBCE_dp * dp_dviol

        # CEGAR proxy gradient
        dL_dviol += λ4 * dceg_dviol12

        # ∂viol12/∂l1 = -1{l2 > l1},  ∂viol12/∂u1 = +1{u1 > u2}
        # ∂viol12/∂l2 = +1{l2 > l1},  ∂viol12/∂u2 = -1{u1 > u2}
        indicator_l = (l2 > l1).astype(float)   # (N, box_dim)
        indicator_u = (u1 > u2).astype(float)

        dl1_task = -dL_dviol[:, None] * indicator_l
        du1_task =  dL_dviol[:, None] * indicator_u
        dl2_task =  dL_dviol[:, None] * indicator_l
        du2_task = -dL_dviol[:, None] * indicator_u

        # Reconstruction: through decoder back to l, u
        dec_grads1, dl1_rec, du1_rec = self.model._backprop_decode(emb1, l1, u1, dcache1)
        dec_grads2, dl2_rec, du2_rec = self.model._backprop_decode(emb2, l2, u2, dcache2)

        # Total gradients on box bounds
        dl1 = dl1_task + λ1 * dl1_rec + λ5 * dl1_3
        du1 = du1_task + λ1 * du1_rec + λ5 * du1_3
        dl2 = dl2_task + λ1 * dl2_rec + λ5 * dl2_3
        du2 = du2_task + λ1 * du2_rec + λ5 * du2_3

        # Backprop through encoder for both sentences
        enc_grads1 = self.model._backprop_encode(dl1, du1, cache1)
        enc_grads2 = self.model._backprop_encode(dl2, du2, cache2)

        # Average encoder grads from both sentences
        enc_grads = {k: 0.5*(enc_grads1[k] + enc_grads2[k]) for k in enc_grads1}

        # Merge decoder grads
        dec_grads = {k: 0.5*(dec_grads1[k] + dec_grads2[k])
                     for k in ['Wd1','bd1','Wd2','bd2']}

        # Update all parameters
        all_grads = {**enc_grads, **dec_grads}
        params = self.model.params_dict()
        self.opt.update(params, all_grads)
        self.model.set_params(params)

        return {
            'total': l_total, 'recon': l_rec,
            'galois': l_gal,  'task': l_tsk,
            'cegar':  λ4 * l_ceg,
            '3class': λ5 * l_3cls,
        }

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, train_data, val_data, n_epochs=40, batch_size=64,
              verbose=True, patience=10):
        loader      = NLIDataLoader(train_data, batch_size=batch_size,
                                    shuffle=True, rng=self.rng)
        best_acc    = -1
        best_state  = None
        pat_ctr     = 0

        for epoch in range(1, n_epochs + 1):
            self._epoch = epoch
            # Flush any cold-start entries accumulated during warmup so that
            # CEGAR starts with a clean buffer of post-warmup mistakes only.
            if epoch == self.warmup_epochs + 1 and self.warmup_epochs > 0:
                self.cegar.clear()

            # Periodic flush: discard stale counterexamples whose geometry
            # is no longer representative (critical when the backbone is
            # being fine-tuned and the embedding space shifts each epoch).
            elif (self.flush_every > 0
                  and epoch > self.warmup_epochs
                  and (epoch - self.warmup_epochs) % self.flush_every == 0):
                self.cegar.clear()

            t0       = time.time()
            ep_losses = []

            for batch in loader:
                loss_d = self._step(batch['emb1'], batch['emb2'], batch['label'])
                ep_losses.append(loss_d)

            mean = {k: float(np.mean([l[k] for l in ep_losses])) for k in ep_losses[0]}
            val_m   = self.evaluate(val_data)
            elapsed = time.time() - t0

            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Loss={mean['total']:.4f} "
                      f"(R={mean['recon']:.3f} G={mean['galois']:.3f} "
                      f"T={mean['task']:.3f} C={mean['cegar']:.3f} "
                      f"3={mean['3class']:.3f}) | "
                      f"Val Acc={val_m['accuracy']:.4f} F1={val_m['f1']:.4f} | "
                      f"buf={len(self.cegar)} | {elapsed:.1f}s")

            self.history.append({
                'epoch': epoch, **mean,
                'val_acc': val_m['accuracy'],
                'val_f1':  val_m['f1'],
            })

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

    def evaluate_3class(self, train_data, test_data):
        """
        Fit a linear 3-class probe on box features extracted from *train_data*
        and evaluate it on *test_data*.

        The 4 features — viol(p→h), viol(h→p), mean_overlap, width_asymmetry —
        capture the directional geometry that distinguishes entailment (contained
        box) from neutral (partial overlap) and contradiction (disjoint).

        Returns
        -------
        dict with keys:
            acc_3class  : float — overall 3-way accuracy
            per_class   : {'entail': float, 'neutral': float, 'contra': float}
            confusion   : (3,3) numpy int array  (rows=true, cols=pred)
        """
        # Encode train set
        l1_tr, u1_tr, _ = self.model.encode(train_data['emb1'])
        l2_tr, u2_tr, _ = self.model.encode(train_data['emb2'])
        feat_tr = compute_3class_features(l1_tr, u1_tr, l2_tr, u2_tr)

        # Encode test set
        l1_te, u1_te, _ = self.model.encode(test_data['emb1'])
        l2_te, u2_te, _ = self.model.encode(test_data['emb2'])
        feat_te = compute_3class_features(l1_te, u1_te, l2_te, u2_te)

        probe = ThreeClassProbe().fit(feat_tr, train_data['label'])
        return probe.evaluate(feat_te, test_data['label'])
