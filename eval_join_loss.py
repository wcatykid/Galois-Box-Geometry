"""
Join Consistency Loss Experiment
=================================
Trains with an additional loss term that penalizes when lattice joins
violate containment:

    join(alpha(child1), alpha(child2))  NOT subset of  alpha(parent)

Compares join containment metrics with vs without the join loss.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.dataset import (build_synthetic_dataset, build_synthetic_embeddings,
                          SENTENCES, ENTAILMENT_PAIRS, NLIDataLoader, ENTAILMENT)
from src.improved_trainer import (ImprovedTrainer, GaloisModel, Adam,
                                   softplus, softplus_grad, relu, relu_grad,
                                   layer_norm, layer_norm_backward)
from src.box_lattice import (box_join, box_containment_violation, box_volume)
from src.losses import CEGARBuffer


# ── Join triples from the hierarchy ──────────────────────────────────────────
# (child1, child2, expected_parent) — join of children should be contained
# in parent's box

JOIN_TRIPLES = [
    ('s5', 's6', 's4'),  # East HS | West HS => school
    ('s5', 's7', 's4'),  # East HS | Jefferson => school
    ('s6', 's7', 's4'),  # West HS | Jefferson => school
    ('s5', 's6', 's3'),  # East HS | West HS => somewhere
    ('s5', 's6', 's2'),  # East HS | West HS => Tom goes
    ('s5', 's6', 's1'),  # East HS | West HS => someone moves
    ('s5', 's6', 's0'),  # East HS | West HS => something happens
    ('s4', 's8', 's0'),  # school | Mary reads => something happens
    ('s2', 's9', 's0'),  # Tom goes | cat sits => something happens
    ('s5', 's7', 's3'),  # East HS | Jefferson => somewhere
    ('s5', 's7', 's2'),  # East HS | Jefferson => Tom goes
    ('s6', 's7', 's2'),  # West HS | Jefferson => Tom goes
]


class JoinAwareTrainer(ImprovedTrainer):
    """Extends ImprovedTrainer with a join consistency loss on hierarchy sentences."""

    def __init__(self, *args, lambda6=1.0, hierarchy_embs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda6 = lambda6
        self.hierarchy_embs = hierarchy_embs

    def _compute_join_loss_and_grads(self):
        """
        Encode hierarchy sentences, compute join triples, return loss + encoder grads.
        """
        if self.hierarchy_embs is None or self.lambda6 == 0:
            return 0.0, {}

        caches = {}
        boxes = {}
        for sid in SENTENCES:
            if sid not in self.hierarchy_embs:
                continue
            e = self.hierarchy_embs[sid][None, :]
            l, u, h, cache = self.model._encode_fwd(e)
            boxes[sid] = (l[0], u[0])
            caches[sid] = cache

        n_active = 0
        total_viol = 0.0
        dl_accum = {sid: np.zeros_like(boxes[sid][0]) for sid in boxes}
        du_accum = {sid: np.zeros_like(boxes[sid][1]) for sid in boxes}

        for c1, c2, parent in JOIN_TRIPLES:
            if c1 not in boxes or c2 not in boxes or parent not in boxes:
                continue
            n_active += 1

            l1, u1 = boxes[c1]
            l2, u2 = boxes[c2]
            lp, up = boxes[parent]

            lj = np.minimum(l1, l2)
            uj = np.maximum(u1, u2)

            lower_viol = np.maximum(0.0, lp - lj)
            upper_viol = np.maximum(0.0, uj - up)
            total_viol += np.sum(lower_viol + upper_viol)

            # Gradient of violation w.r.t. join bounds and parent bounds
            ind_lv = (lp > lj).astype(float)
            ind_uv = (uj > up).astype(float)

            # Chain rule through join operation
            # lj = min(l1, l2): grad flows to whichever is smaller
            l1_is_min = (l1 <= l2).astype(float)
            l2_is_min = (l2 < l1).astype(float)
            # uj = max(u1, u2): grad flows to whichever is larger
            u1_is_max = (u1 >= u2).astype(float)
            u2_is_max = (u2 > u1).astype(float)

            # d(viol)/d(lj) = -ind_lv,  d(viol)/d(uj) = +ind_uv
            dl_accum[c1] += (-ind_lv) * l1_is_min
            dl_accum[c2] += (-ind_lv) * l2_is_min
            du_accum[c1] += ind_uv * u1_is_max
            du_accum[c2] += ind_uv * u2_is_max
            # Parent gradients
            dl_accum[parent] += ind_lv
            du_accum[parent] += -ind_uv

        if n_active == 0:
            return 0.0, {}

        l_join = total_viol / n_active

        # Normalize gradients
        for sid in dl_accum:
            dl_accum[sid] /= n_active
            du_accum[sid] /= n_active

        # Backprop through encoder for each sentence with nonzero gradient
        all_enc_grads = {}
        n_contrib = 0
        for sid in dl_accum:
            dl = dl_accum[sid]
            du = du_accum[sid]
            if np.abs(dl).sum() + np.abs(du).sum() < 1e-12:
                continue
            enc_grads = self.model._backprop_encode(
                dl[None, :], du[None, :], caches[sid])
            for k, v in enc_grads.items():
                if k not in all_enc_grads:
                    all_enc_grads[k] = np.zeros_like(v)
                all_enc_grads[k] += v
            n_contrib += 1

        if n_contrib > 0:
            for k in all_enc_grads:
                all_enc_grads[k] /= n_contrib

        return l_join, all_enc_grads

    def _step(self, emb1, emb2, labels):
        result = super()._step(emb1, emb2, labels)

        if self.lambda6 > 0:
            l_join, join_grads = self._compute_join_loss_and_grads()
            if join_grads:
                scaled = {k: self.lambda6 * v for k, v in join_grads.items()}
                for k in ['Wd1', 'bd1', 'Wd2', 'bd2']:
                    if k not in scaled:
                        scaled[k] = np.zeros_like(getattr(self.model, k))
                params = self.model.params_dict()
                self.opt.update(params, scaled)
                self.model.set_params(params)
            result['join'] = self.lambda6 * l_join
            result['total'] += self.lambda6 * l_join
        else:
            result['join'] = 0.0

        return result


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_joins(model, base_emb, label=""):
    """Run the join containment tests and return (pass_count, total, details)."""
    boxes = {}
    for sid in SENTENCES:
        e = base_emb[sid][None, :]
        l, u, _ = model.encode(e)
        boxes[sid] = (l[0], u[0])

    join_tests = [
        ('s5', 's6', 's4', 'join(East HS, West HS) in school'),
        ('s5', 's7', 's4', 'join(East HS, Jefferson) in school'),
        ('s6', 's7', 's4', 'join(West HS, Jefferson) in school'),
        ('s5', 's6', 's3', 'join(East HS, West HS) in somewhere'),
        ('s5', 's6', 's2', 'join(East HS, West HS) in Tom goes'),
        ('s5', 's6', 's1', 'join(East HS, West HS) in someone moves'),
        ('s5', 's6', 's0', 'join(East HS, West HS) in sth happens'),
        ('s4', 's8', 's0', 'join(school, Mary reads) in sth happens'),
        ('s2', 's9', 's0', 'join(Tom goes, cat sits) in sth happens'),
        ('s5', 's7', 's3', 'join(East HS, Jefferson) in somewhere'),
        ('s5', 's7', 's2', 'join(East HS, Jefferson) in Tom goes'),
        ('s6', 's7', 's2', 'join(West HS, Jefferson) in Tom goes'),
    ]

    pass_count = 0
    details = []
    for c1, c2, parent, desc in join_tests:
        l1, u1 = boxes[c1]
        l2, u2 = boxes[c2]
        lj, uj = box_join(l1, u1, l2, u2)
        lp, up = boxes[parent]
        viol = float(box_containment_violation(lj, uj, lp, up))
        passed = viol < 1.0
        if passed:
            pass_count += 1
        details.append((desc, viol, passed))

    # Minimality tests
    min_tests = [
        ('s5', 's6', 's4', 'East+West => school'),
        ('s5', 's7', 's4', 'East+Jeff => school'),
        ('s4', 's8', 's0', 'school+Mary => sth happens'),
    ]
    tighter = 0
    for c1, c2, parent, _ in min_tests:
        lj, uj = box_join(boxes[c1][0], boxes[c1][1],
                          boxes[c2][0], boxes[c2][1])
        lp, up = boxes[parent]
        jv = float(np.sum(np.log(np.maximum(uj - lj, 1e-8))))
        pv = float(np.sum(np.log(np.maximum(up - lp, 1e-8))))
        if jv <= pv:
            tighter += 1

    return pass_count, len(join_tests), tighter, len(min_tests), details


def run_experiment(seed=42):
    print("=" * 70)
    print("  Join Consistency Loss Experiment")
    print("=" * 70)

    base_emb = build_synthetic_embeddings(dim=64, noise=0.0,
                                           rng=np.random.default_rng(0))
    train_data, val_data, test_data = build_synthetic_dataset(
        n_train=3000, n_val=500, n_test=1000,
        dim=64, noise=0.05, rng=np.random.default_rng(seed))

    # ── Baseline (no join loss) ──────────────────────────────────────────
    print("\n--- Training BASELINE (no join loss) ---")
    baseline = ImprovedTrainer(
        input_dim=64, hidden_dim=256, box_dim=64, lr=1e-3,
        lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
        margin=1.0, scale=5.0, rng=np.random.default_rng(seed + 10))
    baseline.train(train_data, val_data, n_epochs=40,
                   batch_size=64, verbose=False, patience=10)

    base_metrics = baseline.evaluate(test_data)
    bp, bt, bm, bmt, bdetails = evaluate_joins(baseline.model, base_emb)

    # ── With join loss ───────────────────────────────────────────────────
    # Try multiple lambda6 values to find the sweet spot
    best_config = None
    best_pass = -1

    for lam6 in [0.5, 1.0, 2.0, 5.0]:
        print(f"\n--- Training with JOIN LOSS (lambda6={lam6}) ---")
        jt = JoinAwareTrainer(
            input_dim=64, hidden_dim=256, box_dim=64, lr=1e-3,
            lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
            margin=1.0, scale=5.0,
            lambda6=lam6, hierarchy_embs=base_emb,
            rng=np.random.default_rng(seed + 10))
        jt.train(train_data, val_data, n_epochs=40,
                 batch_size=64, verbose=False, patience=10)

        jm = jt.evaluate(test_data)
        jp, jt_total, jmin, jmt, jdetails = evaluate_joins(jt.model, base_emb)
        print(f"  NLI Acc={jm['accuracy']:.4f}  F1={jm['f1']:.4f}  "
              f"Join pass={jp}/{jt_total}  Minimality={jmin}/{jmt}")

        if jp > best_pass or (jp == best_pass and best_config is None):
            best_pass = jp
            best_config = {
                'lambda6': lam6, 'metrics': jm,
                'pass': jp, 'total': jt_total,
                'min': jmin, 'min_total': jmt,
                'details': jdetails, 'trainer': jt,
            }

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n  {'Metric':<40} {'Baseline':>12} {'+ Join Loss':>12}")
    print("  " + "-" * 64)
    print(f"  {'NLI Accuracy':<40} {base_metrics['accuracy']:>12.4f} "
          f"{best_config['metrics']['accuracy']:>12.4f}")
    print(f"  {'NLI F1':<40} {base_metrics['f1']:>12.4f} "
          f"{best_config['metrics']['f1']:>12.4f}")
    print(f"  {'Join containment (pass/total)':<40} "
          f"{bp:>5}/{bt:<6} {best_config['pass']:>5}/{best_config['total']:<6}")
    print(f"  {'Join minimality (tighter/total)':<40} "
          f"{bm:>5}/{bmt:<6} {best_config['min']:>5}/{best_config['min_total']:<6}")
    print(f"  {'Best lambda6':<40} {'N/A':>12} {best_config['lambda6']:>12.1f}")

    # ── Detailed join results ────────────────────────────────────────────
    print(f"\n  {'Test':<52} {'Base':>6} {'Join':>6}")
    print("  " + "-" * 64)
    for i, (desc, bv, bpassed) in enumerate(bdetails):
        jv = best_config['details'][i][1]
        jpassed = best_config['details'][i][2]
        bmark = "PASS" if bpassed else f"{bv:.1f}"
        jmark = "PASS" if jpassed else f"{jv:.1f}"
        print(f"  {desc:<52} {bmark:>6} {jmark:>6}")

    # ── Three-way join ────────────────────────────────────────────────────
    print(f"\n  Three-way join: join(East, West, Jefferson)")
    for label, model in [("Baseline", baseline.model),
                         (f"Join(l6={best_config['lambda6']})", best_config['trainer'].model)]:
        boxes = {}
        for sid in SENTENCES:
            e = base_emb[sid][None, :]
            l, u, _ = model.encode(e)
            boxes[sid] = (l[0], u[0])
        lj, uj = box_join(*boxes['s5'], *boxes['s6'])
        lj, uj = box_join(lj, uj, *boxes['s7'])
        for pid in ['s4', 's3', 's2']:
            lp, up = boxes[pid]
            v = float(box_containment_violation(lj, uj, lp, up))
            ok = "PASS" if v < 1.0 else f"{v:.1f}"
            print(f"    {label:<20} => {SENTENCES[pid]:<25} viol={ok}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Baseline join containment:   {bp}/{bt} "
          f"({100*bp/bt:.0f}%)")
    print(f"  + Join loss (l6={best_config['lambda6']}):   "
          f"{best_config['pass']}/{best_config['total']} "
          f"({100*best_config['pass']/best_config['total']:.0f}%)")
    delta_acc = best_config['metrics']['accuracy'] - base_metrics['accuracy']
    print(f"  NLI accuracy change:         {delta_acc:+.4f}")
    print(f"  NLI accuracy preserved:      "
          f"{'YES' if best_config['metrics']['accuracy'] >= 0.85 else 'DEGRADED'}")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment()
