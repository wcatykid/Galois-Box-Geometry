"""
Sentence-Level Join Experiment on SNLI + SBERT
===============================================
Extracts join triples from SNLI: if two different premises both entail
the same hypothesis, then join(α(p1), α(p2)) ⊑ α(h).

Trains with and without join consistency loss using pre-computed SBERT
embeddings and the NumPy-based Galois trainer.
"""
import sys, os
if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from collections import defaultdict

from src.improved_trainer import ImprovedTrainer
from src.box_lattice import box_join, box_containment_violation

SEED = 42
BOX_DIM = 128
HIDDEN_DIM = 512
N_TRAIN_PAIRS = 30000
N_VAL_PAIRS = 3000
MAX_JOIN_TRIPLES = 500
TRIPLE_BATCH_SIZE = 64


def load_snli_entailment():
    """Load SNLI and return entailment pairs + all labeled pairs."""
    from datasets import load_dataset
    snli = load_dataset('snli', split='train')
    snli = snli.filter(lambda x: x['label'] != -1)

    entailment_pairs = []
    all_pairs = []
    for ex in snli:
        p, h, lab = ex['premise'], ex['hypothesis'], ex['label']
        all_pairs.append((p, h, lab))
        if lab == 0:
            entailment_pairs.append((p, h))

    return entailment_pairs, all_pairs


def extract_join_triples(entailment_pairs, max_triples=MAX_JOIN_TRIPLES,
                         rng=None):
    """
    Find (premise1, premise2, hypothesis) triples where both premises
    entail the same hypothesis.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    hyp_to_premises = defaultdict(list)
    for p, h in entailment_pairs:
        hyp_to_premises[h].append(p)

    # Only hypotheses with >= 2 distinct premises
    triples = []
    for h, premises in hyp_to_premises.items():
        premises = list(set(premises))
        if len(premises) < 2:
            continue
        n_pairs = min(len(premises) * (len(premises) - 1) // 2, 3)
        indices = list(range(len(premises)))
        rng.shuffle(indices)
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                triples.append((premises[indices[i]], premises[indices[j]], h))
                count += 1
                if count >= n_pairs:
                    break
            if count >= n_pairs:
                break

    rng.shuffle(triples)
    if len(triples) > max_triples:
        triples = triples[:max_triples]

    return triples


def encode_sentences_sbert(sentences, batch_size=256):
    """Encode sentences with SBERT, return dict of sentence -> embedding."""
    from sentence_transformers import SentenceTransformer
    print("  Loading SBERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sent_list = list(sentences)
    print(f"  Encoding {len(sent_list)} unique sentences...")
    embeddings = model.encode(sent_list, batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=True)
    emb_dict = {s: embeddings[i] for i, s in enumerate(sent_list)}
    dim = embeddings.shape[1]
    print(f"  SBERT dim={dim}")
    del model
    return emb_dict, dim


def build_training_data(all_pairs, emb_dict, n_train=N_TRAIN_PAIRS,
                        n_val=N_VAL_PAIRS, rng=None):
    """Build train/val from SNLI pairs using pre-computed SBERT embeddings."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    # Map SNLI 3-class labels: 0=entailment, 1=neutral, 2=contradiction
    emb1_list, emb2_list, lab_list = [], [], []
    for p, h, lab in all_pairs:
        if p in emb_dict and h in emb_dict:
            emb1_list.append(emb_dict[p])
            emb2_list.append(emb_dict[h])
            lab_list.append(lab)

    emb1 = np.stack(emb1_list)
    emb2 = np.stack(emb2_list)
    labs = np.array(lab_list)

    # Subsample
    n_total = min(n_train + n_val, len(labs))
    idx = rng.choice(len(labs), n_total, replace=False)
    emb1, emb2, labs = emb1[idx], emb2[idx], labs[idx]

    # Split
    n_val_actual = min(n_val, n_total // 10)
    perm = rng.permutation(n_total)
    tr_idx, vl_idx = perm[n_val_actual:], perm[:n_val_actual]

    train_data = {'emb1': emb1[tr_idx], 'emb2': emb2[tr_idx],
                  'label': labs[tr_idx]}
    val_data = {'emb1': emb1[vl_idx], 'emb2': emb2[vl_idx],
                'label': labs[vl_idx]}
    return train_data, val_data


def evaluate_join_containment(model, triples, emb_dict, threshold=1.0):
    """Test join(α(p1), α(p2)) contained in α(hypothesis)."""
    boxes = {}
    for trip in triples:
        for s in trip:
            if s not in boxes and s in emb_dict:
                e = emb_dict[s][None, :]
                l, u, _ = model.encode(e)
                boxes[s] = (l[0], u[0])

    pass_count = 0
    total_viol = 0.0
    details = []
    for p1, p2, hyp in triples:
        if p1 not in boxes or p2 not in boxes or hyp not in boxes:
            continue
        l1, u1 = boxes[p1]
        l2, u2 = boxes[p2]
        lh, uh = boxes[hyp]
        lj, uj = box_join(l1, u1, l2, u2)
        viol = float(box_containment_violation(lj, uj, lh, uh))
        passed = viol < threshold
        if passed:
            pass_count += 1
        total_viol += viol
        details.append((p1, p2, hyp, viol, passed))

    n = len(details)
    return {
        'pass': pass_count,
        'total': n,
        'rate': pass_count / max(n, 1),
        'mean_viol': total_viol / max(n, 1),
        'details': details,
    }


class SNLIJoinTrainer(ImprovedTrainer):
    """Join-aware trainer using sentence-level triples from SNLI."""

    def __init__(self, *args, lambda6=1.0, join_triples=None,
                 sent_embs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda6 = lambda6
        self.join_triples = join_triples or []
        self.sent_embs = sent_embs or {}
        self._triple_batch_size = min(TRIPLE_BATCH_SIZE,
                                      len(self.join_triples))

    def _compute_join_loss_and_grads(self):
        if not self.join_triples or self.lambda6 == 0:
            return 0.0, {}

        rng = self.rng
        idx = rng.choice(len(self.join_triples), self._triple_batch_size,
                         replace=False)
        batch_triples = [self.join_triples[i] for i in idx]

        sents_needed = set()
        for p1, p2, h in batch_triples:
            sents_needed.update([p1, p2, h])
        sent_list = [s for s in sents_needed if s in self.sent_embs]
        if len(sent_list) < 3:
            return 0.0, {}

        sent_to_idx = {s: i for i, s in enumerate(sent_list)}
        emb_batch = np.stack([self.sent_embs[s] for s in sent_list])

        l_all, u_all, h_all, cache = self.model._encode_fwd(emb_batch)

        N = len(sent_list)
        dl_batch = np.zeros((N, l_all.shape[1]))
        du_batch = np.zeros((N, u_all.shape[1]))

        n_active = 0
        total_viol = 0.0
        for p1, p2, hyp in batch_triples:
            if p1 not in sent_to_idx or p2 not in sent_to_idx or hyp not in sent_to_idx:
                continue
            i1, i2, ih = sent_to_idx[p1], sent_to_idx[p2], sent_to_idx[hyp]
            n_active += 1

            l1, u1 = l_all[i1], u_all[i1]
            l2, u2 = l_all[i2], u_all[i2]
            lh, uh = l_all[ih], u_all[ih]

            lj = np.minimum(l1, l2)
            uj = np.maximum(u1, u2)

            lower_viol = np.maximum(0.0, lh - lj)
            upper_viol = np.maximum(0.0, uj - uh)
            total_viol += np.sum(lower_viol + upper_viol)

            ind_lv = (lh > lj).astype(float)
            ind_uv = (uj > uh).astype(float)

            l1_is_min = (l1 <= l2).astype(float)
            l2_is_min = (l2 < l1).astype(float)
            u1_is_max = (u1 >= u2).astype(float)
            u2_is_max = (u2 > u1).astype(float)

            dl_batch[i1] += (-ind_lv) * l1_is_min
            dl_batch[i2] += (-ind_lv) * l2_is_min
            du_batch[i1] += ind_uv * u1_is_max
            du_batch[i2] += ind_uv * u2_is_max
            dl_batch[ih] += ind_lv
            du_batch[ih] += -ind_uv

        if n_active == 0:
            return 0.0, {}

        l_join = total_viol / n_active
        dl_batch /= n_active
        du_batch /= n_active

        enc_grads = self.model._backprop_encode(dl_batch, du_batch, cache)
        return l_join, enc_grads

    def _step(self, emb1, emb2, labels):
        result = super()._step(emb1, emb2, labels)

        if self.lambda6 > 0:
            l_join, join_grads = self._compute_join_loss_and_grads()
            if join_grads:
                scaled = {k: self.lambda6 * v for k, v in join_grads.items()}
                params = self.model.params_dict()
                self.opt.update(params, scaled)
                self.model.set_params(params)
            result['join'] = self.lambda6 * l_join
            result['total'] += self.lambda6 * l_join
        else:
            result['join'] = 0.0

        return result


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("  Sentence-Level Join Experiment: SNLI + SBERT")
    print("=" * 70)

    # Load SNLI
    print("\nLoading SNLI dataset...")
    t0 = time.time()
    entailment_pairs, all_pairs = load_snli_entailment()
    print(f"  Entailment pairs: {len(entailment_pairs)}")
    print(f"  All pairs: {len(all_pairs)}")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # Extract join triples
    print("\nExtracting sentence-level join triples...")
    join_triples = extract_join_triples(entailment_pairs,
                                        max_triples=MAX_JOIN_TRIPLES, rng=rng)
    print(f"  Join triples: {len(join_triples)}")

    # Collect all unique sentences needed
    all_sentences = set()
    for p1, p2, h in join_triples:
        all_sentences.update([p1, p2, h])
    # Also collect sentences from training pairs (subsample for speed)
    train_sample_idx = rng.choice(len(all_pairs),
                                  min(N_TRAIN_PAIRS + N_VAL_PAIRS, len(all_pairs)),
                                  replace=False)
    for i in train_sample_idx:
        p, h, _ = all_pairs[i]
        all_sentences.update([p, h])
    print(f"  Unique sentences to encode: {len(all_sentences)}")

    # Encode with SBERT
    print("\nEncoding with SBERT...")
    t0 = time.time()
    emb_dict, input_dim = encode_sentences_sbert(all_sentences)
    print(f"  Encoding time: {time.time() - t0:.1f}s")

    # Filter triples to those with embeddings
    join_triples = [(p1, p2, h) for p1, p2, h in join_triples
                    if p1 in emb_dict and p2 in emb_dict and h in emb_dict]
    unique_sents = set()
    for p1, p2, h in join_triples:
        unique_sents.update([p1, p2, h])
    print(f"\n  After filter: {len(join_triples)} triples, "
          f"{len(unique_sents)} unique sentences")

    # Show examples
    print("\n  Sample triples:")
    for p1, p2, h in join_triples[:5]:
        p1_short = p1[:40] + "..." if len(p1) > 40 else p1
        p2_short = p2[:40] + "..." if len(p2) > 40 else p2
        print(f"    join(\"{p1_short}\",")
        print(f"         \"{p2_short}\")")
        print(f"      should be in \"{h}\"")
        print()

    # Build training data
    print("Building SNLI training pairs...")
    train_pairs_subset = [all_pairs[i] for i in train_sample_idx]
    train_data, val_data = build_training_data(train_pairs_subset, emb_dict,
                                                rng=rng)
    print(f"  Train: {len(train_data['label'])}  "
          f"Val: {len(val_data['label'])}  dim={input_dim}")

    # Build sentence embedding lookup for join triples
    join_embs = {}
    for p1, p2, h in join_triples:
        for s in [p1, p2, h]:
            if s in emb_dict:
                join_embs[s] = emb_dict[s]

    # ── Baseline ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Training BASELINE (no join loss)...")
    print("-" * 70)
    t0 = time.time()
    baseline = ImprovedTrainer(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM, lr=5e-4,
        lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
        margin=1.0, scale=5.0,
        rng=np.random.default_rng(SEED + 10))
    baseline.train(train_data, val_data, n_epochs=50, batch_size=256,
                   verbose=True, patience=7)
    base_time = time.time() - t0
    print(f"  Baseline training: {base_time:.1f}s")

    base_metrics = baseline.evaluate(val_data)
    base_joins = evaluate_join_containment(baseline.model, join_triples,
                                           emb_dict)
    print(f"\n  Baseline NLI Acc: {base_metrics['accuracy']:.4f}")
    print(f"  Baseline join pass: {base_joins['pass']}/{base_joins['total']} "
          f"({base_joins['rate']:.1%})")
    print(f"  Baseline mean violation: {base_joins['mean_viol']:.2f}")

    # ── With join loss ──────────────────────────────────────────────────
    best_config = None
    best_pass_rate = -1

    for lam6 in [1.0, 2.0]:
        print(f"\n" + "-" * 70)
        print(f"  Training with JOIN LOSS (lambda6={lam6})...")
        print("-" * 70)
        t0 = time.time()

        jt = SNLIJoinTrainer(
            input_dim=input_dim, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM,
            lr=5e-4,
            lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
            margin=1.0, scale=5.0,
            lambda6=lam6,
            join_triples=join_triples,
            sent_embs=join_embs,
            rng=np.random.default_rng(SEED + 10))
        jt.train(train_data, val_data, n_epochs=50, batch_size=256,
                 verbose=True, patience=7)
        jt_time = time.time() - t0

        jm = jt.evaluate(val_data)
        jj = evaluate_join_containment(jt.model, join_triples, emb_dict)
        print(f"\n  lambda6={lam6}: NLI Acc={jm['accuracy']:.4f}  "
              f"Join pass={jj['pass']}/{jj['total']} ({jj['rate']:.1%})  "
              f"Mean viol={jj['mean_viol']:.2f}  Time={jt_time:.1f}s")

        if jj['rate'] > best_pass_rate:
            best_pass_rate = jj['rate']
            best_config = {
                'lambda6': lam6, 'metrics': jm, 'joins': jj,
                'time': jt_time, 'model': jt.model,
            }

    # ── Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    bj = base_joins
    jj = best_config['joins']
    print(f"\n  {'Metric':<40} {'Baseline':>12} {'+ Join Loss':>12}")
    print("  " + "-" * 64)
    print(f"  {'NLI Accuracy (val)':<40} {base_metrics['accuracy']:>12.4f} "
          f"{best_config['metrics']['accuracy']:>12.4f}")
    print(f"  {'Join containment rate':<40} "
          f"{bj['rate']:>11.1%} {jj['rate']:>11.1%}")
    print(f"  {'Join pass/total':<40} "
          f"{bj['pass']:>5}/{bj['total']:<6} {jj['pass']:>5}/{jj['total']:<6}")
    print(f"  {'Mean violation':<40} {bj['mean_viol']:>12.2f} "
          f"{jj['mean_viol']:>12.2f}")
    print(f"  {'Best lambda6':<40} {'N/A':>12} "
          f"{best_config['lambda6']:>12.1f}")

    # Sample results
    print(f"\n  Sample results (best join-loss model):")
    print(f"  {'Premise 1':<30} {'Premise 2':<30} {'Hypothesis':<25} "
          f"{'Viol':>8} {'Pass?':>6}")
    print("  " + "-" * 101)
    for p1, p2, h, v, passed in jj['details'][:15]:
        p1s = (p1[:27] + "...") if len(p1) > 27 else p1
        p2s = (p2[:27] + "...") if len(p2) > 27 else p2
        hs = (h[:22] + "...") if len(h) > 22 else h
        mark = "PASS" if passed else f"{v:.1f}"
        print(f"  {p1s:<30} {p2s:<30} {hs:<25} {v:>8.2f} {mark:>6}")

    # Violation distribution
    viols_base = [d[3] for d in bj['details']]
    viols_join = [d[3] for d in jj['details']]
    print(f"\n  Violation distribution:")
    print(f"  {'Percentile':<20} {'Baseline':>12} {'+ Join Loss':>12}")
    print("  " + "-" * 44)
    for pct in [25, 50, 75, 90, 95]:
        print(f"  {f'p{pct}':<20} {np.percentile(viols_base, pct):>12.2f} "
              f"{np.percentile(viols_join, pct):>12.2f}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  SNLI join triples tested:       {bj['total']}")
    print(f"  Baseline join containment:      {bj['pass']}/{bj['total']} "
          f"({bj['rate']:.1%})")
    print(f"  + Join loss (l6={best_config['lambda6']}):        "
          f"{jj['pass']}/{jj['total']} ({jj['rate']:.1%})")
    delta_acc = best_config['metrics']['accuracy'] - base_metrics['accuracy']
    print(f"  NLI accuracy change:            {delta_acc:+.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
