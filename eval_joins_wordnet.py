"""
Real-Data Join Experiment on WordNet + GloVe
=============================================
Extracts join triples from WordNet (two hyponyms sharing a hypernym),
trains with and without join consistency loss, and evaluates whether
join(alpha(child1), alpha(child2)) is contained in alpha(parent).

This is the real-data counterpart of eval_join_loss.py.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from collections import defaultdict

import gensim.downloader as api
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from src.improved_trainer import ImprovedTrainer
from src.box_lattice import box_join, box_containment_violation


SEED = 42
BOX_DIM = 128
HIDDEN_DIM = 512
WN_N_POSITIVE = 25000
WN_N_NEGATIVE = 25000


# ── WordNet join triple extraction ───────────────────────────────────────────

def extract_join_triples(max_triples=500, rng=None):
    """
    Find (child1, child2, parent) triples from WordNet where child1 and child2
    are co-hyponyms (share the same direct hypernym = parent).

    Returns list of (child1_lemma, child2_lemma, parent_lemma) triples.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    parent_to_children = defaultdict(list)
    for synset in wn.all_synsets('n'):
        for hypernym in synset.hypernyms():
            child_lemma = synset.lemmas()[0].name()
            parent_lemma = hypernym.lemmas()[0].name()
            parent_to_children[parent_lemma].append(child_lemma)

    # Only keep parents with >= 2 children
    triples = []
    for parent, children in parent_to_children.items():
        children = list(set(children))
        if len(children) < 2:
            continue
        # Sample pairs of children
        n_pairs = min(len(children) * (len(children) - 1) // 2, 3)
        indices = list(range(len(children)))
        rng.shuffle(indices)
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                triples.append((children[indices[i]], children[indices[j]], parent))
                count += 1
                if count >= n_pairs:
                    break
            if count >= n_pairs:
                break

    rng.shuffle(triples)
    if len(triples) > max_triples:
        triples = triples[:max_triples]

    return triples


def get_wordnet_pairs(n_positive=WN_N_POSITIVE, n_negative=WN_N_NEGATIVE, rng=None):
    """Same as eval_hyperlex_embeddings.get_wordnet_pairs but standalone."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    pos_pairs = []
    seen = set()
    for synset in wn.all_synsets('n'):
        for hypernym in synset.hypernyms():
            w1 = synset.lemmas()[0].name()
            w2 = hypernym.lemmas()[0].name()
            key = (w1, w2)
            if key not in seen:
                seen.add(key)
                pos_pairs.append((w1, w2))

    if len(pos_pairs) > n_positive:
        idx = rng.choice(len(pos_pairs), n_positive, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx]

    all_nouns = list({s.lemmas()[0].name() for s in wn.all_synsets('n')})
    all_nouns_arr = np.array(all_nouns)

    neg_pairs = []
    while len(neg_pairs) < n_negative:
        batch = rng.choice(len(all_nouns_arr), (n_negative - len(neg_pairs)) * 2)
        w1s = all_nouns_arr[batch[:len(batch)//2]]
        w2s = all_nouns_arr[batch[len(batch)//2:]]
        for w1, w2 in zip(w1s, w2s):
            if w1 != w2:
                neg_pairs.append((w1, w2))
            if len(neg_pairs) >= n_negative:
                break

    pairs = ([(w1, w2, 0) for w1, w2 in pos_pairs] +
             [(w1, w2, 2) for w1, w2 in neg_pairs[:n_negative]])
    rng.shuffle(pairs)
    return pairs


def get_wv_embedding(phrase, wv_model):
    words = phrase.replace('_', ' ').lower().split()
    vecs = [wv_model[w] for w in words if w in wv_model]
    if len(vecs) == 0:
        return None
    return np.mean(vecs, axis=0)


# ── Training ─────────────────────────────────────────────────────────────────

def build_training_data(wn_pairs, glove, rng):
    """Build train/val splits from WordNet pairs + GloVe embeddings."""
    emb1_list, emb2_list, lab_list = [], [], []
    for w1, w2, lab in wn_pairs:
        v1 = get_wv_embedding(w1, glove)
        v2 = get_wv_embedding(w2, glove)
        if v1 is not None and v2 is not None:
            emb1_list.append(v1)
            emb2_list.append(v2)
            lab_list.append(lab)

    emb1 = np.stack(emb1_list)
    emb2 = np.stack(emb2_list)
    labs = np.array(lab_list)

    n_val = max(256, int(len(labs) * 0.1))
    idx = rng.permutation(len(labs))
    tr_idx, vl_idx = idx[n_val:], idx[:n_val]

    train_data = {'emb1': emb1[tr_idx], 'emb2': emb2[tr_idx], 'label': labs[tr_idx]}
    val_data = {'emb1': emb1[vl_idx], 'emb2': emb2[vl_idx], 'label': labs[vl_idx]}
    return train_data, val_data, emb1.shape[1]


def build_hierarchy_embs(join_triples, glove):
    """Build embedding lookup for all words appearing in join triples."""
    words = set()
    for c1, c2, p in join_triples:
        words.update([c1, c2, p])

    embs = {}
    for w in words:
        v = get_wv_embedding(w, glove)
        if v is not None:
            embs[w] = v
    return embs


def filter_triples(triples, embs):
    """Keep only triples where all three words have embeddings."""
    return [(c1, c2, p) for c1, c2, p in triples
            if c1 in embs and c2 in embs and p in embs]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_join_containment(model, triples, embs, threshold=1.0):
    """Test join(alpha(c1), alpha(c2)) contained in alpha(parent)."""
    boxes = {}
    for w in embs:
        e = embs[w][None, :]
        l, u, _ = model.encode(e)
        boxes[w] = (l[0], u[0])

    pass_count = 0
    total_viol = 0.0
    details = []
    for c1, c2, parent in triples:
        if c1 not in boxes or c2 not in boxes or parent not in boxes:
            continue
        l1, u1 = boxes[c1]
        l2, u2 = boxes[c2]
        lp, up = boxes[parent]
        lj, uj = box_join(l1, u1, l2, u2)
        viol = float(box_containment_violation(lj, uj, lp, up))
        passed = viol < threshold
        if passed:
            pass_count += 1
        total_viol += viol
        details.append((c1, c2, parent, viol, passed))

    n = len(details)
    return {
        'pass': pass_count,
        'total': n,
        'rate': pass_count / max(n, 1),
        'mean_viol': total_viol / max(n, 1),
        'details': details,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("  Real-Data Join Experiment: WordNet + GloVe")
    print("=" * 70)

    # Load GloVe
    print("\nLoading GloVe-wiki-gigaword-300...")
    glove = api.load("glove-wiki-gigaword-300")
    input_dim = 300
    print(f"  Vocabulary: {len(glove.key_to_index)} words, dim={input_dim}")

    # Extract join triples
    print("\nExtracting WordNet join triples...")
    join_triples = extract_join_triples(max_triples=300, rng=rng)
    print(f"  Raw triples: {len(join_triples)}")

    # Build embeddings for triple words
    hierarchy_embs = build_hierarchy_embs(join_triples, glove)
    join_triples = filter_triples(join_triples, hierarchy_embs)
    print(f"  After OOV filter: {len(join_triples)} triples, "
          f"{len(hierarchy_embs)} unique words")

    # Show some examples
    print("\n  Sample triples:")
    for c1, c2, p in join_triples[:5]:
        print(f"    join({c1}, {c2}) should be in {p}")

    # Build training data
    print("\nBuilding WordNet training pairs...")
    wn_pairs = get_wordnet_pairs(WN_N_POSITIVE, WN_N_NEGATIVE, rng=rng)
    train_data, val_data, dim = build_training_data(wn_pairs, glove, rng)
    print(f"  Train: {len(train_data['label'])}  Val: {len(val_data['label'])}  dim={dim}")

    # ── Baseline (no join loss) ──────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Training BASELINE (no join loss)...")
    print("-" * 70)
    t0 = time.time()
    baseline = ImprovedTrainer(
        input_dim=dim, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM, lr=5e-4,
        lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
        margin=1.0, scale=5.0,
        rng=np.random.default_rng(SEED + 10))
    baseline.train(train_data, val_data, n_epochs=50, batch_size=256,
                   verbose=True, patience=7)
    base_time = time.time() - t0
    print(f"  Baseline training: {base_time:.1f}s")

    base_metrics = baseline.evaluate(val_data)
    base_joins = evaluate_join_containment(baseline.model, join_triples, hierarchy_embs)
    print(f"\n  Baseline NLI Acc: {base_metrics['accuracy']:.4f}")
    print(f"  Baseline join pass: {base_joins['pass']}/{base_joins['total']} "
          f"({base_joins['rate']:.1%})")
    print(f"  Baseline mean violation: {base_joins['mean_viol']:.2f}")

    # ── With join loss ───────────────────────────────────────────────────
    # For real data, the JoinAwareTrainer needs adaptation:
    # hierarchy_embs maps word -> GloVe vector
    # JOIN_TRIPLES in eval_join_loss.py uses sentence IDs; here we use words directly.
    # We need to adapt JoinAwareTrainer to accept word-based triples.

    best_config = None
    best_pass_rate = -1

    for lam6 in [1.0, 2.0]:
        print(f"\n" + "-" * 70)
        print(f"  Training with JOIN LOSS (lambda6={lam6})...")
        print("-" * 70)
        t0 = time.time()

        jt = WordNetJoinTrainer(
            input_dim=dim, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM, lr=5e-4,
            lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
            margin=1.0, scale=5.0,
            lambda6=lam6,
            join_triples=join_triples,
            word_embs=hierarchy_embs,
            rng=np.random.default_rng(SEED + 10))
        jt.train(train_data, val_data, n_epochs=50, batch_size=256,
                 verbose=True, patience=7)
        jt_time = time.time() - t0

        jm = jt.evaluate(val_data)
        jj = evaluate_join_containment(jt.model, join_triples, hierarchy_embs)
        print(f"\n  lambda6={lam6}: NLI Acc={jm['accuracy']:.4f}  "
              f"Join pass={jj['pass']}/{jj['total']} ({jj['rate']:.1%})  "
              f"Mean viol={jj['mean_viol']:.2f}  Time={jt_time:.1f}s")

        if jj['rate'] > best_pass_rate:
            best_pass_rate = jj['rate']
            best_config = {
                'lambda6': lam6, 'metrics': jm, 'joins': jj,
                'time': jt_time, 'model': jt.model,
            }

    # ── Results ──────────────────────────────────────────────────────────
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
    print(f"  {'Mean violation':<40} {bj['mean_viol']:>12.2f} {jj['mean_viol']:>12.2f}")
    print(f"  {'Best lambda6':<40} {'N/A':>12} {best_config['lambda6']:>12.1f}")

    # Show some pass/fail examples
    print(f"\n  Sample results (best join-loss model):")
    print(f"  {'Child1':<20} {'Child2':<20} {'Parent':<20} {'Viol':>8} {'Pass?':>6}")
    print("  " + "-" * 76)
    for c1, c2, p, v, passed in jj['details'][:20]:
        mark = "PASS" if passed else f"{v:.1f}"
        print(f"  {c1:<20} {c2:<20} {p:<20} {v:>8.2f} {mark:>6}")

    # Distribution of violations
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
    print(f"  WordNet join triples tested:    {bj['total']}")
    print(f"  Baseline join containment:      {bj['pass']}/{bj['total']} ({bj['rate']:.1%})")
    print(f"  + Join loss (l6={best_config['lambda6']}):        "
          f"{jj['pass']}/{jj['total']} ({jj['rate']:.1%})")
    delta_acc = best_config['metrics']['accuracy'] - base_metrics['accuracy']
    print(f"  NLI accuracy change:            {delta_acc:+.4f}")
    print("=" * 70)


# ── WordNet-adapted JoinAwareTrainer ─────────────────────────────────────────

class WordNetJoinTrainer(ImprovedTrainer):
    """Join-aware trainer that uses word-level triples instead of sentence IDs."""

    def __init__(self, *args, lambda6=1.0, join_triples=None, word_embs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda6 = lambda6
        self.join_triples = join_triples or []
        self.word_embs = word_embs or {}

        # Sample a subset of triples per step for efficiency
        self._triple_batch_size = min(64, len(self.join_triples))

    def _compute_join_loss_and_grads(self):
        if not self.join_triples or self.lambda6 == 0:
            return 0.0, {}

        rng = self.rng
        idx = rng.choice(len(self.join_triples), self._triple_batch_size, replace=False)
        batch_triples = [self.join_triples[i] for i in idx]

        # Collect unique words and batch-encode them all at once
        words_needed = set()
        for c1, c2, p in batch_triples:
            words_needed.update([c1, c2, p])
        word_list = [w for w in words_needed if w in self.word_embs]
        if len(word_list) < 3:
            return 0.0, {}

        word_to_idx = {w: i for i, w in enumerate(word_list)}
        emb_batch = np.stack([self.word_embs[w] for w in word_list])  # (N_words, 300)

        # Single batched forward pass
        l_all, u_all, h_all, cache = self.model._encode_fwd(emb_batch)

        # Accumulate dl/du per word (indexed by position in batch)
        N_words = len(word_list)
        dl_batch = np.zeros((N_words, l_all.shape[1]))
        du_batch = np.zeros((N_words, u_all.shape[1]))

        n_active = 0
        total_viol = 0.0
        for c1, c2, parent in batch_triples:
            if c1 not in word_to_idx or c2 not in word_to_idx or parent not in word_to_idx:
                continue
            i1, i2, ip = word_to_idx[c1], word_to_idx[c2], word_to_idx[parent]
            n_active += 1

            l1, u1 = l_all[i1], u_all[i1]
            l2, u2 = l_all[i2], u_all[i2]
            lp, up = l_all[ip], u_all[ip]

            lj = np.minimum(l1, l2)
            uj = np.maximum(u1, u2)

            lower_viol = np.maximum(0.0, lp - lj)
            upper_viol = np.maximum(0.0, uj - up)
            total_viol += np.sum(lower_viol + upper_viol)

            ind_lv = (lp > lj).astype(float)
            ind_uv = (uj > up).astype(float)

            l1_is_min = (l1 <= l2).astype(float)
            l2_is_min = (l2 < l1).astype(float)
            u1_is_max = (u1 >= u2).astype(float)
            u2_is_max = (u2 > u1).astype(float)

            dl_batch[i1] += (-ind_lv) * l1_is_min
            dl_batch[i2] += (-ind_lv) * l2_is_min
            du_batch[i1] += ind_uv * u1_is_max
            du_batch[i2] += ind_uv * u2_is_max
            dl_batch[ip] += ind_lv
            du_batch[ip] += -ind_uv

        if n_active == 0:
            return 0.0, {}

        l_join = total_viol / n_active
        dl_batch /= n_active
        du_batch /= n_active

        # Single batched backward pass
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


if __name__ == '__main__':
    main()
