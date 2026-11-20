"""
cegar_experiment.py  —  CEGAR Hop-Count Stratification Experiment
=================================================================
Binary decision experiment: does CEGAR provide measurable benefit
on long-range hypernymy (4+ hops) where training signal is sparsest?

Two models, identical except for CEGAR:
  Model A: GloVe + full WordNet, lambda4=0.0 (no CEGAR)
  Model B: GloVe + full WordNet, lambda4=0.3 (with CEGAR)

Evaluation: HyperLex test nouns, stratified by hop count (hyp-1..hyp-4).
Metrics: Kendall tau, Spearman rho, directionality accuracy per bucket.
Bootstrap 95% CIs on delta-tau.

Usage
-----
  PYTHONIOENCODING=utf-8 python -u cegar_experiment.py
"""

import sys, os, time, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import kendalltau, spearmanr
import gensim.downloader as api

from src.improved_trainer import ImprovedTrainer
from src.box_lattice import box_containment_violation

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from eval_hyperlex_embeddings import (
    download_hyperlex, load_hyperlex_splits, filter_nouns,
    galois_scores_from_boxes, get_wv_embedding, DATA_DIR,
    BOX_DIM, HIDDEN_DIM,
)

SEED = 42
TEMPLATE = "A {} is a type of thing."


# ── Full WordNet pairs (reuse logic) ─────────────────────────────────────────

def get_full_wordnet_pairs_with_hops(rng):
    """
    Return ALL WordNet noun hypernymy edges + matched negatives.
    Also computes hop counts for positive pairs.
    Returns (pairs, hop_counts) where pairs = [(w1, w2, label), ...].
    """
    pos_pairs = []
    pos_hops  = []
    seen = set()
    for synset in wn.all_synsets('n'):
        for hypernym in synset.hypernyms():
            w1 = synset.lemmas()[0].name()
            w2 = hypernym.lemmas()[0].name()
            key = (w1, w2)
            if key not in seen:
                seen.add(key)
                pos_pairs.append((w1, w2))
                pos_hops.append(1)  # direct edges are always 1-hop

    n_pos = len(pos_pairs)
    print(f"      Full WordNet: {n_pos} IS-A edges")

    # Negative pairs
    all_nouns = list({s.lemmas()[0].name() for s in wn.all_synsets('n')})
    all_nouns_arr = np.array(all_nouns)
    neg_pairs = []
    while len(neg_pairs) < n_pos:
        need = n_pos - len(neg_pairs)
        batch = rng.choice(len(all_nouns_arr), need * 2)
        w1s = all_nouns_arr[batch[:need]]
        w2s = all_nouns_arr[batch[need:need*2]]
        for w1, w2 in zip(w1s, w2s):
            if w1 != w2 and (w1, w2) not in seen:
                neg_pairs.append((w1, w2))
            if len(neg_pairs) >= n_pos:
                break

    pairs = ([(w1, w2, 0) for w1, w2 in pos_pairs] +
             [(w1, w2, 2) for w1, w2 in neg_pairs[:n_pos]])
    hops  = pos_hops + [None] * len(neg_pairs[:n_pos])
    # Shuffle together
    idx = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in idx]
    hops  = [hops[i]  for i in idx]
    return pairs, hops


# ── Training pipeline ────────────────────────────────────────────────────────

def train_model(emb_lookup, wn_pairs, lambda4, rng, label=""):
    """
    Train a Galois BoxEncoder on full WordNet pairs with given lambda4.
    Returns the trained ImprovedTrainer.
    """
    print(f"\n      Training {label} (lambda4={lambda4})...")

    # Build embeddings
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

    # Train/val split (90/10)
    n_val = max(512, int(len(labs) * 0.1))
    split_rng = np.random.default_rng(SEED)  # same split for both models
    idx = split_rng.permutation(len(labs))
    tr_idx, vl_idx = idx[n_val:], idx[:n_val]

    train_data = {'emb1': emb1[tr_idx], 'emb2': emb2[tr_idx], 'label': labs[tr_idx]}
    val_data   = {'emb1': emb1[vl_idx], 'emb2': emb2[vl_idx], 'label': labs[vl_idx]}
    print(f"      Train: {len(tr_idx)}  Val: {len(vl_idx)}  dim={input_dim}")

    trainer = ImprovedTrainer(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        box_dim=BOX_DIM,
        lr=5e-4,
        lambda1=0.1,
        lambda2=2.0,
        lambda3=1.0,
        lambda4=lambda4,
        lambda5=0.0,
        margin=1.0,
        scale=5.0,
        cegar_margin=1.5,
        warmup_epochs=3,       # let model stabilize before CEGAR
        cegar_max_size=4000,   # larger buffer for full WordNet
        rng=np.random.default_rng(SEED),  # same init for both models
    )
    t0 = time.time()
    trainer.train(train_data, val_data,
                  n_epochs=50, batch_size=256,
                  verbose=True, patience=7)
    print(f"      Done in {time.time()-t0:.1f}s")
    return trainer, train_data, val_data


# ── CEGAR buffer analysis ────────────────────────────────────────────────────

def analyze_cegar_buffer(trainer, emb_lookup, wn_pairs):
    """
    Analyze CEGAR buffer contents by matching buffered box pairs
    back to training word pairs via hop-count proxy.

    Also evaluates the trained model on ALL training pairs to identify
    which pairs are still misclassified (CEGAR-relevant pairs).
    """
    buf = trainer.cegar
    buf_size = len(buf)
    print(f"\n      CEGAR buffer size at training end: {buf_size}")

    if buf_size == 0:
        return {'buf_size': 0, 'mean_hop': None, 'fp_rate': None, 'fn_rate': None}

    # Get buffer contents
    buf_l1  = buf._l1[:buf_size]
    buf_u1  = buf._u1[:buf_size]
    buf_l2  = buf._l2[:buf_size]
    buf_u2  = buf._u2[:buf_size]
    buf_lab = buf._lab[:buf_size]

    # Buffer label distribution
    n_ent = np.sum(buf_lab == 1)     # entailment pairs in buffer (false negatives)
    n_non = np.sum(buf_lab == 0)     # non-entailment in buffer (false positives)
    fp_rate = n_non / buf_size
    fn_rate = n_ent / buf_size
    print(f"      Buffer: {n_ent} false negatives (truth=entail, pred=non-entail)")
    print(f"      Buffer: {n_non} false positives (truth=non-entail, pred=entail)")
    print(f"      FP rate in buffer: {fp_rate:.1%}  FN rate: {fn_rate:.1%}")

    # Post-hoc: evaluate model on ALL training positive pairs to find misclassified ones
    # and compute their hop counts
    print(f"      Computing hop counts for misclassified training pairs...")
    pos_pairs = [(w1, w2) for w1, w2, lab in wn_pairs if lab == 0]  # label 0 = entailment

    misclassified_hops = []
    all_pos_count = 0
    misclass_count = 0

    for w1, w2 in pos_pairs:
        v1 = emb_lookup.get(w1)
        v2 = emb_lookup.get(w2)
        if v1 is None or v2 is None:
            continue

        l1, u1, _ = trainer.model.encode(v1[None, :])
        l2, u2, _ = trainer.model.encode(v2[None, :])
        viol = box_containment_violation(l1, u1, l2, u2)[0]
        p_ent = 1.0 / (1.0 + trainer.scale * viol)

        all_pos_count += 1
        if p_ent < 0.5:  # misclassified: should be entailment but predicted non-entail
            misclass_count += 1
            # Compute hop count for this pair
            hop = get_wordnet_hop(w1, w2)
            if hop is not None:
                misclassified_hops.append(hop)

    print(f"      Positive pairs evaluated: {all_pos_count}")
    print(f"      Misclassified (FN): {misclass_count} ({misclass_count/max(all_pos_count,1):.1%})")

    if misclassified_hops:
        mean_hop = np.mean(misclassified_hops)
        print(f"      Mean hop count of misclassified: {mean_hop:.2f}")
        hop_dist = np.bincount(misclassified_hops, minlength=6)
        for h in range(1, min(6, len(hop_dist))):
            if hop_dist[h] > 0:
                print(f"        hop={h}: {hop_dist[h]} pairs")
    else:
        mean_hop = None
        print(f"      No misclassified positive pairs found")

    return {
        'buf_size': buf_size,
        'fp_rate':  fp_rate,
        'fn_rate':  fn_rate,
        'mean_hop': float(mean_hop) if mean_hop is not None else None,
        'misclass_hops': misclassified_hops,
        'n_misclassified': misclass_count,
        'n_pos_total': all_pos_count,
    }


def get_wordnet_hop(w1, w2):
    """
    Compute minimum hop count from w1 to w2 in WordNet hypernymy graph.
    w1 should be the hyponym, w2 the hypernym.
    Returns int or None.
    """
    hypo_synsets  = wn.synsets(w1.replace(' ', '_'), pos='n')
    hyper_synsets = wn.synsets(w2.replace(' ', '_'), pos='n')
    if not hypo_synsets or not hyper_synsets:
        return None

    min_hops = None
    for hypo in hypo_synsets:
        for hyper in hyper_synsets:
            paths = hypo.hypernym_paths()
            for path in paths:
                if hyper in path:
                    hypo_idx  = path.index(hypo)
                    hyper_idx = path.index(hyper)
                    if hyper_idx > hypo_idx:
                        hops = hyper_idx - hypo_idx
                        if min_hops is None or hops < min_hops:
                            min_hops = hops
    return min_hops


# ── Evaluation per hop bucket ────────────────────────────────────────────────

def evaluate_per_bucket(trainer, pairs, gold, emb_lookup, hop_map):
    """
    Evaluate a trained model on HyperLex test pairs, stratified by hop count.

    pairs: list of dicts with 'word1', 'word2', 'type', 'score_10'
    gold: numpy array of gold scores (0-10)
    hop_map: maps pair type to hop bucket ('1', '2', '3', '4+')
    emb_lookup: word -> embedding

    Returns: per-bucket results dict.
    """
    # Encode all test pairs
    scores_fwd = []
    vf_list = []
    vb_list = []
    valid_pairs = []
    valid_gold = []
    valid_types = []

    for p, g in zip(pairs, gold):
        v1 = emb_lookup.get(p['word1'])
        v2 = emb_lookup.get(p['word2'])
        if v1 is None or v2 is None:
            continue

        l1, u1, _ = trainer.model.encode(v1[None, :])
        l2, u2, _ = trainer.model.encode(v2[None, :])
        vf = box_containment_violation(l1, u1, l2, u2)[0]
        vb = box_containment_violation(l2, u2, l1, u1)[0]

        sf = 1.0 / (1.0 + vf)
        scores_fwd.append(sf)
        vf_list.append(vf)
        vb_list.append(vb)
        valid_pairs.append(p)
        valid_gold.append(g)
        valid_types.append(p['type'])

    scores_fwd = np.array(scores_fwd)
    vf_arr = np.array(vf_list)
    vb_arr = np.array(vb_list)
    valid_gold = np.array(valid_gold)

    # Identify positive (hyp-*) and negative pairs
    is_positive = np.array([t.startswith('hyp-') for t in valid_types])
    is_negative = np.array([not t.startswith('hyp-') and not t.startswith('r-hyp-')
                            for t in valid_types])

    # Build hop buckets for positive pairs
    hop_labels = []
    for t in valid_types:
        if t in hop_map:
            hop_labels.append(hop_map[t])
        else:
            hop_labels.append(None)
    hop_labels = np.array(hop_labels, dtype=object)

    results = {}
    neg_idx = np.where(is_negative)[0]

    for bucket in ['1', '2', '3', '4+']:
        # Positive pairs in this bucket
        pos_idx = np.array([i for i in range(len(valid_types))
                            if hop_labels[i] == bucket])

        if len(pos_idx) < 5:
            results[bucket] = None
            continue

        # For tau/spearman: combine positive pairs (label=1) + negative pairs (label=0)
        combined_idx = np.concatenate([pos_idx, neg_idx])
        combined_scores = scores_fwd[combined_idx]
        combined_labels = np.concatenate([
            np.ones(len(pos_idx)),
            np.zeros(len(neg_idx))
        ])

        tau, p_tau = kendalltau(combined_scores, combined_labels)
        rho, p_rho = spearmanr(combined_scores, combined_labels)

        # Directionality: for positive pairs only
        dir_correct = 0
        dir_total = 0
        for i in pos_idx:
            dir_total += 1
            if vf_arr[i] < vb_arr[i]:
                dir_correct += 1
        dir_acc = dir_correct / max(dir_total, 1)

        # Also compute Spearman on graded scores (positive pairs only vs gold)
        sp_graded, _ = spearmanr(scores_fwd[pos_idx], valid_gold[pos_idx])

        results[bucket] = {
            'n_pos':     len(pos_idx),
            'n_neg':     len(neg_idx),
            'tau':       tau,
            'p_tau':     p_tau,
            'rho':       rho,
            'dir_acc':   dir_acc,
            'sp_graded': sp_graded,
            'scores':    combined_scores,
            'labels':    combined_labels,
        }

    # Overall
    all_pos_idx = np.where(is_positive)[0]
    combined_idx = np.concatenate([all_pos_idx, neg_idx])
    combined_scores = scores_fwd[combined_idx]
    combined_labels = np.concatenate([
        np.ones(len(all_pos_idx)),
        np.zeros(len(neg_idx))
    ])
    tau_all, _ = kendalltau(combined_scores, combined_labels)
    rho_all, _ = spearmanr(combined_scores, combined_labels)
    dir_all = np.sum(vf_arr[all_pos_idx] < vb_arr[all_pos_idx]) / max(len(all_pos_idx), 1)

    results['ALL'] = {
        'n_pos':   len(all_pos_idx),
        'n_neg':   len(neg_idx),
        'tau':     tau_all,
        'rho':     rho_all,
        'dir_acc': dir_all,
        'scores':  combined_scores,
        'labels':  combined_labels,
    }

    return results


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_delta_tau(scores_nc, scores_c, labels, n_bootstrap=1000, rng=None):
    """
    Bootstrap 95% CI on Kendall's tau difference (CEGAR - no_CEGAR).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    n = len(labels)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        s_nc = scores_nc[idx]
        s_c  = scores_c[idx]
        lab  = labels[idx]
        tau_nc, _ = kendalltau(s_nc, lab)
        tau_c,  _ = kendalltau(s_c,  lab)
        deltas.append(tau_c - tau_nc)

    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*80}")
    print(f"  CEGAR Hop-Count Stratification Experiment")
    print(f"{'='*80}")
    print(f"  Input: GloVe-wiki-gigaword-300")
    print(f"  Training: Full WordNet noun hypernymy (~74k edges)")
    print(f"  Model A: lambda4=0.0 (no CEGAR)")
    print(f"  Model B: lambda4=0.3 (with CEGAR, warmup=3, buf=4000)")
    print(f"  Seed: {SEED}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data...")
    download_hyperlex()
    _, _, test_pairs = load_hyperlex_splits()
    te_n = filter_nouns(test_pairs)
    gold_te = np.array([p['score_10'] for p in te_n])

    # HyperLex type → hop bucket mapping
    hop_map = {
        'hyp-1': '1', 'hyp-2': '2', 'hyp-3': '3', 'hyp-4': '4+',
    }

    # Count bucket sizes
    print(f"\n      HyperLex test noun bucket sizes:")
    for bucket in ['1', '2', '3', '4+']:
        types_for_bucket = [k for k, v in hop_map.items() if v == bucket]
        n = sum(1 for p in te_n if p['type'] in types_for_bucket)
        print(f"        Hop {bucket}: {n} positive pairs")
    n_neg = sum(1 for p in te_n
                if not p['type'].startswith('hyp-') and not p['type'].startswith('r-hyp-'))
    print(f"        Negatives (cohyp/no-rel/syn/ant/mero): {n_neg}")

    # ── 2. Load GloVe ─────────────────────────────────────────────────────
    print(f"\n[2/6] Loading GloVe...")
    t0 = time.time()
    glove = api.load('glove-wiki-gigaword-300')
    print(f"      Loaded in {time.time()-t0:.0f}s  vocab={len(glove)}")

    # Build GloVe lookup for all needed words
    print(f"      Building GloVe lookup for WordNet + HyperLex words...")
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

    # ── 3. Generate training data ─────────────────────────────────────────
    print(f"\n[3/6] Generating full WordNet training pairs...")
    rng = np.random.default_rng(SEED)
    wn_pairs, wn_hops = get_full_wordnet_pairs_with_hops(rng)

    # ── 4. Train Model A (no CEGAR) ──────────────────────────────────────
    print(f"\n[4/6] Training Model A (no CEGAR)...")
    trainer_a, _, _ = train_model(emb_lookup, wn_pairs, lambda4=0.0,
                                   rng=np.random.default_rng(SEED),
                                   label="Model A (no CEGAR)")

    # ── 5. Train Model B (with CEGAR) ────────────────────────────────────
    print(f"\n[5/6] Training Model B (with CEGAR)...")
    trainer_b, _, _ = train_model(emb_lookup, wn_pairs, lambda4=0.3,
                                   rng=np.random.default_rng(SEED),
                                   label="Model B (CEGAR)")

    # ── CEGAR buffer analysis ─────────────────────────────────────────────
    print(f"\n      Analyzing CEGAR buffer...")
    buf_stats = analyze_cegar_buffer(trainer_b, emb_lookup, wn_pairs)

    # ── 6. Evaluate both models per hop bucket ────────────────────────────
    print(f"\n[6/6] Evaluating on HyperLex test (stratified by hop)...")
    results_a = evaluate_per_bucket(trainer_a, te_n, gold_te, emb_lookup, hop_map)
    results_b = evaluate_per_bucket(trainer_b, te_n, gold_te, emb_lookup, hop_map)

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    print(f"\n      Computing bootstrap CIs (1000 resamples per bucket)...")
    bootstrap_rng = np.random.default_rng(999)
    ci_results = {}
    for bucket in ['1', '2', '3', '4+', 'ALL']:
        ra = results_a.get(bucket)
        rb = results_b.get(bucket)
        if ra is None or rb is None:
            ci_results[bucket] = None
            continue
        mean_d, ci_lo, ci_hi = bootstrap_delta_tau(
            ra['scores'], rb['scores'], ra['labels'],
            n_bootstrap=1000, rng=bootstrap_rng)
        ci_results[bucket] = {
            'mean_delta': mean_d,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'sig': ci_lo > 0 or ci_hi < 0,  # CI excludes zero
        }

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*90}")
    print(f"  CEGAR Hop-Count Stratification Results (GloVe input, full WordNet training)")
    print(f"{'='*90}")

    hdr = (f"  {'Hop':>4}  {'N_pos':>6}  "
           f"{'tau_nc':>8}  {'tau_c':>8}  {'d_tau':>8}  "
           f"{'Dir_nc':>8}  {'Dir_c':>8}  {'dDir':>8}  "
           f"{'95% CI':>20}")
    print(f"\n{hdr}")
    print(f"  {'-'*86}")

    for bucket in ['1', '2', '3', '4+', 'ALL']:
        ra = results_a.get(bucket)
        rb = results_b.get(bucket)
        ci = ci_results.get(bucket)

        if ra is None or rb is None:
            print(f"  {bucket:>4}  (insufficient data)")
            continue

        d_tau = rb['tau'] - ra['tau']
        d_dir = (rb['dir_acc'] - ra['dir_acc']) * 100

        ci_str = ""
        if ci is not None:
            sig = "*" if ci['sig'] else ""
            ci_str = f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]{sig}"

        if bucket == 'ALL':
            print(f"  {'-'*86}")

        print(f"  {bucket:>4}  {ra['n_pos']:>6}  "
              f"{ra['tau']:>+8.4f}  {rb['tau']:>+8.4f}  {d_tau:>+8.4f}  "
              f"{ra['dir_acc']:>7.1%}  {rb['dir_acc']:>7.1%}  {d_dir:>+7.1f}%  "
              f"{ci_str:>20}")

    # ── CEGAR buffer statistics ───────────────────────────────────────────
    print(f"\n  CEGAR buffer statistics:")
    print(f"    Buffer size at training end:       {buf_stats['buf_size']} pairs")
    print(f"    False positive rate in buffer:     {buf_stats['fp_rate']:.1%}" if buf_stats['fp_rate'] is not None else "    False positive rate in buffer:     N/A")
    print(f"    False negative rate in buffer:     {buf_stats['fn_rate']:.1%}" if buf_stats['fn_rate'] is not None else "    False negative rate in buffer:     N/A")
    if buf_stats['mean_hop'] is not None:
        print(f"    Mean hop count of misclassified:  {buf_stats['mean_hop']:.2f}")
    else:
        print(f"    Mean hop count of misclassified:  N/A")

    # ── Decision criteria ─────────────────────────────────────────────────
    r4 = results_a.get('4+'), results_b.get('4+'), ci_results.get('4+')
    r1 = results_a.get('1'), results_b.get('1'), ci_results.get('1')
    r_all_a = results_a.get('ALL')
    r_all_b = results_b.get('ALL')

    # Extract key numbers
    if r4[0] is not None and r4[1] is not None:
        d_tau_4 = r4[1]['tau'] - r4[0]['tau']
    else:
        d_tau_4 = 0.0

    if r1[0] is not None and r1[1] is not None:
        d_tau_1 = r1[1]['tau'] - r1[0]['tau']
    else:
        d_tau_1 = 0.0

    d_tau_all = r_all_b['tau'] - r_all_a['tau'] if r_all_a and r_all_b else 0.0
    mean_hop_buf = buf_stats['mean_hop']
    ci_4 = ci_results.get('4+')

    print(f"\n  Decision criteria:")
    c1 = d_tau_4 > 0.02
    c2 = d_tau_4 > d_tau_1
    c3 = mean_hop_buf is not None and mean_hop_buf > 2.5
    c4 = d_tau_all >= 0
    c_not1 = d_tau_4 < 0.01
    c_not3 = d_tau_all < -0.01

    print(f"    [{'PASS' if c1 else 'FAIL'}] d_tau(4+) = {d_tau_4:+.4f} > +0.02")
    print(f"    [{'PASS' if c2 else 'FAIL'}] d_tau(4+) = {d_tau_4:+.4f} > d_tau(1) = {d_tau_1:+.4f}")
    print(f"    [{'PASS' if c3 else 'FAIL'}] Mean hop of buffer = {mean_hop_buf if mean_hop_buf else 'N/A'} > 2.5")
    print(f"    [{'PASS' if c4 else 'FAIL'}] d_tau(ALL) = {d_tau_all:+.4f} >= 0")

    validated = c1 and c2 and c3 and c4
    not_validated = c_not1 or (abs(d_tau_4 - d_tau_1) < 0.005) or c_not3

    # ── VERDICT ───────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    if validated:
        ci_str = f"[{ci_4['ci_lo']:+.4f}, {ci_4['ci_hi']:+.4f}]" if ci_4 else "N/A"
        print(f"""
VERDICT A -- CEGAR VALIDATED:

CEGAR is validated as a contribution.
Evidence: d_tau = {d_tau_4:+.4f} {ci_str} on 4+ hop pairs.
Interpretation: CEGAR provides {'statistically significant ' if ci_4 and ci_4['sig'] else ''}improvement
on long-range hypernymy pairs where training signal is sparsest,
consistent with the theoretical prediction.
Action: CEGAR remains a primary claimed contribution in the paper.
Include the hop-stratification table as Table 3 in Section 4.2.
""")
    else:
        ci_str = f"[{ci_4['ci_lo']:+.4f}, {ci_4['ci_hi']:+.4f}]" if ci_4 else "N/A"
        print(f"""
VERDICT B -- CEGAR NOT VALIDATED:

CEGAR is not validated as a contribution on this benchmark.
Evidence: d_tau = {d_tau_4:+.4f} {ci_str} on 4+ hop pairs
         {'(confidence interval includes zero)' if ci_4 and not ci_4['sig'] else ''}.
Interpretation: CEGAR does not provide statistically significant
improvement on long-range pairs. {f'The buffer mean hop count of {mean_hop_buf:.2f}' if mean_hop_buf else 'The buffer analysis'}
suggests counterexamples are not concentrated on hard long-range pairs.
Action: Remove CEGAR from claimed contributions.
Move to Section 6 with the following text verbatim:

"CEGAR is designed for settings where the model makes systematic
errors on underrepresented relationship types. On binary WordNet
classification with random negatives, counterexamples are not
sufficiently structured to drive refinement. We expect CEGAR to
show benefit with hard negative sampling and graded training
objectives, which we leave to future work."
""")
    print(f"{'='*90}")

    # ── Save ──────────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    save_data = {
        'seed': SEED,
        'verdict': 'VALIDATED' if validated else 'NOT_VALIDATED',
        'delta_tau_4plus': d_tau_4,
        'delta_tau_1': d_tau_1,
        'delta_tau_all': d_tau_all,
        'mean_hop_buffer': mean_hop_buf,
        'ci_4plus': ci_results.get('4+'),
        'buf_stats': {k: v for k, v in buf_stats.items() if k != 'misclass_hops'},
    }
    with open('results/cegar_experiment.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"  Results saved to results/cegar_experiment.json")


if __name__ == '__main__':
    main()
