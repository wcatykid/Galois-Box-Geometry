"""
eval_hyperlex_fullwn.py  —  HyperLex with FULL WordNet graph (no sampling)
===========================================================================
Same as eval_hyperlex_embeddings.py but uses ALL ~74k WordNet noun hypernymy
edges instead of sampling 25k.  This brings training conditions closer to the
literature (Vendrov, Vilnis).

Expected improvement: +0.05–0.10 Spearman from more training data alone.

Usage
-----
  PYTHONIOENCODING=utf-8 python -u eval_hyperlex_fullwn.py
"""

import sys, os, time, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
import gensim.downloader as api

from src.improved_trainer import ImprovedTrainer

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

# Reuse functions from the existing scripts
from eval_hyperlex_embeddings import (
    download_hyperlex, load_hyperlex_splits, filter_nouns,
    galois_scores_from_boxes, cosine_score, proj_score,
    evaluate_hyperlex, directionality_test, proj_directionality,
    get_wv_embedding, DATA_DIR, DIR_SCORE_THRESH,
    BOX_DIM, HIDDEN_DIM,
)
from eval_hyperlex import (
    load_v2_model, encode_words_bert, encode_words_boxes,
    V2_CKPT,
)

# Config
WN_EPOCHS   = 50
WN_PATIENCE = 7
WN_BATCH    = 256

TEMPLATE = "A {} is a type of thing."    # best from dev-set selection


# ── Full WordNet pairs (no sampling) ─────────────────────────────────────────

def get_full_wordnet_pairs(rng=None):
    """
    Return ALL WordNet noun hypernymy edges + matched random negatives.
    No sampling — uses the full graph.
    """
    if rng is None:
        rng = np.random.default_rng(42)

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

    n_pos = len(pos_pairs)
    print(f"      Full WordNet: {n_pos} IS-A edges (no sampling)")

    # Negative pairs: matched count
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
    rng.shuffle(pairs)
    print(f"      Positive: {n_pos}  Negative: {len(neg_pairs[:n_pos])}  "
          f"Total: {len(pairs)}")
    return pairs


# ── Training pipeline for one embedding ──────────────────────────────────────

def train_and_eval(emb_name, lookup_fn, pairs_te, gold_te, rng):
    """Train Galois on full WordNet, evaluate on HyperLex test."""
    print(f"\n  {'='*60}")
    print(f"  {emb_name}  —  Full WordNet Training")
    print(f"  {'='*60}")

    # Encode test pairs
    e1_list, e2_list, valid_pairs, valid_gold = [], [], [], []
    oov_count = 0
    for p, g in zip(pairs_te, gold_te):
        v1 = lookup_fn(p['word1'])
        v2 = lookup_fn(p['word2'])
        if v1 is None or v2 is None:
            oov_count += 1
            continue
        e1_list.append(v1)
        e2_list.append(v2)
        valid_pairs.append(p)
        valid_gold.append(g)

    E1 = np.stack(e1_list)
    E2 = np.stack(e2_list)
    gold = np.array(valid_gold)
    print(f"      Test pairs: {len(valid_pairs)}/{len(pairs_te)}  OOV: {oov_count}")

    # Baselines (quick)
    cos = cosine_score(E1, E2)
    m_cos = evaluate_hyperlex(cos, gold)
    prj = proj_score(E1, E2)
    m_prj = evaluate_hyperlex(prj, gold)
    prj_dir, n_dir = proj_directionality(E1, E2, valid_pairs)
    print(f"      Cosine     Sp={m_cos['spearman']:+.4f}  Dir=50.0%")
    print(f"      Projection Sp={m_prj['spearman']:+.4f}  Dir={prj_dir:.1%}")

    # Get full WordNet pairs
    wn_pairs = get_full_wordnet_pairs(rng=rng)

    # Build embeddings for all WordNet words
    all_wn_words = list({w for (w1, w2, _) in wn_pairs for w in (w1, w2)})
    wn_lookup = {}
    wn_oov = 0
    t0 = time.time()
    for w in all_wn_words:
        v = lookup_fn(w)
        if v is not None:
            wn_lookup[w] = v
        else:
            wn_oov += 1
    enc_time = time.time() - t0
    print(f"      WordNet words: {len(all_wn_words)}  OOV: {wn_oov} ({wn_oov/len(all_wn_words):.1%})  "
          f"Encoding: {enc_time:.1f}s")

    # Build training arrays
    emb1_list, emb2_list, lab_list = [], [], []
    for w1, w2, lab in wn_pairs:
        if w1 in wn_lookup and w2 in wn_lookup:
            emb1_list.append(wn_lookup[w1])
            emb2_list.append(wn_lookup[w2])
            lab_list.append(lab)

    emb1_wn = np.stack(emb1_list)
    emb2_wn = np.stack(emb2_list)
    labs_wn = np.array(lab_list)
    input_dim = emb1_wn.shape[1]

    # Train/val split (90/10)
    n_val = max(512, int(len(labs_wn) * 0.1))
    idx = rng.permutation(len(labs_wn))
    tr_idx, vl_idx = idx[n_val:], idx[:n_val]

    train_data = {'emb1': emb1_wn[tr_idx], 'emb2': emb2_wn[tr_idx], 'label': labs_wn[tr_idx]}
    val_data   = {'emb1': emb1_wn[vl_idx], 'emb2': emb2_wn[vl_idx], 'label': labs_wn[vl_idx]}
    print(f"      Train: {len(tr_idx)}  Val: {len(vl_idx)}  dim={input_dim}")

    trainer = ImprovedTrainer(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        box_dim=BOX_DIM,
        lr=5e-4,
        lambda1=0.1,
        lambda2=2.0,
        lambda3=1.0,
        lambda4=0.0,
        lambda5=0.0,
        margin=1.0,
        scale=5.0,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
    )
    t0 = time.time()
    trainer.train(train_data, val_data,
                  n_epochs=WN_EPOCHS, batch_size=WN_BATCH,
                  verbose=True, patience=WN_PATIENCE)
    train_time = time.time() - t0
    print(f"      Training done in {train_time:.1f}s")

    # Evaluate on HyperLex
    l1, u1, _ = trainer.model.encode(E1)
    l2, u2, _ = trainer.model.encode(E2)
    sf, sa, vf, vb = galois_scores_from_boxes(l1, u1, l2, u2)

    m_fwd  = evaluate_hyperlex(sf, gold)
    m_asym = evaluate_hyperlex(sa, gold)
    gal_dir, _ = directionality_test(l1, u1, l2, u2, valid_pairs)

    best_sp = max(m_fwd['spearman'], m_asym['spearman'])
    best_pe = m_fwd['pearson'] if m_fwd['spearman'] >= m_asym['spearman'] else m_asym['pearson']
    best_name = 'fwd' if m_fwd['spearman'] >= m_asym['spearman'] else 'asym'

    print(f"\n      RESULT ({emb_name}):")
    print(f"        score_{best_name}  Sp={best_sp:+.4f}  Pe={best_pe:+.4f}  Dir={gal_dir:.1%}")
    print(f"        (fwd  Sp={m_fwd['spearman']:+.4f}  asym Sp={m_asym['spearman']:+.4f})")

    # Violation geometry
    print(f"\n      Violation geometry:")
    print(f"      {'type':>10}  {'N':>5}  {'vf_p50':>8}  {'vb_p50':>8}  {'asym_p50':>9}")
    for rel in ['hyp-1', 'hyp-2', 'hyp-3', 'cohyp', 'no-rel', 'syn']:
        idx_r = [i for i, p in enumerate(valid_pairs) if p['type'] == rel]
        if len(idx_r) < 5:
            continue
        print(f"      {rel:>10}  {len(idx_r):>5}  {np.median(vf[idx_r]):>8.2f}  "
              f"{np.median(vb[idx_r]):>8.2f}  {np.median(vb[idx_r]-vf[idx_r]):>+9.2f}")

    return {
        'emb_name': emb_name,
        'n_test': len(valid_pairs),
        'oov_test': oov_count,
        'oov_wn': wn_oov,
        'n_train': len(tr_idx),
        'galois_sp_fwd': m_fwd['spearman'],
        'galois_sp_asym': m_asym['spearman'],
        'galois_sp': best_sp,
        'galois_pe': best_pe,
        'galois_dir': gal_dir,
        'cosine_sp': m_cos['spearman'],
        'proj_sp': m_prj['spearman'],
        'proj_dir': prj_dir,
        'train_time': train_time,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*70}")
    print(f"  HyperLex  x  FULL WordNet Graph  (74k+ edges, no sampling)")
    print(f"{'='*70}")

    # Data
    print(f"\n[1/5] Loading HyperLex...")
    download_hyperlex()
    _, _, test_pairs = load_hyperlex_splits()
    te_n = filter_nouns(test_pairs)
    gold_te = np.array([p['score_10'] for p in te_n])
    print(f"      Test nouns: {len(te_n)}")

    # Load embedding models
    print(f"\n[2/5] Loading embedding models...")

    t0 = time.time()
    print(f"      Loading GloVe-wiki-gigaword-300...")
    glove = api.load('glove-wiki-gigaword-300')
    print(f"      Loaded in {time.time()-t0:.0f}s  vocab={len(glove)}")

    t0 = time.time()
    print(f"      Loading Word2Vec-google-news-300...")
    w2v = api.load('word2vec-google-news-300')
    print(f"      Loaded in {time.time()-t0:.0f}s  vocab={len(w2v)}")

    print(f"      Loading v2 fine-tuned BERT for SBERT embeddings...")
    v2_model, v2_tok, v2_cfg = load_v2_model()

    # Define lookup functions
    def glove_lookup(word):
        return get_wv_embedding(word, glove)

    def w2v_lookup(word):
        return get_wv_embedding(word, w2v)

    # For SBERT: pre-encode all needed words (HyperLex test + full WordNet)
    # Build SBERT lookup lazily via a cache dict
    sbert_cache = {}

    def sbert_lookup(word):
        if word in sbert_cache:
            return sbert_cache[word]
        return None   # will be populated in bulk below

    # Pre-encode all words needed for SBERT
    print(f"\n      Pre-encoding ALL words for SBERT (test + WordNet)...")
    # Gather all words we'll need
    test_words = list({p['word1'] for p in te_n} | {p['word2'] for p in te_n})

    # Get all WordNet words too
    wn_words = set()
    for synset in wn.all_synsets('n'):
        wn_words.add(synset.lemmas()[0].name())
        for h in synset.hypernyms():
            wn_words.add(h.lemmas()[0].name())

    all_sbert_words = list(set(test_words) | wn_words)
    print(f"      Total unique words to encode: {len(all_sbert_words)}")

    t0 = time.time()
    # Encode with template, replacing underscores
    texts = [TEMPLATE.format(w.replace('_', ' ')) for w in all_sbert_words]
    embs = encode_words_bert(v2_model, v2_tok,
                              [w.replace('_', ' ') for w in all_sbert_words],
                              TEMPLATE, max_len=v2_cfg.max_len, device='cpu')
    for i, w in enumerate(all_sbert_words):
        sbert_cache[w] = embs[i]
    print(f"      SBERT encoding done in {time.time()-t0:.1f}s")

    # Now redefine sbert_lookup to use the cache
    def sbert_lookup_cached(word):
        return sbert_cache.get(word, None)

    # Run evaluations
    print(f"\n[3/5] GloVe — Full WordNet")
    r_glove = train_and_eval('GloVe-300', glove_lookup, te_n, gold_te,
                              rng=np.random.default_rng(42))

    print(f"\n[4/5] Word2Vec — Full WordNet")
    r_w2v = train_and_eval('Word2Vec-300', w2v_lookup, te_n, gold_te,
                            rng=np.random.default_rng(42))

    print(f"\n[5/5] SBERT (v2 BERT) — Full WordNet")
    r_sbert = train_and_eval('SBERT-384', sbert_lookup_cached, te_n, gold_te,
                              rng=np.random.default_rng(42))

    # ── Combined results table ────────────────────────────────────────────
    # Previous 25k results for comparison
    prev = {
        'GloVe':   {'sp': 0.4482, 'dir': 0.844},
        'W2V':     {'sp': 0.4755, 'dir': 0.865},
        'SBERT':   {'sp': 0.4719, 'dir': 0.881},
    }

    print(f"\n\n{'='*80}")
    print(f"  HyperLex  —  Full WordNet vs 25k Sampled  —  Comparison")
    print(f"{'='*80}")

    print(f"\n  {'Model':<45} {'Input':<10} {'Spearman':>8} {'Dir%':>6}  {'delta_sp':>9}")
    print(f"  {'-'*82}")

    # Literature
    print(f"  {'Vendrov Order Emb (2016)':<45} {'GloVe':<10} {'~0.720':>8} {'~65%':>6}")
    print(f"  {'Vilnis Box Emb (2018)':<45} {'GloVe':<10} {'~0.740':>8} {'  --':>6}")
    print(f"  {'LEAR (Nickel 2017)':<45} {'GloVe':<10} {'~0.765':>8} {'  --':>6}")
    print(f"  {'-'*82}")

    # 25k results
    print(f"  {'Galois BoxEncoder (25k WN sampled)':<45} {'GloVe':<10} {prev['GloVe']['sp']:>+8.4f} {prev['GloVe']['dir']:>5.1%}")
    print(f"  {'Galois BoxEncoder (25k WN sampled)':<45} {'Word2Vec':<10} {prev['W2V']['sp']:>+8.4f} {prev['W2V']['dir']:>5.1%}")
    print(f"  {'Galois BoxEncoder (25k WN sampled)':<45} {'SBERT':<10} {prev['SBERT']['sp']:>+8.4f} {prev['SBERT']['dir']:>5.1%}")
    print(f"  {'-'*82}")

    # Full WN results
    d_gl = r_glove['galois_sp'] - prev['GloVe']['sp']
    d_wv = r_w2v['galois_sp']   - prev['W2V']['sp']
    d_sb = r_sbert['galois_sp'] - prev['SBERT']['sp']

    print(f"  {'Galois BoxEncoder (FULL WN ~74k)':<45} {'GloVe':<10} {r_glove['galois_sp']:>+8.4f} {r_glove['galois_dir']:>5.1%}  {d_gl:>+9.4f}")
    print(f"  {'Galois BoxEncoder (FULL WN ~74k)':<45} {'Word2Vec':<10} {r_w2v['galois_sp']:>+8.4f} {r_w2v['galois_dir']:>5.1%}  {d_wv:>+9.4f}")
    print(f"  {'Galois BoxEncoder (FULL WN ~74k)':<45} {'SBERT':<10} {r_sbert['galois_sp']:>+8.4f} {r_sbert['galois_dir']:>5.1%}  {d_sb:>+9.4f}")

    print(f"\n  Average Spearman improvement from full WN: "
          f"{np.mean([d_gl, d_wv, d_sb]):+.4f}")
    print(f"  Average directionality (full WN): "
          f"{np.mean([r_glove['galois_dir'], r_w2v['galois_dir'], r_sbert['galois_dir']]):.1%}")

    # OOV
    print(f"\n  WordNet training OOV rates:")
    print(f"    GloVe:    {r_glove['oov_wn']} words")
    print(f"    Word2Vec: {r_w2v['oov_wn']} words")
    print(f"    SBERT:    {r_sbert['oov_wn']} words")

    # Training sizes
    print(f"\n  Training sizes (after OOV removal):")
    print(f"    GloVe:    {r_glove['n_train']} pairs")
    print(f"    Word2Vec: {r_w2v['n_train']} pairs")
    print(f"    SBERT:    {r_sbert['n_train']} pairs")

    # Save
    Path('results').mkdir(exist_ok=True)
    out = {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
               for kk, vv in v.items()}
           for k, v in {'glove': r_glove, 'w2v': r_w2v, 'sbert': r_sbert}.items()}
    with open('results/hyperlex_fullwn_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to results/hyperlex_fullwn_results.json")


if __name__ == '__main__':
    main()
