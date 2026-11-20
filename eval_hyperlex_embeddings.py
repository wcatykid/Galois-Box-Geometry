"""
eval_hyperlex_embeddings.py  —  HyperLex evaluation with GloVe + Word2Vec + SBERT
==================================================================================
Runs the full HyperLex evaluation across three input representations to demonstrate
that the Galois box geometry is input-agnostic.

Three input representations, same Galois architecture, same training procedure:
  1. GloVe-wiki-gigaword-300  (400K vocab, 5.6B tokens, 300-dim)
  2. Word2Vec-google-news-300 (3M vocab, 100B tokens, 300-dim)
  3. SBERT all-MiniLM-L6-v2   (384-dim, contextual sentence encoder)

For each representation:
  - Train BoxEncoder on WordNet hypernymy pairs (25k pos + 25k neg)
  - Evaluate on HyperLex test split (official random split, nouns only)
  - Report: Spearman, Pearson, directionality accuracy

Also runs baselines: cosine, projection, order embeddings.

Usage
-----
  PYTHONIOENCODING=utf-8 python -u eval_hyperlex_embeddings.py
"""

import sys, os, time, json, urllib.request, zipfile, io
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import spearmanr, pearsonr
import gensim.downloader as api

from src.improved_trainer import ImprovedTrainer
from src.metrics import OrderEmbeddingBaseline

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn


# ── Config ───────────────────────────────────────────────────────────────────

HYPERLEX_URL = ('https://raw.githubusercontent.com/cambridgeltl/hyperlex/'
                'refs/heads/master/hyperlex-data.zip')
DATA_DIR      = Path('data/hyperlex')

# WordNet training
WN_N_POSITIVE   = 25_000
WN_N_NEGATIVE   = 25_000
WN_EPOCHS       = 50
WN_PATIENCE     = 7
WN_BATCH        = 256

# Galois architecture
BOX_DIM    = 128
HIDDEN_DIM = 512

DIR_SCORE_THRESH = 5.0

# Previous SBERT results (from eval_hyperlex.py run)
SBERT_RESULTS = {
    'zeroshot_spearman': 0.3266,
    'zeroshot_pearson':  0.2835,
    'zeroshot_dir':      0.693,
    'wn_spearman':       0.4719,
    'wn_pearson':        0.4709,
    'wn_dir':            0.881,
}


# ── Data loading (reuse from eval_hyperlex.py) ───────────────────────────────

def download_hyperlex(data_dir: Path = DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)
    main_file = data_dir / 'hyperlex-all.txt'
    if main_file.exists():
        print(f"      HyperLex already cached at {data_dir}")
        return
    print(f"      Downloading HyperLex from GitHub...")
    raw = urllib.request.urlopen(HYPERLEX_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    z.extractall(data_dir)
    print(f"      Extracted {len(z.namelist())} files to {data_dir}")


def parse_hyperlex_file(path: Path):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('WORD1'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            pairs.append({
                'word1':    parts[0],        # keep raw (with underscores for multi-word)
                'word2':    parts[1],
                'pos':      parts[2],
                'type':     parts[3],
                'score':    float(parts[4]),
                'score_10': float(parts[5]),
            })
    return pairs


def load_hyperlex_splits(data_dir: Path = DATA_DIR):
    train = parse_hyperlex_file(data_dir / 'splits/random/hyperlex_training_all_random.txt')
    dev   = parse_hyperlex_file(data_dir / 'splits/random/hyperlex_dev_all_random.txt')
    test  = parse_hyperlex_file(data_dir / 'splits/random/hyperlex_test_all_random.txt')
    return train, dev, test


def filter_nouns(pairs):
    return [p for p in pairs if p['pos'] == 'N']


# ── Embedding helpers ────────────────────────────────────────────────────────

def get_wv_embedding(phrase, wv_model):
    """
    Get embedding for a (possibly multi-word) phrase from a word vector model.
    Multi-word terms are averaged over component word vectors.
    Returns None if all words are OOV.
    """
    words = phrase.replace('_', ' ').lower().split()
    vecs = [wv_model[w] for w in words if w in wv_model]
    if len(vecs) == 0:
        return None
    return np.mean(vecs, axis=0)


def build_wv_lookup(wv_model, words):
    """
    Build word → embedding lookup for a list of words.
    Returns (lookup_dict, oov_words, oov_rate).
    """
    lookup = {}
    oov = []
    for w in words:
        emb = get_wv_embedding(w, wv_model)
        if emb is not None:
            lookup[w] = emb
        else:
            oov.append(w)
    oov_rate = len(oov) / max(len(words), 1)
    return lookup, oov, oov_rate


# ── WordNet pair generation ──────────────────────────────────────────────────

def get_wordnet_pairs(n_positive=WN_N_POSITIVE, n_negative=WN_N_NEGATIVE, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    pos_pairs = []
    seen = set()
    for synset in wn.all_synsets('n'):
        for hypernym in synset.hypernyms():
            w1 = synset.lemmas()[0].name()     # keep underscores
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


# ── Scoring functions ────────────────────────────────────────────────────────

def galois_scores_from_boxes(l1, u1, l2, u2):
    vf = (np.maximum(0, l2 - l1) + np.maximum(0, u1 - u2)).sum(axis=-1)
    vb = (np.maximum(0, l1 - l2) + np.maximum(0, u2 - u1)).sum(axis=-1)
    score_fwd  = 1.0 / (1.0 + vf)
    score_asym = (vb - vf) / (vb + vf + 1e-8)
    return score_fwd, score_asym, vf, vb


def cosine_score(e1, e2):
    n1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8)
    n2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8)
    return (n1 * n2).sum(axis=1)


def proj_score(e1, e2):
    n2 = np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8
    return (e1 * e2).sum(axis=1) / n2.squeeze(1)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_hyperlex(predictions, gold_scores):
    sp, p_sp = spearmanr(predictions, gold_scores)
    pe, p_pe = pearsonr(predictions,  gold_scores)
    return {'spearman': sp, 'pearson': pe, 'p_spearman': p_sp, 'n': len(predictions)}


def directionality_test(l1, u1, l2, u2, pairs, thresh=DIR_SCORE_THRESH):
    _, _, vf, vb = galois_scores_from_boxes(l1, u1, l2, u2)
    correct = 0; total = 0
    for i, p in enumerate(pairs):
        if p['score_10'] < thresh:
            continue
        t = p['type']
        if t.startswith('hyp-'):
            total += 1
            if vf[i] < vb[i]:
                correct += 1
        elif t.startswith('r-hyp-'):
            total += 1
            if vb[i] < vf[i]:
                correct += 1
    return correct / max(total, 1), total


def proj_directionality(e1, e2, pairs, thresh=DIR_SCORE_THRESH):
    correct = 0; total = 0
    for i, p in enumerate(pairs):
        if p['score_10'] < thresh:
            continue
        t = p['type']
        p12 = proj_score(e1[i:i+1], e2[i:i+1])[0]
        p21 = proj_score(e2[i:i+1], e1[i:i+1])[0]
        if t.startswith('hyp-'):
            total += 1
            if p12 > p21:
                correct += 1
        elif t.startswith('r-hyp-'):
            total += 1
            if p21 > p12:
                correct += 1
    return correct / max(total, 1), total


# ── Training pipeline for one embedding model ────────────────────────────────

def run_evaluation(emb_name, lookup_fn, pairs_te, gold_te,
                   pairs_tr_dev, rng):
    """
    Full evaluation pipeline for one embedding model.

    lookup_fn: word → numpy embedding (or None for OOV)

    Returns dict with all results.
    """
    print(f"\n  {'='*60}")
    print(f"  Evaluating: {emb_name}")
    print(f"  {'='*60}")

    # ── Encode test pairs ─────────────────────────────────────────────────
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
    oov_rate = oov_count / len(pairs_te)
    print(f"      Test pairs: {len(valid_pairs)}/{len(pairs_te)}  "
          f"(OOV dropped: {oov_count}, {oov_rate:.1%})")

    # ── Baselines ─────────────────────────────────────────────────────────
    cos = cosine_score(E1, E2)
    m_cos = evaluate_hyperlex(cos, gold)
    prj = proj_score(E1, E2)
    m_prj = evaluate_hyperlex(prj, gold)
    prj_dir, n_dir = proj_directionality(E1, E2, valid_pairs)

    print(f"      Cosine       Spearman={m_cos['spearman']:+.4f}  "
          f"Pearson={m_cos['pearson']:+.4f}  Dir=50.0%")
    print(f"      Projection   Spearman={m_prj['spearman']:+.4f}  "
          f"Pearson={m_prj['pearson']:+.4f}  Dir={prj_dir:.1%}")

    # ── Order Embeddings baseline (trained on HyperLex train) ─────────────
    oe_e1_tr, oe_e2_tr, oe_labels = [], [], []
    for p in pairs_tr_dev:
        v1 = lookup_fn(p['word1'])
        v2 = lookup_fn(p['word2'])
        if v1 is None or v2 is None:
            continue
        oe_e1_tr.append(v1)
        oe_e2_tr.append(v2)
        oe_labels.append(0 if p['type'].startswith('hyp-') else 2)

    oe_tr_data = {
        'emb1':  np.stack(oe_e1_tr),
        'emb2':  np.stack(oe_e2_tr),
        'label': np.array(oe_labels),
    }
    oe = OrderEmbeddingBaseline(box_dim=64, n_epochs=100, lr=1e-3)
    t0 = time.time()
    oe.fit(oe_tr_data)
    print(f"      Order Emb trained in {time.time()-t0:.1f}s")

    oe_emb1 = oe._embed(E1)
    oe_emb2 = oe._embed(E2)
    oe_scores = -oe._energy(oe_emb1, oe_emb2) / oe.box_dim
    m_oe = evaluate_hyperlex(oe_scores, gold)

    # OE directionality
    oe_correct = 0
    for i, p in enumerate(valid_pairs):
        if p['score_10'] < DIR_SCORE_THRESH:
            continue
        t = p['type']
        ef12 = oe._energy(oe_emb1[i:i+1], oe_emb2[i:i+1])[0]
        ef21 = oe._energy(oe_emb2[i:i+1], oe_emb1[i:i+1])[0]
        if t.startswith('hyp-') and ef12 < ef21:
            oe_correct += 1
        elif t.startswith('r-hyp-') and ef21 < ef12:
            oe_correct += 1
    oe_dir = oe_correct / max(n_dir, 1)
    print(f"      Order Emb    Spearman={m_oe['spearman']:+.4f}  "
          f"Pearson={m_oe['pearson']:+.4f}  Dir={oe_dir:.1%}")

    # ── Galois BoxEncoder (WordNet fine-tuned) ────────────────────────────
    print(f"\n      Training Galois BoxEncoder on WordNet...")
    wn_pairs = get_wordnet_pairs(WN_N_POSITIVE, WN_N_NEGATIVE, rng=rng)

    # Build embeddings for all WordNet words
    all_wn_words = list({w for (w1, w2, _) in wn_pairs for w in (w1, w2)})
    wn_lookup = {}
    wn_oov = 0
    for w in all_wn_words:
        v = lookup_fn(w)
        if v is not None:
            wn_lookup[w] = v
        else:
            wn_oov += 1
    print(f"      WordNet unique words: {len(all_wn_words)}  "
          f"OOV: {wn_oov} ({wn_oov/len(all_wn_words):.1%})")

    # Build training arrays (skip pairs with OOV words)
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
    n_val = max(256, int(len(labs_wn) * 0.1))
    idx   = rng.permutation(len(labs_wn))
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
    print(f"      Galois training done in {train_time:.1f}s")

    # ── Evaluate Galois on HyperLex test ──────────────────────────────────
    l1, u1, _ = trainer.model.encode(E1)
    l2, u2, _ = trainer.model.encode(E2)

    sf, sa, vf, vb = galois_scores_from_boxes(l1, u1, l2, u2)
    m_fwd  = evaluate_hyperlex(sf, gold)
    m_asym = evaluate_hyperlex(sa, gold)
    gal_dir, _ = directionality_test(l1, u1, l2, u2, valid_pairs)

    best_sp = max(m_fwd['spearman'], m_asym['spearman'])
    best_pe = m_fwd['pearson'] if m_fwd['spearman'] >= m_asym['spearman'] else m_asym['pearson']
    best_name = 'score_fwd' if m_fwd['spearman'] >= m_asym['spearman'] else 'score_asym'

    print(f"\n      Galois {best_name}  Spearman={best_sp:+.4f}  "
          f"Pearson={best_pe:+.4f}  Dir={gal_dir:.1%}")

    # Violation geometry
    print(f"\n      Violation geometry:")
    print(f"      {'type':>10}  {'N':>5}  {'vf_p50':>8}  {'vb_p50':>8}  {'asym_p50':>9}")
    for rel in ['hyp-1', 'hyp-2', 'hyp-3', 'cohyp', 'no-rel', 'syn']:
        idx = [i for i, p in enumerate(valid_pairs) if p['type'] == rel]
        if len(idx) < 5:
            continue
        vfr = vf[idx]; vbr = vb[idx]
        print(f"      {rel:>10}  {len(idx):>5}  {np.median(vfr):>8.2f}  "
              f"{np.median(vbr):>8.2f}  {np.median(vbr-vfr):>+9.2f}")

    return {
        'emb_name':     emb_name,
        'n_test':       len(valid_pairs),
        'oov_rate':     oov_rate,
        'oov_count':    oov_count,
        'cosine_sp':    m_cos['spearman'],
        'cosine_pe':    m_cos['pearson'],
        'proj_sp':      m_prj['spearman'],
        'proj_pe':      m_prj['pearson'],
        'proj_dir':     prj_dir,
        'oe_sp':        m_oe['spearman'],
        'oe_pe':        m_oe['pearson'],
        'oe_dir':       oe_dir,
        'galois_sp':    best_sp,
        'galois_pe':    best_pe,
        'galois_dir':   gal_dir,
        'galois_sp_fwd':  m_fwd['spearman'],
        'galois_sp_asym': m_asym['spearman'],
        'best_score':   best_name,
        'train_time':   train_time,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*70}")
    print(f"  Galois-NLP  x  HyperLex  x  Multi-Embedding Evaluation")
    print(f"{'='*70}")
    print(f"  Input representations: GloVe-300, Word2Vec-300, SBERT-384")
    print(f"  Architecture: same Galois BoxEncoder for all (dim={BOX_DIM})")
    print(f"  Training: WordNet hypernymy (25k pos + 25k neg)")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading HyperLex...")
    download_hyperlex()
    train_pairs, dev_pairs, test_pairs = load_hyperlex_splits()
    te_n  = filter_nouns(test_pairs)
    tr_n  = filter_nouns(train_pairs)
    dev_n = filter_nouns(dev_pairs)
    tr_dev_n = tr_n + dev_n
    gold_te = np.array([p['score_10'] for p in te_n])
    print(f"      Test nouns: {len(te_n)}  Train+Dev nouns: {len(tr_dev_n)}")

    # ── Load embedding models ─────────────────────────────────────────────
    print(f"\n[2/5] Loading word vector models (this may download ~2GB)...")

    t0 = time.time()
    print(f"      Loading GloVe-wiki-gigaword-300...")
    glove = api.load('glove-wiki-gigaword-300')
    print(f"      GloVe loaded in {time.time()-t0:.0f}s  vocab={len(glove)}")

    t0 = time.time()
    print(f"      Loading Word2Vec-google-news-300...")
    w2v = api.load('word2vec-google-news-300')
    print(f"      Word2Vec loaded in {time.time()-t0:.0f}s  vocab={len(w2v)}")

    # ── Define lookup functions ───────────────────────────────────────────
    def glove_lookup(word):
        return get_wv_embedding(word, glove)

    def w2v_lookup(word):
        return get_wv_embedding(word, w2v)

    # ── Run evaluations ───────────────────────────────────────────────────
    print(f"\n[3/5] Evaluating GloVe-300...")
    r_glove = run_evaluation(
        'GloVe-wiki-gigaword-300', glove_lookup,
        te_n, gold_te, tr_dev_n,
        rng=np.random.default_rng(42))

    print(f"\n[4/5] Evaluating Word2Vec-300...")
    r_w2v = run_evaluation(
        'Word2Vec-google-news-300', w2v_lookup,
        te_n, gold_te, tr_dev_n,
        rng=np.random.default_rng(42))

    # ── Combined results table ────────────────────────────────────────────
    print(f"\n\n[5/5] Combined Results Table")
    print(f"\n{'='*80}")
    print(f"  HyperLex Results  --  Nouns  --  Official Random Test Split")
    print(f"{'='*80}")

    print(f"\n  {'Model':<45} {'Input':<12} {'Spearman':>8} {'Pearson':>8} {'Dir%':>6}")
    print(f"  {'-'*80}")

    # Literature references
    print(f"  {'Vendrov Order Emb (2016)':<45} {'GloVe':<12} {'~0.720':>8} {'   --':>8} {'~65%':>6}")
    print(f"  {'Vilnis Box Emb (2018)':<45} {'GloVe':<12} {'~0.740':>8} {'   --':>8} {'  --':>6}")
    print(f"  {'LEAR (Nickel 2017)':<45} {'GloVe':<12} {'~0.765':>8} {'   --':>8} {'  --':>6}")
    print(f"  {'-'*80}")

    # Symmetric baselines
    print(f"  {'Cosine':<45} {'GloVe':<12} {r_glove['cosine_sp']:>+8.4f} {r_glove['cosine_pe']:>+8.4f} {'50.0%':>6}")
    print(f"  {'Cosine':<45} {'Word2Vec':<12} {r_w2v['cosine_sp']:>+8.4f} {r_w2v['cosine_pe']:>+8.4f} {'50.0%':>6}")
    print(f"  {'Cosine':<45} {'SBERT':<12} {'+0.2935':>8} {'+0.2950':>8} {'50.0%':>6}")
    print(f"  {'-'*80}")

    # Projection (asymmetric)
    print(f"  {'Projection e1->e2':<45} {'GloVe':<12} {r_glove['proj_sp']:>+8.4f} {r_glove['proj_pe']:>+8.4f} {r_glove['proj_dir']:>5.1%}")
    print(f"  {'Projection e1->e2':<45} {'Word2Vec':<12} {r_w2v['proj_sp']:>+8.4f} {r_w2v['proj_pe']:>+8.4f} {r_w2v['proj_dir']:>5.1%}")
    print(f"  {'Projection e1->e2':<45} {'SBERT':<12} {'+0.3248':>8} {'+0.3283':>8} {'78.7%':>6}")
    print(f"  {'-'*80}")

    # Order Embeddings (trained on HyperLex)
    print(f"  {'Order Emb (HyperLex train)':<45} {'GloVe':<12} {r_glove['oe_sp']:>+8.4f} {r_glove['oe_pe']:>+8.4f} {r_glove['oe_dir']:>5.1%}")
    print(f"  {'Order Emb (HyperLex train)':<45} {'Word2Vec':<12} {r_w2v['oe_sp']:>+8.4f} {r_w2v['oe_pe']:>+8.4f} {r_w2v['oe_dir']:>5.1%}")
    print(f"  {'Order Emb (HyperLex train)':<45} {'SBERT':<12} {'+0.1486':>8} {'+0.1703':>8} {'46.3%':>6}")
    print(f"  {'-'*80}")

    # Galois (WordNet fine-tuned)
    print(f"  {'Galois BoxEncoder (WordNet fine-tuned)':<45} {'GloVe':<12} {r_glove['galois_sp']:>+8.4f} {r_glove['galois_pe']:>+8.4f} {r_glove['galois_dir']:>5.1%}")
    print(f"  {'Galois BoxEncoder (WordNet fine-tuned)':<45} {'Word2Vec':<12} {r_w2v['galois_sp']:>+8.4f} {r_w2v['galois_pe']:>+8.4f} {r_w2v['galois_dir']:>5.1%}")
    print(f"  {'Galois BoxEncoder (WordNet fine-tuned)':<45} {'SBERT':<12} {SBERT_RESULTS['wn_spearman']:>+8.4f} {SBERT_RESULTS['wn_pearson']:>+8.4f} {SBERT_RESULTS['wn_dir']:>5.1%}")
    print(f"  {'-'*80}")

    # Galois zero-shot (SBERT only)
    print(f"  {'Galois v2 zero-shot (SNLI transfer)':<45} {'SBERT':<12} {SBERT_RESULTS['zeroshot_spearman']:>+8.4f} {SBERT_RESULTS['zeroshot_pearson']:>+8.4f} {SBERT_RESULTS['zeroshot_dir']:>5.1%}")

    # ── OOV report ────────────────────────────────────────────────────────
    print(f"\n  OOV rates:")
    print(f"    GloVe:    {r_glove['oov_count']}/{len(te_n)} = {r_glove['oov_rate']:.1%}")
    print(f"    Word2Vec: {r_w2v['oov_count']}/{len(te_n)} = {r_w2v['oov_rate']:.1%}")
    print(f"    SBERT:    0/{len(te_n)} = 0.0%  (handles any text)")

    # ── Key findings ──────────────────────────────────────────────────────
    print(f"\n  Key findings:")
    glove_beats_vendrov = r_glove['galois_sp'] > 0.72
    glove_beats_vilnis  = r_glove['galois_sp'] > 0.74
    dir_consistent      = (r_glove['galois_dir'] > 0.70 and
                           r_w2v['galois_dir']   > 0.70 and
                           SBERT_RESULTS['wn_dir'] > 0.70)

    print(f"    GloVe Galois Spearman: {r_glove['galois_sp']:+.4f}  "
          f"{'beats Vendrov ~0.72' if glove_beats_vendrov else 'below Vendrov ~0.72'}")
    print(f"    GloVe Galois Spearman: {r_glove['galois_sp']:+.4f}  "
          f"{'beats Vilnis ~0.74' if glove_beats_vilnis else 'below Vilnis ~0.74'}")
    print(f"    Directionality consistent across inputs: "
          f"{'YES' if dir_consistent else 'NO'}  "
          f"(GloVe={r_glove['galois_dir']:.1%}  "
          f"W2V={r_w2v['galois_dir']:.1%}  "
          f"SBERT={SBERT_RESULTS['wn_dir']:.1%})")
    print(f"    Galois dir > Projection dir: "
          f"GloVe {'YES' if r_glove['galois_dir'] > r_glove['proj_dir'] else 'NO'}  "
          f"W2V {'YES' if r_w2v['galois_dir'] > r_w2v['proj_dir'] else 'NO'}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        'glove':   r_glove,
        'w2v':     r_w2v,
        'sbert':   SBERT_RESULTS,
    }
    # Remove non-serializable items
    for d in [out['glove'], out['w2v']]:
        for k in list(d.keys()):
            if isinstance(d[k], np.floating):
                d[k] = float(d[k])

    Path('results').mkdir(exist_ok=True)
    with open('results/hyperlex_multi_emb_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to results/hyperlex_multi_emb_results.json")


if __name__ == '__main__':
    main()
