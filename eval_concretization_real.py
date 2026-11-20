"""
Concretization Quality on Real Data
====================================
Measures round-trip fidelity γ(α(x)) ≈ x on:
  1. WordNet + GloVe (word-level)
  2. SNLI + SBERT (sentence-level)

For each, trains a baseline model, then evaluates:
  - Cosine similarity between original and reconstructed embeddings
  - Fraction where reconstruction is closer to original than to random
  - Galois consistency gap
  - Nearest-neighbor retrieval accuracy
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from scipy import stats
from collections import defaultdict
from datasets import disable_progress_bar
disable_progress_bar()

from src.improved_trainer import ImprovedTrainer
from src.box_lattice import box_containment_violation

SEED = 42
BOX_DIM = 128
HIDDEN_DIM = 512


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a_norm * b_norm, axis=-1)


def evaluate_concretization(model, embeddings, labels=None, name=""):
    """Evaluate round-trip reconstruction quality."""
    x_orig = embeddings
    l, u, _ = model.encode(x_orig)
    x_recon = model.decode(l, u)

    cos_sim = cosine_similarity(x_orig, x_recon)
    l2_dist = np.linalg.norm(x_orig - x_recon, axis=-1)

    print(f"\n  {name} Round-Trip Reconstruction (n={len(x_orig)}):")
    print(f"    Cosine similarity:  mean={np.mean(cos_sim):.4f}  "
          f"median={np.median(cos_sim):.4f}  "
          f"std={np.std(cos_sim):.4f}")
    print(f"    L2 distance:        mean={np.mean(l2_dist):.4f}  "
          f"median={np.median(l2_dist):.4f}")

    # 95% CI for cosine
    n = len(cos_sim)
    se = np.std(cos_sim) / np.sqrt(n)
    ci_lo = np.mean(cos_sim) - 1.96 * se
    ci_hi = np.mean(cos_sim) + 1.96 * se
    print(f"    Cosine 95% CI:      [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Closer to original than random?
    rng = np.random.default_rng(99)
    perm = rng.permutation(n)
    x_random = x_orig[perm]
    cos_to_random = cosine_similarity(x_random, x_recon)
    closer = np.mean(cos_sim > cos_to_random)

    print(f"    Cosine(recon, random): mean={np.mean(cos_to_random):.4f}")
    print(f"    Recon closer to original: {100*closer:.1f}%")

    # Cohen's d
    diff = cos_sim - cos_to_random
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)
    print(f"    Cohen's d: {cohens_d:.2f}")

    # By class (if labels provided)
    if labels is not None:
        class_names = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}
        print(f"\n    {'Class':<16} {'Cos mean':>10} {'N':>6}")
        print("    " + "-" * 32)
        for cls_id in sorted(set(labels)):
            mask = labels == cls_id
            if mask.sum() > 0:
                cname = class_names.get(cls_id, f'Class {cls_id}')
                print(f"    {cname:<16} {np.mean(cos_sim[mask]):>10.4f} "
                      f"{mask.sum():>6}")

    return {
        'cos_mean': float(np.mean(cos_sim)),
        'cos_median': float(np.median(cos_sim)),
        'cos_std': float(np.std(cos_sim)),
        'cos_ci': (float(ci_lo), float(ci_hi)),
        'l2_mean': float(np.mean(l2_dist)),
        'closer_rate': float(closer),
        'cohens_d': float(cohens_d),
        'cos_to_random': float(np.mean(cos_to_random)),
    }


def evaluate_galois_gap(model, emb1, emb2):
    """Measure Galois consistency gap |s_A(α(x), a) - s_C(x, γ(a))|."""
    l1, u1, _ = model.encode(emb1)
    l2, u2, _ = model.encode(emb2)

    viol_fwd = box_containment_violation(l1, u1, l2, u2)
    s_A = 1.0 / (1.0 + viol_fwd)

    x2_recon = model.decode(l2, u2)
    s_C = cosine_similarity(emb1, x2_recon)
    s_C = (s_C + 1.0) / 2.0

    delta = np.abs(s_A - s_C)

    print(f"    Mean Galois gap:   {np.mean(delta):.4f}")
    print(f"    Median Galois gap: {np.median(delta):.4f}")
    print(f"    95th percentile:   {np.percentile(delta, 95):.4f}")
    print(f"    Max:               {np.max(delta):.4f}")

    return {
        'mean': float(np.mean(delta)),
        'median': float(np.median(delta)),
        'p95': float(np.percentile(delta, 95)),
        'max': float(np.max(delta)),
    }


def nearest_neighbor_accuracy(model, emb_dict, sample_size=500, rng=None):
    """Encode→decode→find nearest original. How often is it the same?"""
    if rng is None:
        rng = np.random.default_rng(SEED)

    keys = list(emb_dict.keys())
    if len(keys) > sample_size:
        idx = rng.choice(len(keys), sample_size, replace=False)
        keys = [keys[i] for i in idx]

    all_embs = np.stack([emb_dict[k] for k in keys])
    l, u, _ = model.encode(all_embs)
    recon = model.decode(l, u)

    # For each reconstructed embedding, find nearest original
    correct = 0
    for i in range(len(keys)):
        cos = cosine_similarity(
            np.tile(recon[i:i+1], (len(keys), 1)), all_embs)
        if np.argmax(cos) == i:
            correct += 1

    acc = correct / len(keys)
    print(f"    Nearest-neighbor retrieval: {correct}/{len(keys)} ({100*acc:.1f}%)")
    return acc


# ── WordNet + GloVe ─────────────────────────────────────────────────────────

def run_wordnet_evaluation():
    import gensim.downloader as api
    import nltk
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import wordnet as wn

    rng = np.random.default_rng(SEED)

    print("\n" + "=" * 70)
    print("  Concretization: WordNet + GloVe")
    print("=" * 70)

    print("\nLoading GloVe-wiki-gigaword-300...")
    glove = api.load("glove-wiki-gigaword-300")
    input_dim = 300

    # Build training data (same as eval_joins_wordnet.py)
    print("Building WordNet training pairs...")
    from eval_joins_wordnet import get_wordnet_pairs, build_training_data
    from eval_joins_wordnet import get_wv_embedding
    wn_pairs = get_wordnet_pairs(25000, 25000, rng=rng)
    train_data, val_data, dim = build_training_data(wn_pairs, glove, rng)
    print(f"  Train: {len(train_data['label'])}  Val: {len(val_data['label'])}")

    # Train model
    print("\nTraining model...")
    t0 = time.time()
    trainer = ImprovedTrainer(
        input_dim=dim, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM, lr=5e-4,
        lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
        margin=1.0, scale=5.0,
        rng=np.random.default_rng(SEED + 10))
    trainer.train(train_data, val_data, n_epochs=50, batch_size=256,
                  verbose=True, patience=7)
    print(f"  Training time: {time.time() - t0:.1f}s")
    model = trainer.model

    # Evaluate concretization on val set (premises and hypotheses)
    print("\n" + "-" * 70)
    print("  Round-Trip Reconstruction")
    print("-" * 70)
    prem_results = evaluate_concretization(
        model, val_data['emb1'], val_data['label'], "Premise (word1)")
    hyp_results = evaluate_concretization(
        model, val_data['emb2'], val_data['label'], "Hypothesis (word2)")

    # Galois consistency gap
    print("\n" + "-" * 70)
    print("  Galois Consistency Gap")
    print("-" * 70)
    gap = evaluate_galois_gap(model, val_data['emb1'], val_data['emb2'])

    # Nearest-neighbor on a sample of GloVe words from WordNet
    print("\n" + "-" * 70)
    print("  Nearest-Neighbor Retrieval")
    print("-" * 70)
    wn_words = list({s.lemmas()[0].name() for s in wn.all_synsets('n')})
    word_embs = {}
    for w in wn_words[:2000]:
        v = get_wv_embedding(w, glove)
        if v is not None:
            word_embs[w] = v
    nn_acc = nearest_neighbor_accuracy(model, word_embs, sample_size=500, rng=rng)

    return {
        'premise': prem_results, 'hypothesis': hyp_results,
        'gap': gap, 'nn_acc': nn_acc,
    }


# ── SNLI + SBERT ────────────────────────────────────────────────────────────

def run_snli_evaluation():
    rng = np.random.default_rng(SEED)

    print("\n" + "=" * 70)
    print("  Concretization: SNLI + SBERT")
    print("=" * 70)

    # Load SNLI
    print("\nLoading SNLI...")
    from eval_joins_snli import load_snli_entailment
    _, all_pairs = load_snli_entailment()

    # Subsample for speed
    n_total = 33000
    sample_idx = rng.choice(len(all_pairs), min(n_total, len(all_pairs)),
                            replace=False)
    pairs_subset = [all_pairs[i] for i in sample_idx]

    # Collect unique sentences
    all_sents = set()
    for p, h, _ in pairs_subset:
        all_sents.update([p, h])
    print(f"  Unique sentences: {len(all_sents)}")

    # Encode with SBERT
    print("\nEncoding with SBERT...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    sent_list = list(all_sents)
    print(f"  Encoding {len(sent_list)} sentences...")
    raw_embs = sbert.encode(sent_list, batch_size=256,
                            show_progress_bar=False, normalize_embeddings=True)
    emb_dict = {s: raw_embs[i] for i, s in enumerate(sent_list)}
    input_dim = raw_embs.shape[1]
    del sbert
    print(f"  SBERT dim={input_dim}, encoding time: {time.time() - t0:.1f}s")

    # Build training data
    print("\nBuilding training data...")
    from eval_joins_snli import build_training_data
    train_data, val_data = build_training_data(pairs_subset, emb_dict, rng=rng)
    print(f"  Train: {len(train_data['label'])}  Val: {len(val_data['label'])}")

    # Train model
    print("\nTraining model...")
    t0 = time.time()
    trainer = ImprovedTrainer(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM, lr=5e-4,
        lambda1=0.1, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.0,
        margin=1.0, scale=5.0,
        rng=np.random.default_rng(SEED + 10))
    trainer.train(train_data, val_data, n_epochs=50, batch_size=256,
                  verbose=True, patience=7)
    print(f"  Training time: {time.time() - t0:.1f}s")
    model = trainer.model

    # Evaluate concretization
    print("\n" + "-" * 70)
    print("  Round-Trip Reconstruction")
    print("-" * 70)
    prem_results = evaluate_concretization(
        model, val_data['emb1'], val_data['label'], "Premise")
    hyp_results = evaluate_concretization(
        model, val_data['emb2'], val_data['label'], "Hypothesis")

    # Galois gap
    print("\n" + "-" * 70)
    print("  Galois Consistency Gap")
    print("-" * 70)
    gap = evaluate_galois_gap(model, val_data['emb1'], val_data['emb2'])

    # Nearest-neighbor on sentence embeddings
    print("\n" + "-" * 70)
    print("  Nearest-Neighbor Retrieval")
    print("-" * 70)
    sample_keys = list(emb_dict.keys())[:1000]
    sample_embs = {k: emb_dict[k] for k in sample_keys}
    nn_acc = nearest_neighbor_accuracy(model, sample_embs, sample_size=500,
                                       rng=rng)

    return {
        'premise': prem_results, 'hypothesis': hyp_results,
        'gap': gap, 'nn_acc': nn_acc,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Concretization Quality on Real Data")
    print("=" * 70)

    wn_results = run_wordnet_evaluation()
    snli_results = run_snli_evaluation()

    # ── Combined summary ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  COMBINED SUMMARY")
    print("=" * 70)

    print(f"\n  {'Metric':<35} {'Synthetic':>12} {'WordNet':>12} {'SNLI':>12}")
    print("  " + "-" * 71)
    print(f"  {'Cosine(x, gamma(alpha(x)))':<35} {'0.924':>12} "
          f"{wn_results['premise']['cos_mean']:>12.4f} "
          f"{snli_results['premise']['cos_mean']:>12.4f}")
    print(f"  {'95% CI lower':<35} {'0.921':>12} "
          f"{wn_results['premise']['cos_ci'][0]:>12.4f} "
          f"{snli_results['premise']['cos_ci'][0]:>12.4f}")
    print(f"  {'95% CI upper':<35} {'0.927':>12} "
          f"{wn_results['premise']['cos_ci'][1]:>12.4f} "
          f"{snli_results['premise']['cos_ci'][1]:>12.4f}")
    print(f"  {'Closer to original':<35} {'94.4%':>12} "
          f"{wn_results['premise']['closer_rate']:>11.1%} "
          f"{snli_results['premise']['closer_rate']:>11.1%}")
    print(f"  {'Cohen d':<35} {'1.40':>12} "
          f"{wn_results['premise']['cohens_d']:>12.2f} "
          f"{snli_results['premise']['cohens_d']:>12.2f}")
    print(f"  {'Cosine to random':<35} {'0.031':>12} "
          f"{wn_results['premise']['cos_to_random']:>12.4f} "
          f"{snli_results['premise']['cos_to_random']:>12.4f}")
    print(f"  {'NN retrieval accuracy':<35} {'100%':>12} "
          f"{wn_results['nn_acc']:>11.1%} "
          f"{snli_results['nn_acc']:>11.1%}")
    print(f"  {'Mean Galois gap':<35} {'0.142':>12} "
          f"{wn_results['gap']['mean']:>12.4f} "
          f"{snli_results['gap']['mean']:>12.4f}")
    print(f"  {'95th pctl Galois gap':<35} {'0.334':>12} "
          f"{wn_results['gap']['p95']:>12.4f} "
          f"{snli_results['gap']['p95']:>12.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
