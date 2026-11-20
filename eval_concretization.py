"""
Concretization Quality Evaluation
===================================
Measures the faithfulness of γ_φ: how well does decoding from a box
reconstruct the original embedding?

Reports:
  1. Cosine similarity between original embedding and γ_φ(α_θ(x))
  2. L2 distance between original and reconstructed
  3. Breakdown by NLI class
  4. Round-trip Galois consistency: is the round-trip embedding closer to
     the original than to random sentences?
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
from src.dataset import build_synthetic_dataset, SENTENCES, build_synthetic_embeddings
from src.improved_trainer import ImprovedTrainer
from src.box_lattice import box_containment_violation, box_midpoint


CLASS_NAMES = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}


def cosine_similarity(a, b):
    """Row-wise cosine similarity."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a_norm * b_norm, axis=-1)


def main():
    seed = 42
    print("=" * 70)
    print("  Concretization Quality Evaluation")
    print("=" * 70)

    # ── Train model ───────────────────────────────────────────────────
    print("\nTraining model...")
    train_data, val_data, test_data = build_synthetic_dataset(
        n_train=3000, n_val=500, n_test=1000,
        dim=64, noise=0.05, rng=np.random.default_rng(seed))

    trainer = ImprovedTrainer(
        input_dim=64, hidden_dim=256, box_dim=64, lr=1e-3,
        lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
        margin=1.0, scale=5.0, rng=np.random.default_rng(seed + 10))
    trainer.train(train_data, val_data, n_epochs=40,
                  batch_size=64, verbose=False, patience=10)
    model = trainer.model

    # ── Evaluate on test set ──────────────────────────────────────────
    print("\nEvaluating concretization quality on test set...\n")

    # Encode and decode both premise and hypothesis
    for role, key in [('Premise', 'emb1'), ('Hypothesis', 'emb2')]:
        x_orig = test_data[key]
        l, u, _ = model.encode(x_orig)
        x_recon = model.decode(l, u)

        cos_sim = cosine_similarity(x_orig, x_recon)
        l2_dist = np.linalg.norm(x_orig - x_recon, axis=-1)

        print(f"  {role} Round-Trip Reconstruction:")
        print(f"    Cosine similarity:  mean={np.mean(cos_sim):.4f}  "
              f"median={np.median(cos_sim):.4f}  "
              f"std={np.std(cos_sim):.4f}")
        print(f"    L2 distance:        mean={np.mean(l2_dist):.4f}  "
              f"median={np.median(l2_dist):.4f}  "
              f"std={np.std(l2_dist):.4f}")

        # Breakdown by class
        print(f"\n    {'Class':<16} {'Cos mean':>10} {'Cos med':>10} "
              f"{'L2 mean':>10} {'L2 med':>10} {'N':>6}")
        print("    " + "─" * 62)
        for cls_id in [0, 1, 2]:
            mask = test_data['label'] == cls_id
            if mask.sum() == 0:
                continue
            c = cos_sim[mask]
            d = l2_dist[mask]
            print(f"    {CLASS_NAMES[cls_id]:<16} {np.mean(c):>10.4f} "
                  f"{np.median(c):>10.4f} {np.mean(d):>10.4f} "
                  f"{np.median(d):>10.4f} {mask.sum():>6}")
        print()

    # ── Round-trip closer than random? ────────────────────────────────
    print("-" * 70)
    print("  Round-Trip Proximity Test")
    print("  Is γ(α(x)) closer to x than to random sentences?")
    print("-" * 70)

    x_orig = test_data['emb1']
    l, u, _ = model.encode(x_orig)
    x_recon = model.decode(l, u)

    cos_to_orig = cosine_similarity(x_orig, x_recon)

    # Compare to random embeddings
    rng = np.random.default_rng(99)
    n = len(x_orig)
    random_idx = rng.permutation(n)
    x_random = x_orig[random_idx]
    cos_to_random = cosine_similarity(x_random, x_recon)

    closer_to_orig = np.mean(cos_to_orig > cos_to_random)

    print(f"\n  Cosine(recon, original):  mean={np.mean(cos_to_orig):.4f}")
    print(f"  Cosine(recon, random):    mean={np.mean(cos_to_random):.4f}")
    print(f"  Recon closer to original: {100*closer_to_orig:.1f}%")

    # Statistical test
    t_stat, p_val = stats.ttest_rel(cos_to_orig, cos_to_random)
    print(f"  Paired t-test: t={t_stat:.2f}, p={p_val:.2e}")

    # ── Galois consistency gap ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Galois Consistency Gap (per-pair)")
    print("  Δ = |s_A(α(x), a) - s_C(x, γ(a))|")
    print("-" * 70)

    x1 = test_data['emb1']
    x2 = test_data['emb2']
    l1, u1, _ = model.encode(x1)
    l2, u2, _ = model.encode(x2)

    # s_A: containment score in abstract domain
    viol_fwd = box_containment_violation(l1, u1, l2, u2)
    s_A = 1.0 / (1.0 + viol_fwd)

    # s_C: similarity in concrete domain via γ
    x2_recon = model.decode(l2, u2)
    s_C = cosine_similarity(x1, x2_recon)
    s_C = (s_C + 1.0) / 2.0  # normalize to [0,1]

    delta = np.abs(s_A - s_C)

    print(f"\n  Mean Δ_Galois:   {np.mean(delta):.4f}")
    print(f"  Median Δ_Galois: {np.median(delta):.4f}")
    print(f"  95th pctl:       {np.percentile(delta, 95):.4f}")
    print(f"  Max Δ_Galois:    {np.max(delta):.4f}")

    print(f"\n    {'Class':<16} {'Mean Δ':>10} {'Median Δ':>10} "
          f"{'95th':>10} {'Max':>10}")
    print("    " + "─" * 52)
    for cls_id in [0, 1, 2]:
        mask = test_data['label'] == cls_id
        d = delta[mask]
        print(f"    {CLASS_NAMES[cls_id]:<16} {np.mean(d):>10.4f} "
              f"{np.median(d):>10.4f} {np.percentile(d, 95):>10.4f} "
              f"{np.max(d):>10.4f}")

    # ── Qualitative: hierarchy sentences ──────────────────────────────
    print("\n" + "-" * 70)
    print("  Qualitative: Reconstruction of Hierarchy Sentences")
    print("-" * 70)

    base_emb = build_synthetic_embeddings(dim=64, noise=0.0,
                                           rng=np.random.default_rng(0))
    hierarchy = ['s0', 's1', 's2', 's3', 's4', 's5']

    print(f"\n  {'Sentence':<35} {'Cos(orig,recon)':>16} {'L2 dist':>10}")
    print("  " + "─" * 61)
    for sid in hierarchy:
        e = base_emb[sid][None, :]
        l, u, _ = model.encode(e)
        x_hat = model.decode(l, u)
        cos = float(cosine_similarity(e, x_hat))
        l2 = float(np.linalg.norm(e - x_hat))
        print(f"  {SENTENCES[sid]:<35} {cos:>16.4f} {l2:>10.4f}")

    # ── Nearest-neighbor retrieval ────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Nearest-Neighbor Retrieval via γ_φ")
    print("  For each sentence: encode → decode → find nearest original embedding")
    print("-" * 70)

    all_sids = list(SENTENCES.keys())
    all_embs = np.array([base_emb[s] for s in all_sids])

    correct = 0
    print(f"\n  {'Original':<35} {'Nearest after round-trip':<35} {'Match?'}")
    print("  " + "─" * 75)
    for i, sid in enumerate(all_sids):
        e = base_emb[sid][None, :]
        l, u, _ = model.encode(e)
        x_hat = model.decode(l, u)
        cos_all = cosine_similarity(
            np.tile(x_hat, (len(all_sids), 1)), all_embs)
        nearest_idx = np.argmax(cos_all)
        nearest_sid = all_sids[nearest_idx]
        match = nearest_sid == sid
        if match:
            correct += 1
        print(f"  {SENTENCES[sid]:<35} {SENTENCES[nearest_sid]:<35} "
              f"{'✓' if match else '✗'}")

    print(f"\n  Nearest-neighbor accuracy: {correct}/{len(all_sids)} "
          f"({100*correct/len(all_sids):.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    cos_all_prem = cosine_similarity(test_data['emb1'],
                                      model.decode(*model.encode(test_data['emb1'])[:2]))
    cos_all_hyp = cosine_similarity(test_data['emb2'],
                                     model.decode(*model.encode(test_data['emb2'])[:2]))
    print(f"  Round-trip cosine (premise):    {np.mean(cos_all_prem):.4f}")
    print(f"  Round-trip cosine (hypothesis): {np.mean(cos_all_hyp):.4f}")
    print(f"  Recon closer to original:       {100*closer_to_orig:.1f}%")
    print(f"  Nearest-neighbor retrieval:      {correct}/{len(all_sids)}")
    print(f"  Mean Galois gap:                {np.mean(delta):.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
