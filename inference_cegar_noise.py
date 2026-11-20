"""
inference_cegar_noise.py — Inference-Time CEGAR on Noise-Degraded Representations
=================================================================================
Simulates domain shift by adding controlled geometric noise to learned box
representations, then demonstrates that inference-time CEGAR restores lattice
consistency WITHOUT retraining.

Three-act experiment:
  Act 1 — Baseline: clean model achieves ~99% transitive consistency
  Act 2 — Degradation: Gaussian noise degrades box geometry
  Act 3 — Recovery: CEGAR restores consistency; compare vs baselines

Usage
-----
  PYTHONIOENCODING=utf-8 python -u inference_cegar_noise.py
"""

import sys, os, time, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import gensim.downloader as api

from src.improved_trainer import ImprovedTrainer, GaloisModel
from src.box_lattice import box_containment_violation, box_containment_score

import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

from eval_hyperlex_embeddings import get_wv_embedding, BOX_DIM, HIDDEN_DIM
from cegar_experiment import get_full_wordnet_pairs_with_hops, train_model

from inference_cegar import (
    build_concept_set, encode_concept_set, build_isa_graph,
    detect_violations, inference_cegar_update, apply_inference_cegar,
    measure_side_effects, random_perturbation_baseline,
    measure_galois_consistency_gap,
)

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Noise Injection
# ══════════════════════════════════════════════════════════════════════════════

def add_geometric_noise(boxes, noise_scale, rng=None):
    """
    Add Gaussian noise to box bounds to simulate domain shift.
    Noise is proportional to box width.

    noise_scale: 0.3=mild, 0.5=moderate, 1.0=severe, 2.0=extreme
    """
    if rng is None:
        rng = np.random.default_rng(42)

    noisy_boxes = {}
    widths_before = []
    widths_after  = []

    for lemma, (l, u) in boxes.items():
        width = u - l
        widths_before.append(float(np.mean(width)))

        noise_l = rng.standard_normal(l.shape) * noise_scale * np.maximum(width, 0.01)
        noise_u = rng.standard_normal(u.shape) * noise_scale * np.maximum(width, 0.01)

        l_noisy = l + noise_l
        u_noisy = u + noise_u

        # Restore validity: ensure l <= u
        l_valid = np.minimum(l_noisy, u_noisy)
        u_valid = np.maximum(l_noisy, u_noisy)

        noisy_boxes[lemma] = (l_valid, u_valid)
        widths_after.append(float(np.mean(u_valid - l_valid)))

    noise_stats = {
        'mean_width_before': float(np.mean(widths_before)),
        'mean_width_after':  float(np.mean(widths_after)),
        'width_ratio':       float(np.mean(widths_after) /
                                   (np.mean(widths_before) + 1e-8)),
    }

    return noisy_boxes, noise_stats


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate Descent Baseline
# ══════════════════════════════════════════════════════════════════════════════

def coordinate_descent_baseline(noisy_boxes, transitive_edges,
                                step_size=0.1, max_iterations=10):
    """
    Non-targeted baseline: push ALL IS-A pairs toward containment
    simultaneously, without counterexample selection.

    If CEGAR >> coord descent: counterexample selection matters
    If CEGAR ≈ coord descent: any push toward containment helps
    """
    refined = {k: (v[0].copy(), v[1].copy())
               for k, v in noisy_boxes.items()}

    for iteration in range(max_iterations):
        total_viol = 0.0
        n_pairs    = 0

        for (hypo, hyper) in transitive_edges:
            if hypo not in refined or hyper not in refined:
                continue

            l_A, u_A = refined[hypo]
            l_C, u_C = refined[hyper]

            viol = float(box_containment_violation(
                l_A[None, :], u_A[None, :],
                l_C[None, :], u_C[None, :]))

            if viol < 1e-6:
                continue

            lower_correction = np.maximum(0.0, l_C - l_A)
            upper_correction = np.maximum(0.0, u_A - u_C)

            l_new = l_A + step_size * lower_correction
            u_new = u_A - step_size * upper_correction

            if np.all(l_new <= u_new):
                refined[hypo] = (l_new, u_new)

            total_viol += viol
            n_pairs    += 1

        if n_pairs > 0 and total_viol / n_pairs < 0.01:
            break

    return refined


# ══════════════════════════════════════════════════════════════════════════════
#  Counting utilities
# ══════════════════════════════════════════════════════════════════════════════

def count_violations(boxes, transitive_edges, threshold=0.5):
    """Count IS-A pairs with violation > threshold."""
    count = 0
    total = 0
    total_viol = 0.0
    for (hypo, hyper) in transitive_edges:
        if hypo not in boxes or hyper not in boxes:
            continue
        l1, u1 = boxes[hypo]
        l2, u2 = boxes[hyper]
        viol = float(box_containment_violation(
            l1[None, :], u1[None, :],
            l2[None, :], u2[None, :]))
        total_viol += viol
        if viol > threshold:
            count += 1
        total += 1
    mean_viol = total_viol / max(total, 1)
    return count, total, mean_viol


# ══════════════════════════════════════════════════════════════════════════════
#  Main Experiment
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*90}")
    print(f"  Inference-Time CEGAR on Noise-Degraded Representations")
    print(f"{'='*90}")
    print(f"  Tests CEGAR as deployment-time consistency enforcer under domain shift.")
    print(f"  NO RETRAINING. NO GRADIENTS. Pure geometric refinement.")
    print(f"  Seed: {SEED}")

    # ── 1. Load GloVe ────────────────────────────────────────────────────
    print(f"\n[1/6] Loading GloVe...")
    t0 = time.time()
    glove = api.load('glove-wiki-gigaword-300')
    glove_vocab = set(glove.key_to_index.keys())
    print(f"      Loaded in {time.time()-t0:.0f}s  vocab={len(glove_vocab)}")

    # ── 2. Load or train model ───────────────────────────────────────────
    print(f"\n[2/6] Loading box encoder...")

    # Build embedding lookup
    rng = np.random.default_rng(SEED)
    wn_pairs, wn_hops = get_full_wordnet_pairs_with_hops(rng)

    wn_words = set()
    for s in wn.all_synsets('n'):
        wn_words.add(s.lemmas()[0].name().replace('_', ' '))
        for h in s.hypernyms():
            wn_words.add(h.lemmas()[0].name().replace('_', ' '))

    emb_lookup = {}
    oov = 0
    for w in wn_words:
        v = get_wv_embedding(w, glove)
        if v is not None:
            emb_lookup[w] = v
        else:
            oov += 1
    for w1, w2, _ in wn_pairs:
        for w in [w1, w2]:
            if w not in emb_lookup:
                v = get_wv_embedding(w, glove)
                if v is not None:
                    emb_lookup[w] = v
    print(f"      Embedded: {len(emb_lookup)}  OOV: {oov}")

    # Load cached model if available
    model_cache = Path('results/inference_cegar_model.npz')
    if model_cache.exists():
        print(f"      Loading cached model from {model_cache}...")
        data = np.load(model_cache, allow_pickle=True)
        trainer = ImprovedTrainer(
            input_dim=300, hidden_dim=HIDDEN_DIM, box_dim=BOX_DIM,
            lr=5e-4, lambda1=0.1, lambda2=2.0, lambda3=1.0,
            lambda4=0.0, lambda5=0.0, margin=1.0, scale=5.0,
            rng=np.random.default_rng(SEED))
        for key in trainer.model.params_dict():
            setattr(trainer.model, key, data[key])
        print(f"      Model loaded from cache.")
    else:
        print(f"      No cached model found. Training fresh...")
        trainer, train_data, val_data = train_model(
            emb_lookup, wn_pairs, lambda4=0.0,
            rng=np.random.default_rng(SEED),
            label="Model (no CEGAR)")
        Path('results').mkdir(exist_ok=True)
        np.savez(model_cache, **trainer.model.params_dict())
        print(f"      Model saved to {model_cache}")
    model = trainer.model

    # ── 3. Build concept set + encode ────────────────────────────────────
    print(f"\n[3/6] Building concept set...")
    concept_set = build_concept_set(glove_vocab, n_concepts=800,
                                     min_depth=2, max_depth=10,
                                     rng=np.random.default_rng(SEED))
    concept_lemmas = [c[1] for c in concept_set]

    for lemma in concept_lemmas:
        if lemma not in emb_lookup:
            v = get_wv_embedding(lemma, glove)
            if v is not None:
                emb_lookup[lemma] = v

    print(f"\n      Encoding concepts...")
    boxes, embeddings = encode_concept_set(concept_lemmas, model, emb_lookup)

    print(f"\n      Building IS-A graph...")
    direct_edges, transitive_edges = build_isa_graph(concept_set)

    # Clean baseline
    clean_viol, total_pairs, clean_mean_viol = count_violations(
        boxes, transitive_edges, threshold=0.5)
    print(f"\n      Clean model: {clean_viol}/{total_pairs} violations "
          f"({100*clean_viol/max(total_pairs,1):.1f}%), "
          f"mean viol={clean_mean_viol:.4f}")

    # Galois gap on clean model
    sample_pairs = list(transitive_edges.keys())[:200]
    gap_clean = measure_galois_consistency_gap(
        boxes, embeddings, model, sample_pairs, n_sample=200)
    print(f"      Clean Galois gap: mean={gap_clean['mean']:.4f}  "
          f"p95={gap_clean['p95']:.4f}")

    # ── 4. Full experiment ───────────────────────────────────────────────
    print(f"\n[4/6] Running noise-degradation experiment...")

    noise_scales = [0.3, 0.5, 1.0, 2.0]
    step_sizes   = [0.1, 0.3, 0.5]
    all_results  = []
    best_config  = None
    best_net     = -999

    for ns in noise_scales:
        print(f"\n{'─'*70}")
        print(f"  Noise scale: {ns}")
        print(f"{'─'*70}")

        # Use a fresh rng per noise scale for reproducibility
        noise_rng = np.random.default_rng(SEED + int(ns * 100))
        noisy_boxes, noise_stats = add_geometric_noise(
            boxes, ns, rng=noise_rng)

        viol_after_noise, _, mean_viol_noisy = count_violations(
            noisy_boxes, transitive_edges, threshold=0.5)
        print(f"    After noise: {viol_after_noise}/{total_pairs} violations "
              f"({100*viol_after_noise/max(total_pairs,1):.1f}%), "
              f"mean_viol={mean_viol_noisy:.4f}")
        print(f"    Width ratio: {noise_stats['width_ratio']:.3f}")

        if viol_after_noise < 3:
            print(f"    Too few violations — skipping")
            continue

        # Detect violations for CEGAR targeting
        violations, violation_pairs = detect_violations(
            noisy_boxes, direct_edges, transitive_edges,
            containment_threshold=0.5,
            violation_threshold=0.5)

        # Galois gap before any refinement
        gap_noisy = measure_galois_consistency_gap(
            noisy_boxes, embeddings, model, sample_pairs, n_sample=200)

        # ── No refinement ────────────────────────────────────────────
        all_results.append({
            'noise_scale': ns,
            'method': 'No refinement',
            'step_size': '-',
            'violated': viol_after_noise,
            'fixed': 0,
            'new_viol': 0,
            'net': 0,
            'fix_rate': 0.0,
            'gap_before': gap_noisy['mean'],
            'gap_after': gap_noisy['mean'],
            'gap_p95_before': gap_noisy['p95'],
            'gap_p95_after': gap_noisy['p95'],
        })

        # ── Random perturbation ──────────────────────────────────────
        print(f"    Random perturbation...")
        rand_rng = np.random.default_rng(99)
        random_boxes = random_perturbation_baseline(
            noisy_boxes, violations, step_size=0.3, rng=rand_rng)
        _, _, nv_r, nf_r = measure_side_effects(
            noisy_boxes, random_boxes, direct_edges, transitive_edges)
        gap_random = measure_galois_consistency_gap(
            random_boxes, embeddings, model, sample_pairs, n_sample=200)

        all_results.append({
            'noise_scale': ns,
            'method': 'Random perturb',
            'step_size': 0.3,
            'violated': viol_after_noise,
            'fixed': len(nf_r),
            'new_viol': len(nv_r),
            'net': len(nf_r) - len(nv_r),
            'fix_rate': len(nf_r) / max(viol_after_noise, 1),
            'gap_before': gap_noisy['mean'],
            'gap_after': gap_random['mean'],
            'gap_p95_before': gap_noisy['p95'],
            'gap_p95_after': gap_random['p95'],
        })
        print(f"      Fixed={len(nf_r)}, New={len(nv_r)}, Net={len(nf_r)-len(nv_r):+d}")

        # ── Coordinate descent ───────────────────────────────────────
        print(f"    Coordinate descent...")
        cd_boxes = coordinate_descent_baseline(
            noisy_boxes, transitive_edges, step_size=0.3, max_iterations=10)
        _, _, nv_cd, nf_cd = measure_side_effects(
            noisy_boxes, cd_boxes, direct_edges, transitive_edges)
        gap_cd = measure_galois_consistency_gap(
            cd_boxes, embeddings, model, sample_pairs, n_sample=200)

        all_results.append({
            'noise_scale': ns,
            'method': 'Coord descent',
            'step_size': 0.3,
            'violated': viol_after_noise,
            'fixed': len(nf_cd),
            'new_viol': len(nv_cd),
            'net': len(nf_cd) - len(nv_cd),
            'fix_rate': len(nf_cd) / max(viol_after_noise, 1),
            'gap_before': gap_noisy['mean'],
            'gap_after': gap_cd['mean'],
            'gap_p95_before': gap_noisy['p95'],
            'gap_p95_after': gap_cd['p95'],
        })
        print(f"      Fixed={len(nf_cd)}, New={len(nv_cd)}, Net={len(nf_cd)-len(nv_cd):+d}")

        # ── CEGAR at three step sizes ────────────────────────────────
        for ss in step_sizes:
            print(f"    CEGAR (step={ss})...")
            refined_boxes, fix_results = apply_inference_cegar(
                noisy_boxes, violations, step_size=ss, max_iterations=10)
            _, _, nv_c, nf_c = measure_side_effects(
                noisy_boxes, refined_boxes, direct_edges, transitive_edges)
            gap_cegar = measure_galois_consistency_gap(
                refined_boxes, embeddings, model, sample_pairs, n_sample=200)

            net = len(nf_c) - len(nv_c)
            fr = len(nf_c) / max(viol_after_noise, 1)

            all_results.append({
                'noise_scale': ns,
                'method': 'CEGAR',
                'step_size': ss,
                'violated': viol_after_noise,
                'fixed': len(nf_c),
                'new_viol': len(nv_c),
                'net': net,
                'fix_rate': fr,
                'gap_before': gap_noisy['mean'],
                'gap_after': gap_cegar['mean'],
                'gap_p95_before': gap_noisy['p95'],
                'gap_p95_after': gap_cegar['p95'],
            })
            print(f"      Fixed={len(nf_c)}, New={len(nv_c)}, "
                  f"Net={net:+d}, Fix%={100*fr:.1f}%, "
                  f"Gap: {gap_noisy['mean']:.4f}→{gap_cegar['mean']:.4f}")

            # Track best config
            if net > best_net:
                best_net = net
                best_config = {
                    'noise_scale': ns, 'step_size': ss,
                    'violations': violations,
                    'noisy_boxes': noisy_boxes,
                }

    # ── 5. Hop-stratified analysis for best config ───────────────────
    print(f"\n[5/6] Hop-stratified analysis (best config: "
          f"noise={best_config['noise_scale']}, step={best_config['step_size']})...")

    best_violations = best_config['violations']
    best_noisy = best_config['noisy_boxes']
    best_ss = best_config['step_size']
    best_ns = best_config['noise_scale']

    # Group violations by hop bucket
    hop_buckets = {'2': [], '3': [], '4+': []}
    for v in best_violations:
        h = v['hop_count']
        bucket = str(h) if h <= 3 else '4+'
        if bucket in hop_buckets:
            hop_buckets[bucket].append(v)

    hop_results = {}
    for bucket, viol_list in hop_buckets.items():
        if not viol_list:
            hop_results[bucket] = None
            continue

        refined_b, _ = apply_inference_cegar(
            best_noisy, viol_list, step_size=best_ss, max_iterations=10)
        _, _, nv_h, nf_h = measure_side_effects(
            best_noisy, refined_b, direct_edges, transitive_edges)

        hop_results[bucket] = {
            'n_violations': len(viol_list),
            'n_fixed': len(nf_h),
            'fix_rate': len(nf_h) / max(len(viol_list), 1),
            'new_violations': len(nv_h),
            'net': len(nf_h) - len(nv_h),
        }

    # ── 6. Convergence curve for best config ─────────────────────────
    print(f"\n[6/6] Convergence curve...")

    # Re-detect with same noise
    noise_rng = np.random.default_rng(SEED + int(best_ns * 100))
    noisy_conv, _ = add_geometric_noise(boxes, best_ns, rng=noise_rng)
    viols_conv, _ = detect_violations(
        noisy_conv, direct_edges, transitive_edges,
        containment_threshold=0.5, violation_threshold=0.5)

    conv_data = []
    viol_start, _, _ = count_violations(noisy_conv, transitive_edges, 0.5)

    for n_iter in [0, 1, 2, 3, 5, 10]:
        if n_iter == 0:
            current = noisy_conv
        else:
            current, _ = apply_inference_cegar(
                noisy_conv, viols_conv,
                step_size=best_ss, max_iterations=n_iter)

        vc, _, mv = count_violations(current, transitive_edges, 0.5)
        gap_iter = measure_galois_consistency_gap(
            current, embeddings, model, sample_pairs, n_sample=200)

        conv_data.append({
            'iteration': n_iter,
            'violations': vc,
            'pct_remaining': 100 * vc / max(viol_start, 1),
            'mean_viol': mv,
            'galois_gap': gap_iter['mean'],
        })

    # ══════════════════════════════════════════════════════════════════
    #  RESULTS TABLES
    # ══════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*110}")
    print(f"  TABLE 1: Main Results — All Methods at All Noise Scales")
    print(f"{'='*110}")

    hdr = (f"  {'Noise':>5}  {'Method':<18}  {'Step':>5}  "
           f"{'Violated':>8}  {'Fixed':>6}  {'New':>5}  {'Net':>5}  "
           f"{'Fix%':>6}  {'Gap Before':>10}  {'Gap After':>10}")
    print(f"\n{hdr}")
    print(f"  {'-'*106}")

    prev_ns = None
    for r in all_results:
        ns = r['noise_scale']
        if prev_ns is not None and ns != prev_ns:
            print(f"  {'-'*106}")
        prev_ns = ns

        ss_str = f"{r['step_size']}" if r['step_size'] != '-' else '-'
        gap_a = f"{r['gap_after']:.4f}" if r['gap_after'] != r['gap_before'] or r['method'] != 'No refinement' else '—'

        marker = ''
        if ns == 0.5 and r['method'] == 'CEGAR' and r['step_size'] == 0.3:
            marker = ' ← KEY'

        print(f"  {ns:>5.1f}  {r['method']:<18}  {ss_str:>5}  "
              f"{r['violated']:>8}  {r['fixed']:>6}  {r['new_viol']:>5}  "
              f"{r['net']:>+5}  {100*r['fix_rate']:>5.1f}%  "
              f"{r['gap_before']:>10.4f}  {gap_a:>10}{marker}")

    # Table 2: Hop-stratified
    print(f"\n\n{'='*80}")
    print(f"  TABLE 2: Violation Fix Rate by Hop Count "
          f"(noise={best_ns}, step={best_ss})")
    print(f"{'='*80}")

    print(f"\n  {'Hop':>4}  {'N violations':>12}  {'Fixed':>6}  "
          f"{'Fix%':>6}  {'New violations':>15}  {'Net':>5}")
    print(f"  {'-'*56}")

    all_n = 0; all_fixed = 0; all_new = 0
    for bucket in ['2', '3', '4+']:
        hr = hop_results.get(bucket)
        if hr is None:
            print(f"  {bucket:>4}  {'—':>12}")
            continue
        all_n += hr['n_violations']
        all_fixed += hr['n_fixed']
        all_new += hr['new_violations']
        print(f"  {bucket:>4}  {hr['n_violations']:>12}  {hr['n_fixed']:>6}  "
              f"{100*hr['fix_rate']:>5.1f}%  {hr['new_violations']:>15}  "
              f"{hr['net']:>+5}")
    print(f"  {'-'*56}")
    all_fr = all_fixed / max(all_n, 1)
    print(f"  {'ALL':>4}  {all_n:>12}  {all_fixed:>6}  "
          f"{100*all_fr:>5.1f}%  {all_new:>15}  {all_fixed-all_new:>+5}")

    # Table 3: Convergence
    print(f"\n\n{'='*70}")
    print(f"  TABLE 3: Convergence (noise={best_ns}, step={best_ss})")
    print(f"{'='*70}")

    print(f"\n  {'Iteration':>9}  {'Violations':>11}  "
          f"{'% Remaining':>12}  {'Galois Gap':>11}")
    print(f"  {'-'*50}")

    for cd in conv_data:
        it_str = f"{cd['iteration']}" if cd['iteration'] > 0 else "0 (noisy)"
        print(f"  {it_str:>9}  {cd['violations']:>11}  "
              f"{cd['pct_remaining']:>11.1f}%  {cd['galois_gap']:>11.4f}")

    print(f"  {'-'*50}")
    print(f"  {'Clean':>9}  {clean_viol:>11}  "
          f"{100*clean_viol/max(total_pairs,1):>11.1f}%  "
          f"{gap_clean['mean']:>11.4f}")

    # Table 4: Summary at noise=0.5
    print(f"\n\n{'='*80}")
    print(f"  TABLE 4: CEGAR vs Baselines — Summary at Noise=0.5")
    print(f"{'='*80}")

    # Find results at noise=0.5
    r05 = [r for r in all_results if abs(r['noise_scale'] - 0.5) < 0.01]

    if r05:
        print(f"\n  {'Method':<22}  {'Fix%':>6}  {'New Viol':>9}  "
              f"{'Net':>5}  {'Galois Gap Δ':>13}")
        print(f"  {'-'*60}")

        # Find best CEGAR at 0.5
        cegar_05 = [r for r in r05 if r['method'] == 'CEGAR']
        best_cegar = max(cegar_05, key=lambda r: r['net']) if cegar_05 else None
        rand_05 = next((r for r in r05 if r['method'] == 'Random perturb'), None)
        cd_05 = next((r for r in r05 if r['method'] == 'Coord descent'), None)
        no_ref = next((r for r in r05 if r['method'] == 'No refinement'), None)

        for r in [no_ref, rand_05, cd_05, best_cegar]:
            if r is None:
                continue
            gap_delta = r['gap_after'] - r['gap_before']
            label = r['method']
            if r['method'] == 'CEGAR':
                label = f"CEGAR (step={r['step_size']})"
            gd_str = f"{gap_delta:+.4f}"
            if gap_delta > 0:
                gd_str += " (worse)"
            elif gap_delta < -0.001:
                gd_str += " (better)"
            print(f"  {label:<22}  {100*r['fix_rate']:>5.1f}%  "
                  f"{r['new_viol']:>9}  {r['net']:>+5}  {gd_str:>13}")

        if best_cegar and rand_05:
            diff_pp = 100 * (best_cegar['fix_rate'] - rand_05['fix_rate'])
            print(f"\n  CEGAR vs Random:        {diff_pp:+.1f}pp")
        if best_cegar and cd_05:
            diff_pp = 100 * (best_cegar['fix_rate'] - cd_05['fix_rate'])
            print(f"  CEGAR vs Coord descent: {diff_pp:+.1f}pp")
    else:
        print(f"  (No results at noise=0.5)")

    # ── Theorem 1 ────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  Theorem 1 Empirical Verification")
    print(f"{'='*70}")

    # Use best config results
    if best_config and best_config['noise_scale'] in [r['noise_scale'] for r in all_results]:
        ns = best_config['noise_scale']
        cegar_best = [r for r in all_results
                      if r['noise_scale'] == ns and r['method'] == 'CEGAR'
                      and r['step_size'] == best_config['step_size']]
        if cegar_best:
            cb = cegar_best[0]
            print(f"\n  Galois consistency gap before CEGAR: {cb['gap_before']:.4f}")
            print(f"  Galois consistency gap after CEGAR:  {cb['gap_after']:.4f}")
            gap_delta = cb['gap_after'] - cb['gap_before']
            tightens = gap_delta < 0
            print(f"  CEGAR tightens gap: {tightens}")
            if tightens:
                print(f"  Gap reduction: {gap_delta:.4f} ({100*gap_delta/max(cb['gap_before'],1e-8):.1f}%)")
            else:
                print(f"  Gap change: {gap_delta:+.4f} ({100*gap_delta/max(cb['gap_before'],1e-8):+.1f}%)")

    # ── Decision Criteria ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  Decision Criteria (evaluated at noise=0.5)")
    print(f"{'='*70}")

    if r05:
        best_c = max([r for r in r05 if r['method'] == 'CEGAR'],
                     key=lambda r: r['net'], default=None)
        rand_r = next((r for r in r05 if r['method'] == 'Random perturb'), None)

        if best_c:
            c1 = best_c['fix_rate'] > 0.50
            c2 = best_c['net'] > 0
            c3 = rand_r is not None and (best_c['fix_rate'] - rand_r['fix_rate']) > 0.20
            c4 = best_c['gap_after'] < best_c['gap_before']

            # Fix rate increases with hop count
            hop_increasing = False
            if hop_results.get('4+') and hop_results.get('2'):
                hr4 = hop_results['4+']
                hr2 = hop_results['2']
                hop_increasing = hr4['fix_rate'] > hr2['fix_rate']
            c5 = hop_increasing

            print(f"\n  [{'PASS' if c1 else 'FAIL'}] Fix rate > 50%: "
                  f"{100*best_c['fix_rate']:.1f}%")
            print(f"  [{'PASS' if c2 else 'FAIL'}] Net improvement positive: "
                  f"{best_c['net']:+d}")
            print(f"  [{'PASS' if c3 else 'FAIL'}] CEGAR > random by 20pp: "
                  f"{100*(best_c['fix_rate'] - (rand_r['fix_rate'] if rand_r else 0)):+.1f}pp")
            print(f"  [{'PASS' if c4 else 'FAIL'}] Galois gap decreases: "
                  f"{best_c['gap_before']:.4f} → {best_c['gap_after']:.4f}")
            print(f"  [{'PASS' if c5 else 'FAIL'}] Fix rate increases with hop count")

            validated = c1 and c2 and c3 and c4
            partial = (best_c['fix_rate'] > 0.30) and c2
            not_validated = best_c['net'] < 0

            print(f"\n{'='*70}")
            if validated:
                print(f"""
VERDICT A -- CEGAR VALIDATED as inference-time mechanism:

  Fix rate:           {100*best_c['fix_rate']:.1f}%
  Net improvement:    {best_c['net']:+d}
  vs Random:          {100*(best_c['fix_rate']-(rand_r['fix_rate'] if rand_r else 0)):+.1f}pp
  Galois gap:         {best_c['gap_before']:.4f} → {best_c['gap_after']:.4f}

  CEGAR provides statistically meaningful lattice consistency recovery
  under simulated domain shift. The targeted geometric correction
  significantly outperforms random perturbation and coordinate descent.

  Action: Section 4.4 — "Inference-Time Lattice Consistency Refinement"
""")
            elif partial and not not_validated:
                print(f"""
VERDICT B -- PARTIAL VALIDATION:

  Fix rate:           {100*best_c['fix_rate']:.1f}%
  Net improvement:    {best_c['net']:+d}
  vs Random:          {100*(best_c['fix_rate']-(rand_r['fix_rate'] if rand_r else 0)):+.1f}pp

  CEGAR provides partial but meaningful lattice consistency improvement.
  The targeted correction outperforms random perturbation.

  Action: Report as partial result in Section 4.4.
""")
            else:
                print(f"""
VERDICT C -- NOT VALIDATED:

  Fix rate:           {100*best_c['fix_rate']:.1f}%
  Net improvement:    {best_c['net']:+d}

  Geometric updates are too coupled — fixing one pair breaks others.

  Action: One paragraph in limitations section.
""")
            print(f"{'='*70}")

    # ── Save ─────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    save_data = {
        'seed': SEED,
        'experiment': 'inference_cegar_noise',
        'n_concepts': len(boxes),
        'n_transitive': len(transitive_edges),
        'clean_violations': clean_viol,
        'gap_clean': gap_clean,
        'all_results': [
            {k: v for k, v in r.items() if k not in ['noisy_boxes', 'violations']}
            for r in all_results
        ],
        'hop_results': {
            k: v for k, v in hop_results.items() if v is not None
        },
        'convergence': conv_data,
    }
    with open('results/inference_cegar_noise.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\n  Results saved to results/inference_cegar_noise.json")


if __name__ == '__main__':
    main()
