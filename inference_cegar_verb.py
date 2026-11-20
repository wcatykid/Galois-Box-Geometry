"""
inference_cegar_verb.py — Inference-Time CEGAR on Verb IS-A Hierarchy
=====================================================================
Domain shift experiment: noun-trained box model applied to verb IS-A
relationships. CEGAR detects violated transitive IS-A relationships
and applies targeted geometric corrections WITHOUT retraining.

Usage
-----
  PYTHONIOENCODING=utf-8 python -u inference_cegar_verb.py
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

# Reuse from previous inference CEGAR experiments
from inference_cegar import (
    detect_violations,
    inference_cegar_update,
    apply_inference_cegar,
    measure_side_effects,
    random_perturbation_baseline,
    measure_galois_consistency_gap,
)
from inference_cegar_noise import coordinate_descent_baseline

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: Build Verb Concept Set
# ══════════════════════════════════════════════════════════════════════════════

def build_verb_concept_set(glove_vocab, n_concepts=5000,
                           min_depth=2, max_depth=8, rng=None):
    """
    Select verb synsets for evaluation, stratified by depth.
    Use wide depth range to maximize transitive edges.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    candidates_by_depth = {d: [] for d in range(min_depth, max_depth + 1)}

    for synset in wn.all_synsets('v'):
        depth = synset.min_depth()
        if depth < min_depth or depth > max_depth:
            continue
        lemma = synset.lemmas()[0].name().replace('_', ' ')
        words = lemma.split()
        if all(w in glove_vocab for w in words):
            candidates_by_depth[depth].append((synset, lemma, depth))

    total_available = sum(len(v) for v in candidates_by_depth.values())
    selected = []
    for depth, cands in candidates_by_depth.items():
        n_sample = int(n_concepts * len(cands) / max(total_available, 1))
        n_sample = min(n_sample, len(cands))
        if cands and n_sample > 0:
            idx = rng.choice(len(cands), size=n_sample, replace=False)
            selected.extend([cands[i] for i in idx])

    print(f"      Verb concept set: {len(selected)} concepts")
    for d in range(min_depth, max_depth + 1):
        n = sum(1 for c in selected if c[2] == d)
        print(f"        Depth {d}: {n}")

    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Encode Verb Concepts
# ══════════════════════════════════════════════════════════════════════════════

def encode_verb_concepts(verb_lemmas, model, glove):
    """Encode verb concepts using noun-trained box model. Bare word encoding."""
    boxes = {}
    embeddings = {}
    skipped = []
    glove_vocab = set(glove.key_to_index.keys())

    for lemma in verb_lemmas:
        words = lemma.split()
        word_vecs = [glove[w] for w in words if w in glove_vocab]
        if not word_vecs:
            skipped.append(lemma)
            continue

        emb = np.mean(word_vecs, axis=0).astype(np.float64)
        l, u, _ = model.encode(emb[None, :])
        boxes[lemma] = (l, u)
        embeddings[lemma] = emb

    print(f"      Encoded: {len(boxes)} verb concepts, skipped: {len(skipped)}")
    return boxes, embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Build Verb IS-A Graph
# ══════════════════════════════════════════════════════════════════════════════

def build_verb_isa_graph(verb_concept_set, verb_boxes):
    """Extract IS-A edges within the verb concept set from WordNet."""
    synset_to_lemma = {}
    for synset, lemma, depth in verb_concept_set:
        if lemma in verb_boxes:
            synset_to_lemma[synset] = lemma
    available_synsets = set(synset_to_lemma.keys())

    direct_edges = set()
    transitive_edges = {}

    for synset in available_synsets:
        lemma = synset_to_lemma[synset]

        # Direct edges
        for hypernym in synset.hypernyms():
            if hypernym in available_synsets:
                hyper_lemma = synset_to_lemma[hypernym]
                direct_edges.add((lemma, hyper_lemma))

        # Transitive edges
        for path in synset.hypernym_paths():
            for ancestor in path:
                if ancestor in available_synsets and ancestor != synset:
                    anc_lemma = synset_to_lemma[ancestor]
                    hop_count = len(path) - 1 - path.index(ancestor)
                    key = (lemma, anc_lemma)
                    if key not in transitive_edges or \
                       transitive_edges[key] > hop_count:
                        transitive_edges[key] = hop_count

    print(f"\n      Verb IS-A graph:")
    print(f"        Direct edges:     {len(direct_edges)}")
    print(f"        Transitive edges: {len(transitive_edges)}")

    # Violation stats by hop
    by_hop = {}
    for (hypo, hyper), hops in transitive_edges.items():
        if hypo not in verb_boxes or hyper not in verb_boxes:
            continue
        l1, u1 = verb_boxes[hypo]
        l2, u2 = verb_boxes[hyper]
        v = float(box_containment_violation(l1, u1, l2, u2))
        bucket = str(hops) if hops <= 3 else '4+'
        by_hop.setdefault(bucket, []).append(v)

    print(f"\n        Violation rate by hop (threshold=0.5):")
    for bucket in ['1', '2', '3', '4+']:
        viols = by_hop.get(bucket, [])
        if not viols:
            continue
        viol_rate = sum(1 for v in viols if v > 0.5) / len(viols)
        print(f"          Hop {bucket}: {len(viols)} pairs, "
              f"{100*viol_rate:.1f}% violated, "
              f"mean={np.mean(viols):.3f}, median={np.median(viols):.3f}")

    return direct_edges, transitive_edges


# ══════════════════════════════════════════════════════════════════════════════
#  Measurement helper
# ══════════════════════════════════════════════════════════════════════════════

def measure_state(boxes, transitive_edges, threshold=0.5):
    """Count violations and compute stats."""
    viols = []
    for (hypo, hyper) in transitive_edges:
        if hypo not in boxes or hyper not in boxes:
            continue
        l1, u1 = boxes[hypo]
        l2, u2 = boxes[hyper]
        v = float(box_containment_violation(l1, u1, l2, u2))
        viols.append(v)
    violated = sum(1 for v in viols if v > threshold)
    return {
        'n_pairs':     len(viols),
        'n_violated':  violated,
        'viol_rate':   violated / max(len(viols), 1),
        'mean_viol':   float(np.mean(viols)) if viols else 0,
        'median_viol': float(np.median(viols)) if viols else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*90}")
    print(f"  Inference-Time CEGAR on Verb IS-A Hierarchy (Domain Shift)")
    print(f"{'='*90}")
    print(f"  Noun-trained model applied to verb hypernymy — genuine domain shift.")
    print(f"  NO RETRAINING. NO GRADIENTS. Pure geometric refinement.")
    print(f"  Seed: {SEED}")

    # ── 1. Load GloVe ────────────────────────────────────────────────────
    print(f"\n[1/7] Loading GloVe...")
    t0 = time.time()
    glove = api.load('glove-wiki-gigaword-300')
    glove_vocab = set(glove.key_to_index.keys())
    print(f"      Loaded in {time.time()-t0:.0f}s  vocab={len(glove_vocab)}")

    # ── 2. Load noun-trained model ───────────────────────────────────────
    print(f"\n[2/7] Loading noun-trained box encoder...")
    model_cache = Path('results/inference_cegar_model.npz')
    if not model_cache.exists():
        print(f"      ERROR: No cached model at {model_cache}")
        print(f"      Run inference_cegar.py first to train and cache the model.")
        return

    data = np.load(model_cache, allow_pickle=True)
    rng = np.random.default_rng(SEED)
    model = GaloisModel(300, HIDDEN_DIM, BOX_DIM, rng)
    for key in model.params_dict():
        setattr(model, key, data[key])
    print(f"      Model loaded from cache.")

    # ── 3. Build verb concept set ────────────────────────────────────────
    print(f"\n[3/7] Building verb concept set...")
    # Use ALL available verb concepts (no subsampling) to maximize
    # transitive edges. Verbs are shallower than nouns, so we need
    # more concepts to get enough transitive pairs.
    verb_concept_set = build_verb_concept_set(
        glove_vocab, n_concepts=5000, min_depth=2, max_depth=8,
        rng=np.random.default_rng(SEED))
    verb_lemmas = [c[1] for c in verb_concept_set]

    # ── 4. Encode verb concepts ──────────────────────────────────────────
    print(f"\n[4/7] Encoding verb concepts with noun-trained model...")
    verb_boxes, verb_embeddings = encode_verb_concepts(
        verb_lemmas, model, glove)

    # ── 5. Build verb IS-A graph ─────────────────────────────────────────
    print(f"\n[5/7] Building verb IS-A graph...")
    verb_direct, verb_transitive = build_verb_isa_graph(
        verb_concept_set, verb_boxes)

    # ── Pre-flight checks ────────────────────────────────────────────────
    print(f"\n      Pre-flight checks:")
    state_before = measure_state(verb_boxes, verb_transitive)

    check1 = len(verb_transitive) >= 80
    check2 = state_before['n_violated'] >= 20
    check3 = state_before['mean_viol'] < 3.0

    print(f"        [{'PASS' if check1 else 'FAIL'}] "
          f"Transitive pairs >= 80: {len(verb_transitive)}")
    print(f"        [{'PASS' if check2 else 'FAIL'}] "
          f"Violations >= 20: {state_before['n_violated']}")
    print(f"        [{'PASS' if check3 else 'FAIL'}] "
          f"Mean violation < 3.0: {state_before['mean_viol']:.4f}")

    if not (check1 and check2 and check3):
        print(f"\n      PRE-FLIGHT FAILED. Aborting.")
        if not check1:
            print(f"      Not enough transitive pairs. Need larger concept set.")
        if not check2:
            print(f"      Not enough violations. Model generalizes well to verbs.")
        if not check3:
            print(f"      Violations too large for local correction.")
        return

    print(f"\n      Pre-flight PASSED. Running experiment.")
    print(f"      Before CEGAR:")
    print(f"        Pairs:     {state_before['n_pairs']}")
    print(f"        Violated:  {state_before['n_violated']} "
          f"({100*state_before['viol_rate']:.1f}%)")
    print(f"        Mean viol: {state_before['mean_viol']:.4f}")
    print(f"        Median:    {state_before['median_viol']:.4f}")

    # ── 6. Detect violations ─────────────────────────────────────────────
    print(f"\n[6/7] Detecting violations and running CEGAR + baselines...")

    # Use containment_threshold=0.5, violation_threshold=0.5
    # to capture all meaningfully violated pairs
    verb_violations, verb_violation_pairs = detect_violations(
        verb_boxes, verb_direct, verb_transitive,
        containment_threshold=0.5,
        violation_threshold=0.5)

    print(f"\n      Violations to fix: {len(verb_violations)}")
    for h in ['1', '2', '3', '4+']:
        n = sum(1 for v in verb_violations
                if (str(v['hop_count']) if v['hop_count'] <= 3 else '4+') == h)
        if n > 0:
            print(f"        Hop {h}: {n}")

    # ── Galois gap before ────────────────────────────────────────────────
    sample_pairs = list(verb_transitive.keys())
    gap_before = measure_galois_consistency_gap(
        verb_boxes, verb_embeddings, model, sample_pairs)
    print(f"\n      Galois gap before: mean={gap_before['mean']:.4f}  "
          f"p95={gap_before['p95']:.4f}")

    # ── Baseline 1: Random perturbation ──────────────────────────────────
    print(f"\n      Running random perturbation baseline...")
    random_boxes = random_perturbation_baseline(
        verb_boxes, verb_violations, step_size=0.3,
        rng=np.random.default_rng(99))
    state_random = measure_state(random_boxes, verb_transitive)
    _, _, rand_new, rand_fixed = measure_side_effects(
        verb_boxes, random_boxes, verb_direct, verb_transitive)
    gap_random = measure_galois_consistency_gap(
        random_boxes, verb_embeddings, model, sample_pairs)
    print(f"        Fixed={len(rand_fixed)}, New={len(rand_new)}, "
          f"Net={len(rand_fixed)-len(rand_new):+d}")

    # ── Baseline 2: Coordinate descent ───────────────────────────────────
    print(f"\n      Running coordinate descent baseline...")
    cd_boxes = coordinate_descent_baseline(
        verb_boxes, verb_transitive, step_size=0.3, max_iterations=15)
    state_cd = measure_state(cd_boxes, verb_transitive)
    _, _, cd_new, cd_fixed = measure_side_effects(
        verb_boxes, cd_boxes, verb_direct, verb_transitive)
    gap_cd = measure_galois_consistency_gap(
        cd_boxes, verb_embeddings, model, sample_pairs)
    print(f"        Fixed={len(cd_fixed)}, New={len(cd_new)}, "
          f"Net={len(cd_fixed)-len(cd_new):+d}")

    # ── CEGAR at three step sizes ────────────────────────────────────────
    step_sizes = [0.1, 0.3, 0.5]
    max_iterations = 15
    cegar_results = {}

    for ss in step_sizes:
        print(f"\n      Running CEGAR (step={ss}, max_iter={max_iterations})...")
        refined, fix_results = apply_inference_cegar(
            verb_boxes, verb_violations,
            step_size=ss, max_iterations=max_iterations)

        state_after = measure_state(refined, verb_transitive)
        _, _, new_viol, fixed = measure_side_effects(
            verb_boxes, refined, verb_direct, verb_transitive)

        gap_after = measure_galois_consistency_gap(
            refined, verb_embeddings, model, sample_pairs)

        cegar_results[ss] = {
            'state_after':  state_after,
            'fixed':        len(fixed),
            'new_viol':     len(new_viol),
            'net':          len(fixed) - len(new_viol),
            'fix_rate':     len(fixed) / max(state_before['n_violated'], 1),
            'gap_before':   gap_before['mean'],
            'gap_after':    gap_after['mean'],
            'gap_p95_after': gap_after['p95'],
            'fix_results':  fix_results,
        }

        print(f"        Fixed={len(fixed)}/{state_before['n_violated']} "
              f"({100*len(fixed)/max(state_before['n_violated'],1):.1f}%), "
              f"New={len(new_viol)}, Net={len(fixed)-len(new_viol):+d}, "
              f"Gap: {gap_before['mean']:.4f}→{gap_after['mean']:.4f}")

    # ── Best step size ───────────────────────────────────────────────────
    best_ss = max(cegar_results, key=lambda k: cegar_results[k]['net'])
    best = cegar_results[best_ss]
    print(f"\n      Best step size: {best_ss} "
          f"(net={best['net']:+d}, fix_rate={100*best['fix_rate']:.1f}%)")

    # ── 7. Convergence curve ─────────────────────────────────────────────
    print(f"\n[7/7] Convergence curve (step={best_ss})...")
    convergence = []
    for n_iter in [0, 1, 2, 3, 5, 8, 10, 15]:
        if n_iter == 0:
            current = verb_boxes
        else:
            current, _ = apply_inference_cegar(
                verb_boxes, verb_violations,
                step_size=best_ss, max_iterations=n_iter)
        st = measure_state(current, verb_transitive)
        gp = measure_galois_consistency_gap(
            current, verb_embeddings, model, sample_pairs)
        convergence.append({
            'iter': n_iter, **st, 'gap': gp['mean'],
        })
        print(f"      iter={n_iter:>2}: violated={st['n_violated']:>4} "
              f"({100*st['viol_rate']:>5.1f}%)  "
              f"mean_viol={st['mean_viol']:.4f}  gap={gp['mean']:.4f}")

    # ── Hop-stratified analysis ──────────────────────────────────────────
    print(f"\n      Hop-stratified analysis (step={best_ss})...")
    hop_buckets = {'1': [], '2': [], '3': [], '4+': []}
    for v in verb_violations:
        h = str(v['hop_count']) if v['hop_count'] <= 3 else '4+'
        hop_buckets[h].append(v)

    hop_results = {}
    for bucket in ['1', '2', '3', '4+']:
        viols = hop_buckets[bucket]
        if not viols:
            hop_results[bucket] = None
            continue

        refined_h, _ = apply_inference_cegar(
            verb_boxes, viols,
            step_size=best_ss, max_iterations=max_iterations)

        _, _, new_h, fixed_h = measure_side_effects(
            verb_boxes, refined_h, verb_direct, verb_transitive)

        mean_v = np.mean([v['violation'] for v in viols])
        hop_results[bucket] = {
            'n':        len(viols),
            'fixed':    len(fixed_h),
            'new_viol': len(new_h),
            'net':      len(fixed_h) - len(new_h),
            'fix_rate': len(fixed_h) / max(len(viols), 1),
            'mean_viol': float(mean_v),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS TABLES
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*100}")
    print(f"  TABLE 1: Main Results — Verb Domain Shift")
    print(f"{'='*100}")

    hdr = (f"  {'Method':<24} {'Step':>5} {'Violated':>9} {'Fixed':>7} "
           f"{'Fix%':>7} {'New':>5} {'Net':>6} "
           f"{'Gap Before':>11} {'Gap After':>10} {'Gap Δ':>8}")
    print(f"\n{hdr}")
    print(f"  {'-'*96}")

    # No refinement
    print(f"  {'No refinement':<24} {'—':>5} "
          f"{state_before['n_violated']:>9} {'0':>7} {'0.0%':>7} "
          f"{'0':>5} {'+0':>6} "
          f"{gap_before['mean']:>11.4f} {'—':>10} {'—':>8}")

    # Random
    print(f"  {'Random perturbation':<24} {'0.3':>5} "
          f"{state_before['n_violated']:>9} {len(rand_fixed):>7} "
          f"{100*len(rand_fixed)/max(state_before['n_violated'],1):>6.1f}% "
          f"{len(rand_new):>5} {len(rand_fixed)-len(rand_new):>+6} "
          f"{gap_before['mean']:>11.4f} {gap_random['mean']:>10.4f} "
          f"{gap_random['mean']-gap_before['mean']:>+8.4f}")

    # Coordinate descent
    print(f"  {'Coordinate descent':<24} {'0.3':>5} "
          f"{state_before['n_violated']:>9} {len(cd_fixed):>7} "
          f"{100*len(cd_fixed)/max(state_before['n_violated'],1):>6.1f}% "
          f"{len(cd_new):>5} {len(cd_fixed)-len(cd_new):>+6} "
          f"{gap_before['mean']:>11.4f} {gap_cd['mean']:>10.4f} "
          f"{gap_cd['mean']-gap_before['mean']:>+8.4f}")

    # CEGAR at each step size
    for ss in step_sizes:
        cr = cegar_results[ss]
        marker = " ← BEST" if ss == best_ss else ""
        print(f"  {'CEGAR':<24} {ss:>5} "
              f"{state_before['n_violated']:>9} {cr['fixed']:>7} "
              f"{100*cr['fix_rate']:>6.1f}% "
              f"{cr['new_viol']:>5} {cr['net']:>+6} "
              f"{cr['gap_before']:>11.4f} {cr['gap_after']:>10.4f} "
              f"{cr['gap_after']-cr['gap_before']:>+8.4f}{marker}")

    # ── Table 2: Hop-Stratified ──────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  TABLE 2: Hop-Stratified Results (step={best_ss})")
    print(f"{'='*80}")

    print(f"\n  {'Hop':>4} {'N':>6} {'Fixed':>7} {'Fix%':>7} "
          f"{'New':>5} {'Net':>6} {'Mean Viol':>10}")
    print(f"  {'-'*50}")

    total_n = total_fixed = total_new = 0
    for bucket in ['1', '2', '3', '4+']:
        hr = hop_results.get(bucket)
        if hr is None:
            continue
        total_n += hr['n']
        total_fixed += hr['fixed']
        total_new += hr['new_viol']
        print(f"  {bucket:>4} {hr['n']:>6} {hr['fixed']:>7} "
              f"{100*hr['fix_rate']:>6.1f}% "
              f"{hr['new_viol']:>5} {hr['net']:>+6} "
              f"{hr['mean_viol']:>10.4f}")

    print(f"  {'-'*50}")
    total_fix_rate = total_fixed / max(total_n, 1)
    print(f"  {'ALL':>4} {total_n:>6} {total_fixed:>7} "
          f"{100*total_fix_rate:>6.1f}% "
          f"{total_new:>5} {total_fixed-total_new:>+6} "
          f"{state_before['mean_viol']:>10.4f}")

    # ── Table 3: Convergence ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  TABLE 3: Convergence (step={best_ss})")
    print(f"{'='*70}")

    print(f"\n  {'Iter':>5} {'Violated':>10} {'Viol%':>8} "
          f"{'Mean Viol':>10} {'Galois Gap':>11}")
    print(f"  {'-'*50}")

    for c in convergence:
        label = '0 (verb)' if c['iter'] == 0 else str(c['iter'])
        print(f"  {label:>8} {c['n_violated']:>10} "
              f"{100*c['viol_rate']:>7.1f}% "
              f"{c['mean_viol']:>10.4f} {c['gap']:>11.4f}")

    print(f"  {'-'*50}")
    print(f"  {'noun model':>8} {'3':>10} {'0.7%':>8} "
          f"{'0.0430':>10} {'0.8002':>11}")

    # ── Table 4: Summary ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  TABLE 4: CEGAR vs Baselines — Summary")
    print(f"{'='*70}")

    best_fix_pct = 100 * best['fix_rate']
    rand_fix_pct = 100 * len(rand_fixed) / max(state_before['n_violated'], 1)
    cd_fix_pct = 100 * len(cd_fixed) / max(state_before['n_violated'], 1)

    print(f"\n  {'Method':<24} {'Fix%':>7} {'New Viol':>9} "
          f"{'Net':>6} {'Gap Δ':>8}")
    print(f"  {'-'*55}")
    print(f"  {'No refinement':<24} {'0.0%':>7} {'0':>9} {'+0':>6} "
          f"{'+0.000':>8}")
    print(f"  {'Random perturbation':<24} {rand_fix_pct:>6.1f}% "
          f"{len(rand_new):>9} {len(rand_fixed)-len(rand_new):>+6} "
          f"{gap_random['mean']-gap_before['mean']:>+8.4f}")
    print(f"  {'Coordinate descent':<24} {cd_fix_pct:>6.1f}% "
          f"{len(cd_new):>9} {len(cd_fixed)-len(cd_new):>+6} "
          f"{gap_cd['mean']-gap_before['mean']:>+8.4f}")
    print(f"  {'CEGAR (best)':<24} {best_fix_pct:>6.1f}% "
          f"{best['new_viol']:>9} {best['net']:>+6} "
          f"{best['gap_after']-best['gap_before']:>+8.4f}")
    print(f"  {'-'*55}")
    print(f"  CEGAR vs Random:       {best_fix_pct - rand_fix_pct:>+.1f}pp")
    print(f"  CEGAR vs Coord desc:   {best_fix_pct - cd_fix_pct:>+.1f}pp")

    # ── Theorem 1 ────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  Theorem 1 Empirical Verification")
    print(f"{'='*70}")
    print(f"\n  Galois gap before CEGAR: {gap_before['mean']:.4f} "
          f"(p95={gap_before['p95']:.4f})")
    print(f"  Galois gap after CEGAR:  {best['gap_after']:.4f}")
    gap_delta = best['gap_after'] - gap_before['mean']
    tightens = gap_delta < 0
    print(f"  CEGAR tightens gap: {tightens}")
    print(f"  Gap change: {gap_delta:+.4f} "
          f"({100*gap_delta/max(abs(gap_before['mean']),1e-8):+.1f}%)")

    # ── Decision Criteria ────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  Decision Criteria")
    print(f"{'='*70}")

    c1 = best['fix_rate'] > 0.40
    c2 = best['net'] > 0
    c3 = (best_fix_pct - rand_fix_pct) >= 20
    c4 = (best_fix_pct - cd_fix_pct) >= 10
    c6 = tightens

    # c5: fix rate increases with hop count
    hop_fix_rates = {}
    for bucket in ['1', '2', '3', '4+']:
        hr = hop_results.get(bucket)
        if hr and hr['n'] > 0:
            hop_fix_rates[bucket] = hr['fix_rate']
    c5 = (hop_fix_rates.get('4+', 0) > hop_fix_rates.get('1', 0)
          if '4+' in hop_fix_rates and '1' in hop_fix_rates else False)

    print(f"\n  [{'PASS' if c1 else 'FAIL'}] Fix rate > 40%: "
          f"{100*best['fix_rate']:.1f}%")
    print(f"  [{'PASS' if c2 else 'FAIL'}] Net improvement positive: "
          f"{best['net']:+d}")
    print(f"  [{'PASS' if c3 else 'FAIL'}] CEGAR > random by ≥20pp: "
          f"{best_fix_pct - rand_fix_pct:+.1f}pp")
    print(f"  [{'PASS' if c4 else 'FAIL'}] CEGAR > coord descent by ≥10pp: "
          f"{best_fix_pct - cd_fix_pct:+.1f}pp")
    print(f"  [{'PASS' if c5 else 'FAIL'}] Fix rate 4+ > 1: "
          f"{100*hop_fix_rates.get('4+',0):.1f}% vs "
          f"{100*hop_fix_rates.get('1',0):.1f}%")
    print(f"  [{'PASS' if c6 else 'FAIL'}] Galois gap decreases: "
          f"{gap_before['mean']:.4f} → {best['gap_after']:.4f}")

    n_pass = sum([c1, c2, c3, c4, c5, c6])
    validated = c1 and c2 and c3 and c6
    partial = (best['fix_rate'] >= 0.20 and best['net'] > 0
               and not validated)
    not_valid = best['net'] <= 0

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    if validated:
        print(f"""
VERDICT A -- CEGAR VALIDATED as inference-time domain adaptation mechanism.

  Fix rate:          {100*best['fix_rate']:.1f}%
  Net improvement:   {best['net']:+d} pairs
  vs Random:         {best_fix_pct - rand_fix_pct:+.1f}pp
  vs Coord descent:  {best_fix_pct - cd_fix_pct:+.1f}pp
  Galois gap:        {gap_before['mean']:.4f} → {best['gap_after']:.4f} ({gap_delta:+.4f})

  The noun-trained box model, when applied to verb IS-A relationships,
  exhibits {100*state_before['viol_rate']:.1f}% transitivity violations. Inference-time CEGAR
  recovers {100*best['fix_rate']:.1f}% of violated pairs in {max_iterations} iterations using only
  geometric box corrections — no retraining, no labeled verb data.

  Action: Section 4.4 "Inference-Time Lattice Consistency Refinement"
  with full results tables.
""")
    elif partial:
        print(f"""
VERDICT B -- PARTIAL VALIDATION.

  Fix rate:          {100*best['fix_rate']:.1f}%
  Net improvement:   {best['net']:+d} pairs
  Galois gap:        {gap_before['mean']:.4f} → {best['gap_after']:.4f}

  CEGAR provides meaningful but incomplete consistency recovery
  under genuine domain shift (noun → verb).

  Action: Report honestly in Section 4.4 with caveats.
""")
    else:
        print(f"""
VERDICT C -- NOT VALIDATED.

  Fix rate:          {100*best['fix_rate']:.1f}%
  Net improvement:   {best['net']:+d} pairs

  Geometric updates too coupled for local correction.

  Action: One paragraph in limitations. Stop all CEGAR experiments.
""")
    print(f"{'='*90}")

    # ── Save ─────────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    save_data = {
        'seed': SEED,
        'experiment': 'inference_cegar_verb',
        'verdict': ('VALIDATED' if validated else
                    'PARTIAL' if partial else 'NOT_VALIDATED'),
        'n_concepts': len(verb_boxes),
        'n_direct': len(verb_direct),
        'n_transitive': len(verb_transitive),
        'before': state_before,
        'best_step_size': best_ss,
        'best_fix_rate': best['fix_rate'],
        'best_net': best['net'],
        'gap_before': gap_before['mean'],
        'gap_after': best['gap_after'],
        'cegar_results': {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'fix_results'}
            for k, v in cegar_results.items()
        },
        'baselines': {
            'random': {'fixed': len(rand_fixed), 'new': len(rand_new)},
            'coord_descent': {'fixed': len(cd_fixed), 'new': len(cd_new)},
        },
        'hop_results': {k: v for k, v in hop_results.items() if v is not None},
        'convergence': convergence,
        'criteria': {
            'fix_rate_40': c1, 'net_positive': c2,
            'vs_random_20pp': c3, 'vs_cd_10pp': c4,
            'hop_increasing': c5, 'gap_decreases': c6,
        },
    }
    with open('results/inference_cegar_verb.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"  Results saved to results/inference_cegar_verb.json")


if __name__ == '__main__':
    main()
