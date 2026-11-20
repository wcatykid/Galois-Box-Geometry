"""
Lattice Join Experiment
========================
Tests whether lattice joins satisfy the predicted containment:
  join(α(s_specific1), α(s_specific2)) ⊑ α(s_general)

Uses the synthetic "Tom goes" hierarchy and the trained model.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.dataset import build_synthetic_embeddings, SENTENCES, ENTAILMENT_PAIRS
from src.box_lattice import (box_join, box_containment_violation,
                              box_volume, box_meet, box_width)
from src.improved_trainer import ImprovedTrainer


def load_model(seed=42):
    """Train a fresh model (same config as run_experiment.py)."""
    from src.dataset import build_synthetic_dataset
    train_data, val_data, test_data = build_synthetic_dataset(
        n_train=3000, n_val=500, n_test=1000,
        dim=64, noise=0.05, rng=np.random.default_rng(seed))

    trainer = ImprovedTrainer(
        input_dim=64, hidden_dim=256, box_dim=64, lr=1e-3,
        lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
        margin=1.0, scale=5.0, rng=np.random.default_rng(seed + 10))
    trainer.train(train_data, val_data, n_epochs=40,
                  batch_size=64, verbose=False, patience=10)
    return trainer.model


def main():
    print("=" * 70)
    print("  Lattice Join Experiment")
    print("=" * 70)

    # Build clean embeddings and train model
    print("\nTraining model...")
    model = load_model(seed=42)
    base_emb = build_synthetic_embeddings(dim=64, noise=0.0,
                                           rng=np.random.default_rng(0))

    # Encode all sentences
    boxes = {}
    for sid, text in SENTENCES.items():
        e = base_emb[sid][None, :]
        l, u, _ = model.encode(e)
        boxes[sid] = (l[0], u[0])

    # ── Test 1: Join of siblings ⊑ parent ──────────────────────────────
    # The hierarchy:
    #   s0: Something happens (most general)
    #   s1: Someone moves
    #   s2: Tom goes
    #   s3: Tom goes somewhere
    #   s4: Tom goes to school
    #   s5: Tom goes to East High School
    #   s6: Tom goes to West High School
    #   s7: Tom goes to Jefferson Elementary
    #   s8-s11: unrelated

    join_tests = [
        # (child1, child2, expected_parent, description)
        ('s5', 's6', 's4', 'join(East HS, West HS) ⊑ school'),
        ('s5', 's7', 's4', 'join(East HS, Jefferson) ⊑ school'),
        ('s6', 's7', 's4', 'join(West HS, Jefferson) ⊑ school'),
        ('s5', 's6', 's3', 'join(East HS, West HS) ⊑ somewhere'),
        ('s5', 's6', 's2', 'join(East HS, West HS) ⊑ Tom goes'),
        ('s5', 's6', 's1', 'join(East HS, West HS) ⊑ someone moves'),
        ('s5', 's6', 's0', 'join(East HS, West HS) ⊑ something happens'),
        ('s4', 's8', 's0', 'join(school, Mary reads) ⊑ something happens'),
        ('s2', 's9', 's0', 'join(Tom goes, cat sits) ⊑ something happens'),
        # Multi-way joins via pairwise
        ('s5', 's7', 's3', 'join(East HS, Jefferson) ⊑ somewhere'),
        ('s5', 's7', 's2', 'join(East HS, Jefferson) ⊑ Tom goes'),
        ('s6', 's7', 's2', 'join(West HS, Jefferson) ⊑ Tom goes'),
        # Tighter tests: join should NOT be contained in siblings
        ('s5', 's6', 's5', 'join(East HS, West HS) ⊑ East HS [should FAIL]'),
        ('s5', 's6', 's6', 'join(East HS, West HS) ⊑ West HS [should FAIL]'),
    ]

    print("\n" + "-" * 70)
    print("  Test 1: Join Containment — join(α(s1), α(s2)) ⊑ α(parent)?")
    print("-" * 70)
    print(f"  {'Description':<52} {'Viol':>8} {'Result':>8}")
    print("  " + "─" * 68)

    pass_count = 0
    total_expected_pass = 0
    for c1, c2, parent, desc in join_tests:
        l1, u1 = boxes[c1]
        l2, u2 = boxes[c2]
        lj, uj = box_join(l1, u1, l2, u2)
        lp, up = boxes[parent]
        viol = float(box_containment_violation(lj, uj, lp, up))

        should_fail = '[should FAIL]' in desc
        if not should_fail:
            total_expected_pass += 1

        threshold = 1.0
        contained = viol < threshold
        if should_fail:
            status = "FAIL ✓" if not contained else "FAIL ✗"
        else:
            status = "PASS ✓" if contained else "FAIL ✗"
            if contained:
                pass_count += 1

        print(f"  {desc:<52} {viol:>8.3f} {status:>8}")

    print(f"\n  Join containment: {pass_count}/{total_expected_pass} "
          f"({100*pass_count/total_expected_pass:.1f}%) passed")

    # ── Test 2: Join is the LEAST upper bound ──────────────────────────
    print("\n" + "-" * 70)
    print("  Test 2: Join Minimality — join should be tighter than parent")
    print("-" * 70)
    print(f"  {'Pair':<30} {'Join vol':>10} {'Parent vol':>12} {'Tighter?':>10}")
    print("  " + "─" * 62)

    tighter_count = 0
    minimality_tests = [
        ('s5', 's6', 's4', 'East+West → school'),
        ('s5', 's7', 's4', 'East+Jeff → school'),
        ('s4', 's8', 's0', 'school+Mary → sth happens'),
    ]
    for c1, c2, parent, label in minimality_tests:
        l1, u1 = boxes[c1]
        l2, u2 = boxes[c2]
        lj, uj = box_join(l1, u1, l2, u2)
        lp, up = boxes[parent]

        join_vol = float(np.sum(np.log(np.maximum(uj - lj, 1e-8))))
        parent_vol = float(np.sum(np.log(np.maximum(up - lp, 1e-8))))
        tighter = join_vol <= parent_vol
        if tighter:
            tighter_count += 1
        print(f"  {label:<30} {join_vol:>10.2f} {parent_vol:>12.2f} "
              f"{'YES ✓':>10}" if tighter else
              f"  {label:<30} {join_vol:>10.2f} {parent_vol:>12.2f} "
              f"{'NO ✗':>10}")

    print(f"\n  Join tighter than parent: {tighter_count}/{len(minimality_tests)}")

    # ── Test 3: Individual containment in join ─────────────────────────
    print("\n" + "-" * 70)
    print("  Test 3: Operands ⊑ Join — α(child) ⊑ join(α(child), α(sibling))?")
    print("-" * 70)

    operand_tests = [
        ('s5', 's6'), ('s5', 's7'), ('s6', 's7'),
        ('s4', 's8'), ('s2', 's9'),
    ]
    op_pass = 0
    for c1, c2 in operand_tests:
        l1, u1 = boxes[c1]
        l2, u2 = boxes[c2]
        lj, uj = box_join(l1, u1, l2, u2)

        v1 = float(box_containment_violation(l1, u1, lj, uj))
        v2 = float(box_containment_violation(l2, u2, lj, uj))
        ok1 = v1 < 1e-6
        ok2 = v2 < 1e-6
        if ok1:
            op_pass += 1
        if ok2:
            op_pass += 1
        print(f"  {SENTENCES[c1]:<30} ⊑ join: viol={v1:.6f}  {'✓' if ok1 else '✗'}")
        print(f"  {SENTENCES[c2]:<30} ⊑ join: viol={v2:.6f}  {'✓' if ok2 else '✗'}")

    total_op = len(operand_tests) * 2
    print(f"\n  Operand ⊑ join: {op_pass}/{total_op} "
          f"({100*op_pass/total_op:.1f}%) — should be 100% by construction")

    # ── Test 4: Three-way join ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Test 4: Three-way join — join(East, West, Jefferson) ⊑ school?")
    print("-" * 70)

    l5, u5 = boxes['s5']
    l6, u6 = boxes['s6']
    l7, u7 = boxes['s7']
    lj12, uj12 = box_join(l5, u5, l6, u6)
    lj123, uj123 = box_join(lj12, uj12, l7, u7)

    for parent_id in ['s4', 's3', 's2', 's1', 's0']:
        lp, up = boxes[parent_id]
        viol = float(box_containment_violation(lj123, uj123, lp, up))
        ok = viol < 1.0
        print(f"  join(East,West,Jeff) ⊑ {SENTENCES[parent_id]:<28} "
              f"viol={viol:.3f}  {'✓' if ok else '✗'}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Join containment (join ⊑ parent):     {pass_count}/{total_expected_pass}")
    print(f"  Operand containment (child ⊑ join):   {op_pass}/{total_op}")
    print(f"  Join minimality (tighter than parent): {tighter_count}/{len(minimality_tests)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
