"""
Unit tests for the Galois NLP implementation.

Covers:
  - box_lattice: containment violation, score, join, meet, axioms
  - encoder: forward pass shape/range
  - losses: galois_loss, task_loss, reconstruction_loss, CEGAR buffer
  - dataset: generation shape, class balance, embeddings
  - metrics: NLI metrics on known predictions, AUC, STS metrics
  - improved_trainer: forward/backward shapes, full mini-training round-trip
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed=0):
    return np.random.default_rng(seed)


# ===========================================================================
# 1. box_lattice
# ===========================================================================

class TestBoxLattice(unittest.TestCase):

    def setUp(self):
        from src.box_lattice import (
            box_containment_violation, box_containment_score,
            box_join, box_meet, box_volume, box_midpoint, box_width,
            is_valid_box, check_reflexivity, check_antisymmetry, check_transitivity,
        )
        self.bv  = box_containment_violation
        self.bs  = box_containment_score
        self.bj  = box_join
        self.bm  = box_meet
        self.bvol = box_volume
        self.bmp = box_midpoint
        self.bw  = box_width
        self.iv  = is_valid_box
        self.ref = check_reflexivity
        self.anti= check_antisymmetry
        self.trans=check_transitivity

    # --- containment_violation ---

    def test_violation_zero_when_contained(self):
        """[0,4] contained in [-1,5] → violation = 0."""
        l1, u1 = np.array([[0., 0.]]), np.array([[4., 4.]])
        l2, u2 = np.array([[-1., -1.]]), np.array([[5., 5.]])
        viol = self.bv(l1, u1, l2, u2)
        np.testing.assert_allclose(viol, 0.0, atol=1e-8)

    def test_violation_positive_when_not_contained(self):
        """[0,6] not contained in [1,5]: lower and upper violations."""
        l1, u1 = np.array([[0., 0.]]), np.array([[6., 6.]])
        l2, u2 = np.array([[1., 1.]]), np.array([[5., 5.]])
        viol = self.bv(l1, u1, l2, u2)
        # lower_viol = max(0, 1-0)*2 = 2, upper_viol = max(0, 6-5)*2 = 2 => 4
        self.assertGreater(float(viol[0]), 0.0)

    def test_violation_batch(self):
        """Batch of 3 pairs."""
        N, D = 3, 4
        rng = _rng(1)
        l1 = rng.standard_normal((N, D))
        u1 = l1 + np.abs(rng.standard_normal((N, D)))
        l2 = rng.standard_normal((N, D))
        u2 = l2 + np.abs(rng.standard_normal((N, D)))
        viol = self.bv(l1, u1, l2, u2)
        self.assertEqual(viol.shape, (N,))
        self.assertTrue(np.all(viol >= 0.0))

    # --- containment_score ---

    def test_score_perfect_containment(self):
        """Perfect containment → score close to 1."""
        l1, u1 = np.array([[1., 1.]]), np.array([[3., 3.]])
        l2, u2 = np.array([[0., 0.]]), np.array([[4., 4.]])
        score = self.bs(l1, u1, l2, u2, scale=10.0)
        np.testing.assert_allclose(score, 1.0, atol=1e-7)

    def test_score_range(self):
        """Score always in [0, 1]."""
        rng = _rng(2)
        N, D = 50, 8
        l1 = rng.standard_normal((N, D)); u1 = l1 + np.abs(rng.standard_normal((N, D)))
        l2 = rng.standard_normal((N, D)); u2 = l2 + np.abs(rng.standard_normal((N, D)))
        sc = self.bs(l1, u1, l2, u2)
        self.assertTrue(np.all(sc >= 0.0) and np.all(sc <= 1.0))

    # --- join / meet ---

    def test_join_contains_both(self):
        """Join box must contain both input boxes."""
        l1, u1 = np.array([1., 2.]), np.array([3., 5.])
        l2, u2 = np.array([0., 3.]), np.array([4., 6.])
        jl, ju = self.bj(l1, u1, l2, u2)
        # join should have l=min(l1,l2), u=max(u1,u2)
        np.testing.assert_array_equal(jl, np.array([0., 2.]))
        np.testing.assert_array_equal(ju, np.array([4., 6.]))

    def test_meet_inside_both(self):
        """Meet box must be inside both input boxes."""
        l1, u1 = np.array([0., 0.]), np.array([4., 4.])
        l2, u2 = np.array([1., 1.]), np.array([3., 5.])
        ml, mu, valid = self.bm(l1, u1, l2, u2)
        self.assertTrue(bool(valid.all()))
        # meet lower = max(l1,l2), upper = min(u1,u2)
        np.testing.assert_array_equal(ml, np.array([1., 1.]))
        np.testing.assert_array_equal(mu, np.array([3., 4.]))

    def test_meet_empty(self):
        """Non-overlapping boxes → meet is invalid."""
        l1, u1 = np.array([0.]), np.array([1.])
        l2, u2 = np.array([2.]), np.array([3.])
        ml, mu, valid = self.bm(l1, u1, l2, u2)
        self.assertFalse(bool(valid.all()))

    # --- volume / midpoint / width ---

    def test_volume_positive(self):
        l, u = np.array([[0., 0.]]), np.array([[2., 2.]])
        vol = self.bvol(l, u)
        # vol = sum(log(widths)) = log(2) + log(2) ≈ 1.386
        np.testing.assert_allclose(vol[0], np.log(2)*2, atol=1e-5)

    def test_midpoint(self):
        l, u = np.array([0., 2.]), np.array([4., 6.])
        mid = self.bmp(l, u)
        np.testing.assert_allclose(mid, np.array([2., 4.]))

    def test_width(self):
        l, u = np.array([1., 3.]), np.array([4., 7.])
        w = self.bw(l, u)
        np.testing.assert_array_equal(w, np.array([3., 4.]))

    # --- is_valid_box ---

    def test_valid_box(self):
        l, u = np.array([[0., 0.]]), np.array([[1., 1.]])
        self.assertTrue(bool(self.iv(l, u).all()))

    def test_invalid_box(self):
        l, u = np.array([[2., 0.]]), np.array([[1., 1.]])
        self.assertFalse(bool(self.iv(l, u).all()))

    # --- lattice axioms ---

    def test_reflexivity(self):
        rng = _rng(3)
        N, D = 20, 8
        l = rng.standard_normal((N, D))
        u = l + np.abs(rng.standard_normal((N, D)))
        self.assertTrue(self.ref(l, u))

    def test_antisymmetry_equal_boxes(self):
        """Two identical boxes are mutually contained → antisymmetry holds."""
        l = np.array([[0., 1.]])
        u = np.array([[2., 3.]])
        self.assertTrue(self.anti(l, u, l, u))

    def test_transitivity(self):
        """Nested boxes: a ⊆ b ⊆ c → a ⊆ c."""
        l3, u3 = np.array([[0., 0.]]), np.array([[9., 9.]])
        l2, u2 = np.array([[1., 1.]]), np.array([[7., 7.]])
        l1, u1 = np.array([[2., 2.]]), np.array([[5., 5.]])
        self.assertTrue(self.trans(l1, u1, l2, u2, l3, u3))


# ===========================================================================
# 2. encoder (standalone functions)
# ===========================================================================

class TestEncoder(unittest.TestCase):

    def setUp(self):
        from src.encoder import init_encoder_params, encode, init_decoder_params, decode
        self.init_enc = init_encoder_params
        self.encode   = encode
        self.init_dec = init_decoder_params
        self.decode   = decode
        self.D, self.H, self.B = 32, 64, 10

    def test_encode_output_shapes(self):
        rng = _rng(10)
        params = self.init_enc(self.D, self.H, self.B, rng)
        x = rng.standard_normal((5, self.D))
        l, u, h = self.encode(x, params)
        self.assertEqual(l.shape, (5, self.B))
        self.assertEqual(u.shape, (5, self.B))
        self.assertEqual(h.shape, (5, self.H))

    def test_encode_l_leq_u(self):
        """Encoder must produce valid boxes: l ≤ u everywhere."""
        rng = _rng(11)
        params = self.init_enc(self.D, self.H, self.B, rng)
        x = rng.standard_normal((20, self.D))
        l, u, _ = self.encode(x, params)
        self.assertTrue(np.all(l <= u))

    def test_decode_output_shape(self):
        rng = _rng(12)
        params_dec = self.init_dec(self.B, self.H, self.D, rng)
        l = rng.standard_normal((5, self.B))
        u = l + np.abs(rng.standard_normal((5, self.B)))
        x_hat = self.decode(l, u, params_dec)
        self.assertEqual(x_hat.shape, (5, self.D))

    def test_encode_single_sample(self):
        """Single sample (1D input broadcast) should work."""
        rng = _rng(13)
        params = self.init_enc(self.D, self.H, self.B, rng)
        x = rng.standard_normal((1, self.D))
        l, u, h = self.encode(x, params)
        self.assertEqual(l.shape, (1, self.B))


# ===========================================================================
# 3. losses
# ===========================================================================

class TestLosses(unittest.TestCase):

    def setUp(self):
        from src.losses import (
            reconstruction_loss, galois_loss, task_loss,
            CEGARBuffer, cegar_loss, total_loss,
        )
        self.rec  = reconstruction_loss
        self.gal  = galois_loss
        self.tsk  = task_loss
        self.cegar_cls = CEGARBuffer
        self.cegar_loss = cegar_loss
        self.total = total_loss

    # --- reconstruction_loss ---

    def test_recon_zero_on_perfect(self):
        x = np.array([[1., 2., 3.]])
        self.assertAlmostEqual(self.rec(x, x), 0.0, places=10)

    def test_recon_positive(self):
        x     = np.ones((4, 8))
        x_hat = np.zeros((4, 8))
        self.assertGreater(self.rec(x, x_hat), 0.0)

    def test_recon_symmetric_in_expectation(self):
        """MSE(a,b) == MSE(b,a)."""
        rng = _rng(20)
        a = rng.standard_normal((10, 16))
        b = rng.standard_normal((10, 16))
        np.testing.assert_allclose(self.rec(a, b), self.rec(b, a))

    # --- galois_loss ---

    def test_galois_loss_zero_on_correct_entailment(self):
        """Perfect containment for entailment pairs → loss = 0."""
        N, D = 4, 8
        # l1 strictly inside l2 (l2<l1<u1<u2)
        l2 = np.zeros((N, D)) - 2.0
        u2 = np.zeros((N, D)) + 2.0
        l1 = np.zeros((N, D)) - 1.0
        u1 = np.zeros((N, D)) + 1.0
        labels = np.zeros(N, dtype=int)   # all entailment
        loss = self.gal(l1, u1, l2, u2, labels, margin=1.0)
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_galois_loss_positive_on_violation(self):
        """Entailment pair where box is NOT contained → loss > 0."""
        N, D = 2, 4
        l1 = np.ones((N, D)) * 5.0
        u1 = np.ones((N, D)) * 6.0
        l2 = np.zeros((N, D))
        u2 = np.ones((N, D)) * 3.0
        labels = np.zeros(N, dtype=int)   # entailment
        loss = self.gal(l1, u1, l2, u2, labels)
        self.assertGreater(loss, 0.0)

    def test_galois_hinge_direction_correct(self):
        """Non-entailment pairs with low violation → penalised (margin-viol > 0)."""
        N, D = 4, 4
        # boxes nearly identical → viol ≈ 0 << margin
        l1 = np.zeros((N, D))
        u1 = np.ones((N, D))
        l2 = l1.copy()
        u2 = u1.copy()
        labels = np.ones(N, dtype=int)    # neutral (non-entailment)
        loss = self.gal(l1, u1, l2, u2, labels, margin=2.0)
        self.assertGreater(loss, 0.0)

    # --- task_loss ---

    def test_task_loss_returns_tuple(self):
        N, D = 6, 4
        l1 = np.zeros((N, D)) - 1.; u1 = np.zeros((N, D)) + 1.
        l2 = np.zeros((N, D)) - 2.; u2 = np.zeros((N, D)) + 2.
        labels = np.array([0, 1, 0, 1, 2, 0])
        loss, probs = self.tsk(l1, u1, l2, u2, labels)
        self.assertIsInstance(loss, float)
        self.assertEqual(probs.shape, (N,))

    def test_task_loss_probs_in_range(self):
        rng = _rng(30)
        N, D = 20, 8
        l1 = rng.standard_normal((N, D)); u1 = l1 + np.abs(rng.standard_normal((N, D)))
        l2 = rng.standard_normal((N, D)); u2 = l2 + np.abs(rng.standard_normal((N, D)))
        labels = rng.integers(0, 3, N)
        _, probs = self.tsk(l1, u1, l2, u2, labels)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    # --- CEGAR buffer ---

    def test_cegar_buffer_add_and_sample(self):
        buf = self.cegar_cls(max_size=100)
        N, D = 10, 4
        rng = _rng(40)
        l1 = rng.standard_normal((N, D)); u1 = l1 + 1.
        l2 = rng.standard_normal((N, D)); u2 = l2 + 1.
        labels = np.zeros(N, dtype=int)
        preds  = np.ones(N, dtype=int)   # all wrong → all added
        buf.add(l1, u1, l2, u2, labels, preds)
        self.assertEqual(len(buf), N)
        sampled = buf.sample(5)
        self.assertIsNotNone(sampled)
        # 4 box arrays + 1 label array
        sl1, su1, sl2, su2, slab = sampled
        self.assertEqual(sl1.shape, (5, D))
        self.assertEqual(slab.shape, (5,))

    def test_cegar_buffer_max_size(self):
        buf = self.cegar_cls(max_size=5)
        N, D = 10, 2
        rng = _rng(41)
        l1 = rng.standard_normal((N, D)); u1 = l1 + 1.
        l2 = rng.standard_normal((N, D)); u2 = l2 + 1.
        labels = np.zeros(N, dtype=int)
        preds  = np.ones(N, dtype=int)
        buf.add(l1, u1, l2, u2, labels, preds)
        self.assertLessEqual(len(buf), 5)

    def test_cegar_empty_returns_none(self):
        buf = self.cegar_cls()
        self.assertIsNone(buf.sample(10))

    def test_cegar_buffer_no_add_when_correct(self):
        """Correctly predicted pairs should not be added."""
        buf = self.cegar_cls(max_size=100)
        N, D = 8, 4
        rng = _rng(42)
        l1 = rng.standard_normal((N, D)); u1 = l1 + 1.
        l2 = rng.standard_normal((N, D)); u2 = l2 + 1.
        labels = np.ones(N, dtype=int)
        preds  = np.ones(N, dtype=int)   # all correct → nothing added
        buf.add(l1, u1, l2, u2, labels, preds)
        self.assertEqual(len(buf), 0)

    def test_cegar_ring_buffer_wraps(self):
        """Ring buffer should overwrite oldest entries when full."""
        max_size, D = 6, 2
        buf = self.cegar_cls(max_size=max_size)
        rng = _rng(43)
        for _ in range(3):   # add 4 misclassified pairs three times = 12 total
            N = 4
            l1 = rng.standard_normal((N, D)); u1 = l1 + 1.
            l2 = rng.standard_normal((N, D)); u2 = l2 + 1.
            labels = np.zeros(N, dtype=int)
            preds  = np.ones(N, dtype=int)
            buf.add(l1, u1, l2, u2, labels, preds)
        self.assertEqual(len(buf), max_size)   # capped at max_size

    def test_cegar_buffer_clear(self):
        """clear() must reset size/head without deallocating arrays."""
        buf = self.cegar_cls(max_size=20)
        N, D = 8, 4
        rng = _rng(45)
        l1 = rng.standard_normal((N, D)); u1 = l1 + 1.
        l2 = rng.standard_normal((N, D)); u2 = l2 + 1.
        labels = np.zeros(N, dtype=int); preds = np.ones(N, dtype=int)
        buf.add(l1, u1, l2, u2, labels, preds)
        self.assertEqual(len(buf), N)
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertIsNone(buf.sample(5))
        # Arrays should still be allocated (reuse without realloc)
        self.assertIsNotNone(buf._l1)

    def test_cegar_label_convention_preserved(self):
        """Sampled labels must match what was stored (1=ent, 0=non-ent)."""
        buf = self.cegar_cls(max_size=50)
        D = 4
        rng = _rng(44)
        # 3 entailment (label=1) misclassified, 3 non-ent (label=0) misclassified
        N = 6
        l1 = rng.standard_normal((N, D)); u1 = l1 + 1.
        l2 = rng.standard_normal((N, D)); u2 = l2 + 1.
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=int)
        preds  = np.array([0, 0, 0, 1, 1, 1], dtype=int)  # all wrong
        buf.add(l1, u1, l2, u2, labels, preds)
        _, _, _, _, slab = buf.sample(6)
        self.assertEqual(set(slab.tolist()), {0, 1})   # both classes present

    # --- total_loss ---

    def test_total_loss_weighted_sum(self):
        val = self.total(1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(val, 10.0)

    def test_total_loss_zero_weight(self):
        val = self.total(1.0, 2.0, 3.0, 999.0, lambda1=1., lambda2=1., lambda3=1., lambda4=0.)
        self.assertAlmostEqual(val, 6.0)


# ===========================================================================
# 4. dataset
# ===========================================================================

class TestDataset(unittest.TestCase):

    def setUp(self):
        from src.dataset import (
            build_synthetic_embeddings, build_synthetic_dataset, build_synthetic_sts,
            NLIDataLoader, SENTENCES,
        )
        self.build_emb = build_synthetic_embeddings
        self.build_nli = build_synthetic_dataset
        self.build_sts = build_synthetic_sts
        self.DataLoader = NLIDataLoader
        self.sentences  = SENTENCES

    def test_embeddings_shape_and_count(self):
        emb = self.build_emb(dim=32, rng=_rng(0))
        self.assertEqual(len(emb), len(self.sentences))
        for k, v in emb.items():
            self.assertEqual(v.shape, (32,))

    def test_embeddings_normalised(self):
        emb = self.build_emb(dim=64, rng=_rng(1))
        for k, v in emb.items():
            np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_build_nli_shapes(self):
        train, val, test = self.build_nli(n_train=60, n_val=20, n_test=30,
                                          dim=16, rng=_rng(2))
        for split, n in [(train, 60), (val, 20), (test, 30)]:
            self.assertEqual(split['emb1'].shape, (n, 16))
            self.assertEqual(split['emb2'].shape, (n, 16))
            self.assertEqual(split['label'].shape, (n,))
            self.assertEqual(len(split['sent1']), n)
            self.assertEqual(len(split['sent2']), n)

    def test_build_nli_labels_in_range(self):
        train, _, _ = self.build_nli(n_train=60, dim=16, rng=_rng(3))
        self.assertTrue(np.all(train['label'] >= 0))
        self.assertTrue(np.all(train['label'] <= 2))

    def test_build_nli_class_balance(self):
        """Each split should have roughly equal label proportions."""
        train, _, _ = self.build_nli(n_train=300, dim=16, rng=_rng(4))
        counts = np.bincount(train['label'], minlength=3)
        # Each class should have at least 20% of samples
        for c in counts:
            self.assertGreater(c, 30)

    def test_build_sts_shape(self):
        sts = self.build_sts(n=50, dim=16, rng=_rng(5))
        self.assertEqual(sts['emb1'].shape, (50, 16))
        self.assertEqual(sts['score'].shape, (50,))

    def test_build_sts_score_range(self):
        sts = self.build_sts(n=100, dim=32, rng=_rng(6))
        self.assertTrue(np.all(sts['score'] >= 0.0))
        self.assertTrue(np.all(sts['score'] <= 5.0))

    def test_dataloader_batch_count(self):
        train, _, _ = self.build_nli(n_train=50, dim=8, rng=_rng(7))
        loader = self.DataLoader(train, batch_size=16, shuffle=False)
        batches = list(loader)
        total = sum(b['emb1'].shape[0] for b in batches)
        self.assertEqual(total, 50)

    def test_dataloader_shuffle(self):
        """Shuffled loader should produce same total but different order."""
        train, _, _ = self.build_nli(n_train=30, dim=8, rng=_rng(8))
        l1 = self.DataLoader(train, batch_size=30, shuffle=False)
        l2 = self.DataLoader(train, batch_size=30, shuffle=True, rng=_rng(99))
        b1 = list(l1)[0]
        b2 = list(l2)[0]
        # Same labels present (just reordered)
        np.testing.assert_array_equal(
            np.sort(b1['label']), np.sort(b2['label'])
        )


# ===========================================================================
# 5. metrics
# ===========================================================================

class TestMetrics(unittest.TestCase):

    def setUp(self):
        from src.metrics import (
            compute_nli_metrics, compute_sts_metrics, compute_box_stats,
            MajorityClassBaseline, CosineSimilarityBaseline,
            DotProductBaseline, LinearSVMBaseline, OrderEmbeddingBaseline,
        )
        self.nli   = compute_nli_metrics
        self.sts   = compute_sts_metrics
        self.boxes = compute_box_stats
        self.Maj   = MajorityClassBaseline
        self.Cos   = CosineSimilarityBaseline
        self.Dot   = DotProductBaseline
        self.SVM   = LinearSVMBaseline
        self.OE    = OrderEmbeddingBaseline

    # --- NLI metrics ---

    def test_perfect_classifier(self):
        """p_entail = 1 for all entailment, 0 for all others → acc=1, f1=1."""
        labels = np.array([0, 0, 1, 2, 1, 0])
        p      = np.array([1., 1., 0., 0., 0., 1.])
        m = self.nli(p, labels, threshold=0.5)
        self.assertAlmostEqual(m['accuracy'],  1.0, places=5)
        self.assertAlmostEqual(m['f1'],        1.0, places=5)

    def test_all_wrong_classifier(self):
        """Inverted predictions → low accuracy."""
        labels = np.array([0, 0, 1, 2])
        p      = np.array([0., 0., 1., 1.])
        m = self.nli(p, labels, threshold=0.5)
        self.assertLess(m['f1'], 0.5)

    def test_metrics_keys_present(self):
        labels = np.array([0, 1, 2, 0])
        p      = np.array([0.8, 0.2, 0.1, 0.9])
        m = self.nli(p, labels)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            self.assertIn(key, m)

    def test_auc_random_classifier(self):
        """Random classifier (p = 0.5 for all) → AUC ≈ 0.5."""
        rng = _rng(50)
        N = 200
        labels = rng.integers(0, 3, N)
        p = np.full(N, 0.5)
        m = self.nli(p, labels)
        self.assertAlmostEqual(m['auc'], 0.5, delta=0.05)

    def test_auc_perfect_classifier(self):
        """Perfect score → AUC = 1.0."""
        labels = np.array([0, 0, 1, 2, 1])
        p = np.array([1., 1., 0., 0., 0.])
        m = self.nli(p, labels)
        self.assertAlmostEqual(m['auc'], 1.0, places=5)

    # --- STS metrics ---

    def test_sts_perfect(self):
        pred = np.array([1., 2., 3., 4., 5.])
        true = pred.copy()
        m = self.sts(pred, true)
        self.assertAlmostEqual(m['pearson_r'], 1.0, places=5)
        self.assertAlmostEqual(m['spearman_rho'], 1.0, places=5)
        self.assertAlmostEqual(m['mse'], 0.0, places=10)

    # --- box stats ---

    def test_box_stats_keys(self):
        rng = _rng(51)
        N, D = 10, 8
        l1 = rng.standard_normal((N, D)); u1 = l1 + np.abs(rng.standard_normal((N, D)))
        l2 = rng.standard_normal((N, D)); u2 = l2 + np.abs(rng.standard_normal((N, D)))
        stats = self.boxes(l1, u1, l2, u2)
        for key in ['box_vol1_mean', 'box_width_mean', 'containment_viol_mean', 'zero_viol_rate']:
            self.assertIn(key, stats)

    # --- baselines ---

    def test_majority_baseline(self):
        from src.dataset import build_synthetic_dataset
        train, _, test = build_synthetic_dataset(n_train=90, dim=16, rng=_rng(60))
        m_obj = self.Maj().fit(train)
        metrics = m_obj.evaluate(test)
        self.assertIn('accuracy', metrics)
        # Majority class on balanced data → ~0.33
        self.assertGreater(metrics['accuracy'], 0.0)

    def test_cosine_baseline_fit_and_eval(self):
        from src.dataset import build_synthetic_dataset
        train, _, test = build_synthetic_dataset(n_train=90, dim=16, rng=_rng(61))
        m_obj = self.Cos().fit(train)
        metrics = m_obj.evaluate(test)
        self.assertIn('f1', metrics)

    def test_dot_baseline(self):
        from src.dataset import build_synthetic_dataset
        train, _, test = build_synthetic_dataset(n_train=90, dim=16, rng=_rng(62))
        m_obj = self.Dot().fit(train)
        metrics = m_obj.evaluate(test)
        self.assertIn('accuracy', metrics)

    def test_svm_baseline(self):
        from src.dataset import build_synthetic_dataset
        train, _, test = build_synthetic_dataset(n_train=90, dim=16, rng=_rng(63))
        m_obj = self.SVM(n_epochs=10).fit(train)
        metrics = m_obj.evaluate(test)
        self.assertIn('accuracy', metrics)

    def test_order_embedding_baseline(self):
        from src.dataset import build_synthetic_dataset
        train, _, test = build_synthetic_dataset(n_train=90, dim=16, rng=_rng(64))
        m_obj = self.OE(box_dim=16, n_epochs=10).fit(train)
        metrics = m_obj.evaluate(test)
        self.assertIn('accuracy', metrics)


# ===========================================================================
# 6. improved_trainer — GaloisModel & ImprovedTrainer
# ===========================================================================

class TestImprovedTrainer(unittest.TestCase):

    def _make_model(self, D=16, H=32, B=16, seed=0):
        from src.improved_trainer import GaloisModel
        return GaloisModel(D, H, B, _rng(seed))

    def _make_trainer(self, D=16, H=32, B=16, seed=0):
        from src.improved_trainer import ImprovedTrainer
        return ImprovedTrainer(
            input_dim=D, hidden_dim=H, box_dim=B, lr=1e-3,
            lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
            margin=1.0, scale=5.0, rng=_rng(seed),
        )

    # --- forward ---

    def test_encode_shapes(self):
        D, H, B = 16, 32, 8
        model = self._make_model(D, H, B)
        rng = _rng(70)
        x = rng.standard_normal((5, D))
        l, u, h = model.encode(x)
        self.assertEqual(l.shape, (5, B))
        self.assertEqual(u.shape, (5, B))

    def test_encode_valid_boxes(self):
        model = self._make_model()
        rng = _rng(71)
        x = rng.standard_normal((20, 16))
        l, u, _ = model.encode(x)
        self.assertTrue(np.all(l <= u))

    def test_decode_shape(self):
        D, H, B = 16, 32, 8
        model = self._make_model(D, H, B)
        rng = _rng(72)
        l = rng.standard_normal((4, B))
        u = l + np.abs(rng.standard_normal((4, B)))
        x_hat = model.decode(l, u)
        self.assertEqual(x_hat.shape, (4, D))

    # --- backward ---

    def test_backprop_encode_grad_shapes(self):
        D, H, B = 16, 32, 8
        model = self._make_model(D, H, B)
        rng = _rng(73)
        x = rng.standard_normal((4, D))
        l, u, h, cache = model._encode_fwd(x)
        dl = rng.standard_normal((4, B))
        du = rng.standard_normal((4, B))
        grads = model._backprop_encode(dl, du, cache)
        for key in ['W1', 'b1', 'W2', 'b2', 'Wc', 'bc', 'Ww', 'bw']:
            self.assertIn(key, grads)
            self.assertEqual(grads[key].shape, getattr(model, key).shape)

    def test_backprop_decode_grad_shapes(self):
        D, H, B = 16, 32, 8
        model = self._make_model(D, H, B)
        rng = _rng(74)
        l = rng.standard_normal((4, B)); u = l + 1.
        x_orig = rng.standard_normal((4, D))
        _, dcache = model._decode_fwd(l, u)
        grads, dl, du = model._backprop_decode(x_orig, l, u, dcache)
        for key in ['Wd1', 'bd1', 'Wd2', 'bd2']:
            self.assertIn(key, grads)
            self.assertEqual(grads[key].shape, getattr(model, key).shape)
        self.assertEqual(dl.shape, (4, B))
        self.assertEqual(du.shape, (4, B))

    # --- numerical gradient check on reconstruction loss ---

    def test_numerical_grad_decoder_Wd2(self):
        """Finite-difference check on dWd2 for reconstruction loss.

        _backprop_decode computes dL/dx_hat = 2*(x_hat-x_orig)/(N*D), which
        is the gradient of L = mean((x_hat - x_orig)^2)  (no ½ factor).
        The test loss must match this exact form.
        """
        D, H, B = 8, 16, 8
        model = self._make_model(D, H, B, seed=80)
        rng = _rng(81)
        l = rng.standard_normal((3, B)); u = l + 1.
        x_orig = rng.standard_normal((3, D))

        # Loss that _backprop_decode differentiates: mean((x_hat - x_orig)^2)
        def loss_fn(wd2_flat):
            model.Wd2 = wd2_flat.reshape(H, D)
            x_hat, _ = model._decode_fwd(l, u)
            return np.mean((x_hat - x_orig) ** 2)

        _, dcache = model._decode_fwd(l, u)
        grads, _, _ = model._backprop_decode(x_orig, l, u, dcache)
        analytic = grads['Wd2'].ravel()

        eps = 1e-5
        wd2_flat = model.Wd2.ravel().copy()
        num_grad = np.zeros_like(wd2_flat)
        for i in range(len(wd2_flat)):
            w_p = wd2_flat.copy(); w_p[i] += eps
            w_m = wd2_flat.copy(); w_m[i] -= eps
            num_grad[i] = (loss_fn(w_p) - loss_fn(w_m)) / (2 * eps)

        # Restore
        model.Wd2 = wd2_flat.reshape(H, D)
        rel_err = np.linalg.norm(analytic - num_grad) / (np.linalg.norm(analytic + num_grad) + 1e-8)
        self.assertLess(rel_err, 0.01, f"Relative gradient error too large: {rel_err:.4f}")

    # --- single training step ---

    def test_single_step_returns_losses(self):
        D = 16
        trainer = self._make_trainer(D=D, H=32, B=8)
        rng = _rng(90)
        emb1 = rng.standard_normal((8, D))
        emb2 = rng.standard_normal((8, D))
        labels = np.array([0, 0, 1, 2, 0, 1, 2, 1])
        loss_d = trainer._step(emb1, emb2, labels)
        for key in ['total', 'recon', 'galois', 'task']:
            self.assertIn(key, loss_d)
            self.assertFalse(np.isnan(loss_d[key]), f"NaN in {key}")

    def test_loss_decreases_over_training(self):
        """After 10 epochs on a small synthetic split, loss should decrease."""
        from src.dataset import build_synthetic_dataset
        D = 16
        trainer = self._make_trainer(D=D, H=32, B=16, seed=100)
        train, val, _ = build_synthetic_dataset(n_train=120, n_val=40, dim=D, rng=_rng(101))
        history = trainer.train(train, val, n_epochs=10, batch_size=32,
                                verbose=False, patience=20)
        # Total loss at epoch 10 should be ≤ epoch 1
        self.assertLessEqual(history[-1]['total'], history[0]['total'] * 2.0)

    def test_evaluate_returns_metrics(self):
        from src.dataset import build_synthetic_dataset
        D = 16
        trainer = self._make_trainer(D=D, H=32, B=16, seed=110)
        _, _, test = build_synthetic_dataset(n_train=60, n_val=20, n_test=40,
                                              dim=D, rng=_rng(111))
        metrics = trainer.evaluate(test)
        for key in ['accuracy', 'f1', 'auc']:
            self.assertIn(key, metrics)

    def test_params_dict_roundtrip(self):
        """Saving and restoring params should be lossless."""
        model = self._make_model()
        params = model.params_dict()
        # mutate one param
        original_W1 = params['W1'].copy()
        params['W1'] *= 0.0
        model.set_params(params)
        np.testing.assert_array_equal(model.W1, np.zeros_like(original_W1))

    # --- warmup + flush_every ---

    def test_warmup_suppresses_cegar(self):
        """During warmup epochs lambda4 must be forced to zero (no CEGAR loss)."""
        from src.dataset import build_synthetic_dataset
        D = 16
        trainer = self._make_trainer(D=D, H=32, B=16, seed=130)
        trainer.warmup_epochs = 999   # never leave warmup
        rng = _rng(131)
        emb1 = rng.standard_normal((8, D))
        emb2 = rng.standard_normal((8, D))
        labels = np.array([0, 0, 1, 2, 0, 1, 2, 1])
        trainer._epoch = 1            # inside warmup
        loss_d = trainer._step(emb1, emb2, labels)
        self.assertEqual(loss_d['cegar'], 0.0)

    def test_flush_every_clears_buffer(self):
        """flush_every=2 must clear the buffer after epochs 2, 4, 6 …"""
        from src.dataset import build_synthetic_dataset
        D = 16
        trainer = self._make_trainer(D=D, H=32, B=16, seed=140)
        trainer.flush_every = 2
        trainer.warmup_epochs = 0
        train, val, _ = build_synthetic_dataset(
            n_train=64, n_val=20, dim=D, rng=_rng(141))
        trainer.train(train, val, n_epochs=4, batch_size=32,
                      verbose=False, patience=20)
        # After epoch 4 the buffer was just flushed, so size should be
        # at most one epoch's worth of mistakes (much less than max_size).
        self.assertLess(len(trainer.cegar), trainer.cegar.max_size)

    # --- early stopping ---

    def test_early_stopping_triggers(self):
        """With patience=1, training should stop very early."""
        from src.dataset import build_synthetic_dataset
        D = 8
        trainer = self._make_trainer(D=D, H=16, B=8, seed=120)
        train, val, _ = build_synthetic_dataset(n_train=60, n_val=20, dim=D, rng=_rng(121))
        history = trainer.train(train, val, n_epochs=100, batch_size=30,
                                verbose=False, patience=1)
        self.assertLess(len(history), 100)


# ===========================================================================
# 6b. ThreeClassProbe & extract_box_features
# ===========================================================================

class TestThreeClassProbe(unittest.TestCase):

    def _make_boxes(self, N=60, D=8, seed=0):
        rng = _rng(seed)
        l = rng.standard_normal((N, D))
        u = l + np.abs(rng.standard_normal((N, D))) + 0.1
        return l, u

    def test_extract_box_features_shape(self):
        from src.metrics import extract_box_features
        l1, u1 = self._make_boxes(N=20, D=8, seed=10)
        l2, u2 = self._make_boxes(N=20, D=8, seed=11)
        feat = extract_box_features(l1, u1, l2, u2)
        self.assertEqual(feat.shape, (20, 4))

    def test_extract_box_features_viol_fwd_nonneg(self):
        from src.metrics import extract_box_features
        l1, u1 = self._make_boxes(N=30, D=8, seed=12)
        l2, u2 = self._make_boxes(N=30, D=8, seed=13)
        feat = extract_box_features(l1, u1, l2, u2)
        self.assertTrue(np.all(feat[:, 0] >= 0.0), "viol_fwd must be ≥ 0")
        self.assertTrue(np.all(feat[:, 1] >= 0.0), "viol_bwd must be ≥ 0")

    def test_extract_box_features_overlap_nonneg(self):
        from src.metrics import extract_box_features
        l1, u1 = self._make_boxes(N=20, D=4, seed=14)
        l2, u2 = self._make_boxes(N=20, D=4, seed=15)
        feat = extract_box_features(l1, u1, l2, u2)
        self.assertTrue(np.all(feat[:, 2] >= 0.0), "overlap must be ≥ 0")

    def test_extract_box_features_containment_gives_zero_viol_fwd(self):
        """If box1 ⊆ box2, viol_fwd should be 0."""
        from src.metrics import extract_box_features
        # Construct containment: l2 ≤ l1 and u1 ≤ u2
        N, D = 10, 6
        rng = _rng(16)
        l2 = rng.standard_normal((N, D)) - 2.0
        u2 = l2 + 4.0
        l1 = l2 + 0.5   # strictly inside
        u1 = u2 - 0.5
        feat = extract_box_features(l1, u1, l2, u2)
        np.testing.assert_allclose(feat[:, 0], 0.0, atol=1e-8,
                                   err_msg="viol_fwd should be 0 when contained")

    def test_three_class_probe_fit_returns_self(self):
        from src.metrics import ThreeClassProbe, extract_box_features
        l1, u1 = self._make_boxes(N=60, D=8, seed=20)
        l2, u2 = self._make_boxes(N=60, D=8, seed=21)
        feat = extract_box_features(l1, u1, l2, u2)
        labels = np.array([i % 3 for i in range(60)])
        probe = ThreeClassProbe(n_epochs=10)
        result = probe.fit(feat, labels)
        self.assertIs(result, probe)

    def test_three_class_probe_predict_shape(self):
        from src.metrics import ThreeClassProbe, extract_box_features
        l1, u1 = self._make_boxes(N=60, D=8, seed=22)
        l2, u2 = self._make_boxes(N=60, D=8, seed=23)
        feat = extract_box_features(l1, u1, l2, u2)
        labels = np.array([i % 3 for i in range(60)])
        probe = ThreeClassProbe(n_epochs=5).fit(feat, labels)
        preds = probe.predict(feat)
        self.assertEqual(preds.shape, (60,))
        self.assertTrue(np.all((preds >= 0) & (preds <= 2)))

    def test_three_class_probe_evaluate_keys(self):
        from src.metrics import ThreeClassProbe, extract_box_features
        l1, u1 = self._make_boxes(N=60, D=8, seed=24)
        l2, u2 = self._make_boxes(N=60, D=8, seed=25)
        feat = extract_box_features(l1, u1, l2, u2)
        labels = np.array([i % 3 for i in range(60)])
        probe = ThreeClassProbe(n_epochs=5).fit(feat, labels)
        res = probe.evaluate(feat, labels)
        self.assertIn('acc_3class', res)
        self.assertIn('per_class', res)
        self.assertIn('confusion', res)
        for nm in ['entail', 'neutral', 'contra']:
            self.assertIn(nm, res['per_class'])

    def test_three_class_probe_confusion_shape(self):
        from src.metrics import ThreeClassProbe, extract_box_features
        l1, u1 = self._make_boxes(N=90, D=8, seed=26)
        l2, u2 = self._make_boxes(N=90, D=8, seed=27)
        feat = extract_box_features(l1, u1, l2, u2)
        labels = np.array([i % 3 for i in range(90)])
        probe = ThreeClassProbe(n_epochs=5).fit(feat, labels)
        res = probe.evaluate(feat, labels)
        self.assertEqual(res['confusion'].shape, (3, 3))
        # Confusion rows must sum to per-class counts
        for c in range(3):
            self.assertEqual(res['confusion'][c].sum(), int(np.sum(labels == c)))

    def test_three_class_probe_learns_on_separable(self):
        """On linearly separable features the probe should reach >80% train acc."""
        from src.metrics import ThreeClassProbe
        rng = _rng(30)
        N = 300
        # Build clearly separable 4-D features for 3 classes
        X = np.zeros((N, 4))
        y = np.repeat([0, 1, 2], N // 3)
        X[y == 0] = rng.standard_normal((N // 3, 4)) + np.array([5., -5.,  3., 0.])
        X[y == 1] = rng.standard_normal((N // 3, 4)) + np.array([0.,  0.,  0., 0.])
        X[y == 2] = rng.standard_normal((N // 3, 4)) + np.array([-5., 5., -3., 0.])
        probe = ThreeClassProbe(n_epochs=400, lr=0.1).fit(X, y)
        res = probe.evaluate(X, y)
        self.assertGreater(res['acc_3class'], 0.80,
                           f"Expected >80% on separable data, got {res['acc_3class']:.3f}")

    def test_evaluate_3class_trainer(self):
        """ImprovedTrainer.evaluate_3class should return a dict with acc_3class."""
        from src.improved_trainer import ImprovedTrainer
        from src.dataset import build_synthetic_dataset
        D = 16
        trainer = ImprovedTrainer(
            input_dim=D, hidden_dim=32, box_dim=8, lr=1e-3,
            lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.0,
            margin=1.0, scale=5.0, rng=_rng(200),
        )
        train, _, test = build_synthetic_dataset(
            n_train=120, n_val=30, n_test=60, dim=D, rng=_rng(201))
        res = trainer.evaluate_3class(train, test)
        self.assertIn('acc_3class', res)
        self.assertIn('per_class', res)
        self.assertIn('confusion', res)
        self.assertGreaterEqual(res['acc_3class'], 0.0)
        self.assertLessEqual(res['acc_3class'], 1.0)

    def test_evaluate_3class_trained_better_than_random(self):
        """After training, 3-class acc should beat 33% random baseline."""
        from src.improved_trainer import ImprovedTrainer
        from src.dataset import build_synthetic_dataset
        D = 16
        trainer = ImprovedTrainer(
            input_dim=D, hidden_dim=64, box_dim=16, lr=2e-3,
            lambda1=0.3, lambda2=2.0, lambda3=1.0, lambda4=0.0,
            margin=1.0, scale=5.0, rng=_rng(210),
        )
        train, val, test = build_synthetic_dataset(
            n_train=240, n_val=60, n_test=120, dim=D, rng=_rng(211))
        trainer.train(train, val, n_epochs=15, batch_size=32,
                      verbose=False, patience=20)
        res = trainer.evaluate_3class(train, test)
        self.assertGreater(res['acc_3class'], 1/3 - 0.05,
                           f"3-class acc {res['acc_3class']:.3f} is no better than random")


# ===========================================================================
# 7. webui_service — light smoke tests (no server required)
# ===========================================================================

class TestWebuiService(unittest.TestCase):

    def test_default_config_keys(self):
        from src.webui_service import default_training_config
        cfg = default_training_config()
        for key in ['seed', 'n_train', 'n_val', 'lr', 'box_dim', 'n_epochs']:
            self.assertIn(key, cfg)

    def test_coerce_config_types(self):
        from src.webui_service import _coerce_config
        cfg = _coerce_config({"n_epochs": "5", "lr": "0.001", "seed": 7})
        self.assertIsInstance(cfg['n_epochs'], int)
        self.assertIsInstance(cfg['lr'], float)

    def test_list_sentence_catalog(self):
        from src.webui_service import list_sentence_catalog
        catalog = list_sentence_catalog()
        self.assertIsInstance(catalog, list)
        self.assertGreater(len(catalog), 0)
        for item in catalog:
            self.assertIn('id', item)
            self.assertIn('text', item)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
