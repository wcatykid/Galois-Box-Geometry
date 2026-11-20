"""
Real-data experiment: SBERT embeddings + SNLI + existing NumPy Galois trainer.
  - Model : all-MiniLM-L6-v2  (384-dim, ~90 MB, CPU-friendly)
  - Data  : SNLI train[:12000] / validation[:2000] / test[:2000]
  - Trainer: ImprovedTrainer (unchanged), input_dim=384

Reports both:
  • Binary entailment-vs-rest  (Acc / F1 / AUC)
  • 3-class {entail, neutral, contra}  accuracy via linear box-feature probe
    features: [viol(p→h), viol(h→p), mean_overlap, width_asymmetry]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from src.improved_trainer import ImprovedTrainer
from src.metrics import (
    compute_nli_metrics,
    extract_box_features, compute_3class_features, ThreeClassProbe,
    CosineSimilarityBaseline, OrderEmbeddingBaseline,
)
from src.box_lattice import box_containment_score, check_reflexivity, check_antisymmetry, check_transitivity

# ── 1. Load SNLI ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  Galois-NLP  ×  SBERT (all-MiniLM-L6-v2)  ×  SNLI")
print("="*70)

print("\n[1/5] Loading SNLI …")
raw_train = load_dataset('snli', split='train[:12000]').filter(lambda x: x['label'] != -1)
raw_val   = load_dataset('snli', split='validation[:2000]').filter(lambda x: x['label'] != -1)
raw_test  = load_dataset('snli', split='test[:2000]').filter(lambda x: x['label'] != -1)

print(f"      train={len(raw_train)}  val={len(raw_val)}  test={len(raw_test)}")
for split_name, split in [('train', raw_train), ('val', raw_val), ('test', raw_test)]:
    counts = np.bincount(split['label'], minlength=3)
    print(f"      {split_name}: entail={counts[0]}  neutral={counts[1]}  contra={counts[2]}")

# ── 2. Encode with SBERT ─────────────────────────────────────────────────────
print("\n[2/5] Encoding with SBERT (this takes ~1-2 min on CPU) …")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

t0 = time.time()
emb1_train = sbert.encode(raw_train['premise'],    batch_size=256, show_progress_bar=True, convert_to_numpy=True)
emb2_train = sbert.encode(raw_train['hypothesis'], batch_size=256, show_progress_bar=True, convert_to_numpy=True)
emb1_val   = sbert.encode(raw_val['premise'],      batch_size=256, show_progress_bar=True, convert_to_numpy=True)
emb2_val   = sbert.encode(raw_val['hypothesis'],   batch_size=256, show_progress_bar=True, convert_to_numpy=True)
emb1_test  = sbert.encode(raw_test['premise'],     batch_size=256, show_progress_bar=True, convert_to_numpy=True)
emb2_test  = sbert.encode(raw_test['hypothesis'],  batch_size=256, show_progress_bar=True, convert_to_numpy=True)
print(f"      Encoding done in {time.time()-t0:.1f}s  — embedding dim: {emb1_train.shape[1]}")

train_data = {'emb1': emb1_train, 'emb2': emb2_train, 'label': np.array(raw_train['label'])}
val_data   = {'emb1': emb1_val,   'emb2': emb2_val,   'label': np.array(raw_val['label'])}
test_data  = {'emb1': emb1_test,  'emb2': emb2_test,  'label': np.array(raw_test['label'])}

INPUT_DIM = emb1_train.shape[1]   # 384

# ── 3. Baselines ─────────────────────────────────────────────────────────────
print("\n[3/5] Baselines …")

t0 = time.time()
cos_bl = CosineSimilarityBaseline().fit(train_data)
cos_m  = cos_bl.evaluate(test_data)
cos_t  = time.time() - t0
print(f"      Cosine Sim (SBERT raw)   Acc={cos_m['accuracy']:.4f}  F1={cos_m['f1']:.4f}  AUC={cos_m['auc']:.4f}  ({cos_t:.1f}s)")

t0 = time.time()
oe_bl = OrderEmbeddingBaseline(box_dim=64, n_epochs=30, lr=1e-3).fit(train_data)
oe_m  = oe_bl.evaluate(test_data)
oe_t  = time.time() - t0
print(f"      Order Embeddings         Acc={oe_m['accuracy']:.4f}  F1={oe_m['f1']:.4f}  AUC={oe_m['auc']:.4f}  ({oe_t:.1f}s)")

# ── 4. Galois-NLP  ────────────────────────────────────────────────────────────
print("\n[4/5] Training Galois-NLP on SBERT embeddings …")

print("\n  --- Ablation: no CEGAR ---")
tr_nc = ImprovedTrainer(
    input_dim=INPUT_DIM, hidden_dim=512, box_dim=128,
    lr=5e-4,
    lambda1=0.3, lambda2=2.0, lambda3=1.0, lambda4=0.0, lambda5=0.5,
    margin=1.0, scale=5.0,
    # 3-class margins calibrated from violation magnitude analysis
    ent_bwd_target=10.0, neu_fwd_margin=3.0, neu_bwd_ceiling=6.0,
    con_fwd_margin=5.0,  overlap_weight=0.1,
    rng=np.random.default_rng(7),
)
t0 = time.time()
tr_nc.train(train_data, val_data, n_epochs=30, batch_size=128, verbose=True, patience=7)
nc_time = time.time() - t0
nc_m = tr_nc.evaluate(test_data)
print(f"\n  No-CEGAR test  Acc={nc_m['accuracy']:.4f}  F1={nc_m['f1']:.4f}  AUC={nc_m['auc']:.4f}  ({nc_time:.1f}s)")

print("\n  --- Full Galois-NLP (with CEGAR, warmup=5, cegar_margin=5.0) ---")
trainer = ImprovedTrainer(
    input_dim=INPUT_DIM, hidden_dim=512, box_dim=128,
    lr=5e-4,
    lambda1=0.3, lambda2=2.0, lambda3=1.0, lambda4=0.3, lambda5=0.5,
    margin=1.0, scale=5.0,
    cegar_margin=5.0,       # SBERT 384-dim violations are O(10-100), not O(1)
    warmup_epochs=5,        # don't activate CEGAR until model has a base
    cegar_max_size=4000,    # larger buffer for real data diversity
    # 3-class margins calibrated from violation magnitude analysis
    ent_bwd_target=10.0, neu_fwd_margin=3.0, neu_bwd_ceiling=6.0,
    con_fwd_margin=5.0,  overlap_weight=0.1,
    rng=np.random.default_rng(42),
)
t0 = time.time()
history = trainer.train(train_data, val_data, n_epochs=30, batch_size=128, verbose=True, patience=7)
cegar_time = time.time() - t0

l1, u1, _ = trainer.model.encode(test_data['emb1'])
l2, u2, _ = trainer.model.encode(test_data['emb2'])
p_ent = box_containment_score(l1, u1, l2, u2, scale=trainer.scale)
full_m = compute_nli_metrics(p_ent, test_data['label'])

# ── 5. 3-class probe evaluation ───────────────────────────────────────────────
print("\n[5/5] 3-Class Linear Probe Evaluation …")

# --- Cosine baseline: 1-D feature [cos(p,h)] (symmetric — upper bound for cos)
def _cos_feat(data):
    e1 = data['emb1'] / (np.linalg.norm(data['emb1'], axis=1, keepdims=True) + 1e-8)
    e2 = data['emb2'] / (np.linalg.norm(data['emb2'], axis=1, keepdims=True) + 1e-8)
    return np.sum(e1 * e2, axis=1, keepdims=True)  # (N, 1)

cos_probe = ThreeClassProbe().fit(_cos_feat(train_data), train_data['label'])
cos_3     = cos_probe.evaluate(_cos_feat(test_data), test_data['label'])

# --- Order Embeddings baseline: 2-D asymmetric features [e(p→h), e(h→p)]
def _oe_feat(data, oe):
    e1 = oe._embed(data['emb1'])
    e2 = oe._embed(data['emb2'])
    fwd = oe._energy(e1, e2) / oe.box_dim   # normalised
    bwd = oe._energy(e2, e1) / oe.box_dim
    return np.stack([fwd, bwd], axis=1)      # (N, 2)

oe_probe = ThreeClassProbe().fit(_oe_feat(train_data, oe_bl), train_data['label'])
oe_3     = oe_probe.evaluate(_oe_feat(test_data, oe_bl), test_data['label'])

# --- Galois-NLP models: 4-D box features
nc_3   = tr_nc.evaluate_3class(train_data, test_data)
full_3 = trainer.evaluate_3class(train_data, test_data)

# ── Binary results table ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SNLI Test-Set Results — Binary Entailment vs Rest")
print("="*70)
header = f"  {'Model':<36}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}  {'Time':>7}"
print(header)
print("  " + "-"*66)
for name, m, t in [
    ("Cosine Sim (SBERT raw)",          cos_m,  cos_t),
    ("Order Embeddings",                oe_m,   oe_t),
    ("Galois-NLP  (no CEGAR)",          nc_m,   nc_time),
    ("Galois-NLP  (CEGAR warmup=5)",    full_m, cegar_time),
]:
    print(f"  {name:<36}  {m['accuracy']:>6.4f}  {m['f1']:>6.4f}  {m['auc']:>6.4f}  {t:>6.1f}s")
print("="*70)

# ── 3-class results table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SNLI Test-Set Results — 3-Class (entail / neutral / contra)")
print("  Features: Galois=[viol_fwd, viol_bwd, asymmetry, viol_sum, overlap, w1, w2, width_ratio, width_diff]  (9-D)")
print("            OE=[energy(p→h), energy(h→p)]   Cosine=[cos(p,h)]")
print("="*70)
hdr3 = f"  {'Model':<36}  {'3-Acc':>6}  {'ent':>5}  {'neu':>5}  {'con':>5}"
print(hdr3)
print("  " + "-"*66)
for name, r3 in [
    ("Cosine (1-D sym. feature)",       cos_3),
    ("Order Embeddings (2-D asym.)",    oe_3),
    ("Galois-NLP  (no CEGAR)",          nc_3),
    ("Galois-NLP  (CEGAR warmup=5)",    full_3),
]:
    pc = r3['per_class']
    print(f"  {name:<36}  {r3['acc_3class']:>6.4f}  "
          f"{pc.get('entail', float('nan')):>5.3f}  "
          f"{pc.get('neutral', float('nan')):>5.3f}  "
          f"{pc.get('contra', float('nan')):>5.3f}")
print("="*70)

# ── Confusion matrix for best model ──────────────────────────────────────────
print("\n  Confusion Matrix — Galois-NLP (CEGAR)  [rows=true, cols=pred]")
print("  Labels: 0=entail  1=neutral  2=contra")
cm = full_3['confusion']
print(f"  {'':>10}  pred_ent  pred_neu  pred_con")
for i, nm in enumerate(['true_ent', 'true_neu', 'true_con']):
    print(f"  {nm:>10}  {cm[i,0]:>8d}  {cm[i,1]:>8d}  {cm[i,2]:>8d}")

# ── Lattice axioms ────────────────────────────────────────────────────────────
print("\n  Lattice Axiom Verification (learned boxes on test set):")
print(f"    Reflexivity  : {'PASS' if check_reflexivity(l1, u1)           else 'FAIL'}")
print(f"    Antisymmetry : {'PASS' if check_antisymmetry(l1, u1, l2, u2)  else 'FAIL'}")
print(f"    Transitivity : {'PASS' if check_transitivity(l1, u1, l2, u2, l1, u1) else 'FAIL'}")

# ── CEGAR training curve ──────────────────────────────────────────────────────
print("\n  CEGAR Training Curve (every 3rd epoch):")
print(f"  {'Epoch':>6}  {'ValAcc':>8}  {'ValF1':>7}  {'L_gal':>7}  {'L_task':>7}  {'L_ceg':>7}  {'L_3cls':>7}")
for h in history[::3]:
    print(f"  {h['epoch']:>6}  {h['val_acc']:>8.4f}  {h['val_f1']:>7.4f}  "
          f"{h['galois']:>7.4f}  {h['task']:>7.4f}  {h['cegar']:>7.4f}  {h['3class']:>7.4f}")
