"""
Main Experiment Script — Galois NLP
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, time, json
from pathlib import Path

from src.dataset          import build_synthetic_dataset, build_synthetic_sts
from src.improved_trainer import ImprovedTrainer
from src.metrics          import (
    compute_nli_metrics, compute_sts_metrics, compute_box_stats,
    CosineSimilarityBaseline, DotProductBaseline,
    MajorityClassBaseline, LinearSVMBaseline, OrderEmbeddingBaseline)
from src.box_lattice      import (
    box_containment_score, box_containment_violation,
    check_reflexivity, check_antisymmetry, check_transitivity)


def print_table(results, title="Results"):
    print(f"\n{'='*76}\n  {title}\n{'='*76}")
    cols   = ['Model',                  'Acc',  'F1',   'AUC',  'Prec', 'Recall','Time(s)']
    widths = [34,                        8,      8,      8,      8,      8,       8]
    print(''.join(f"{c:<{w}}" for c,w in zip(cols,widths)))
    print('─'*76)
    for r in results:
        vals = [r['model'], f"{r.get('accuracy',0):.4f}", f"{r.get('f1',0):.4f}",
                f"{r.get('auc',0):.4f}", f"{r.get('precision',0):.4f}",
                f"{r.get('recall',0):.4f}", f"{r.get('train_time',0):.1f}"]
        print(''.join(f"{v:<{w}}" for v,w in zip(vals,widths)))
    print('='*76)


def run_experiment(seed=42, n_train=3000, n_val=500, n_test=1000,
                   input_dim=64, box_dim=64, hidden_dim=256,
                   n_epochs=40, batch_size=64, lr=1e-3, verbose=True):

    print(f"\n{'#'*76}")
    print(f"  Galois-Structured NLP — Full Experiment  (seed={seed})")
    print(f"  n_train={n_train}  box_dim={box_dim}  hidden={hidden_dim}  lr={lr}")
    print(f"{'#'*76}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_data, val_data, test_data = build_synthetic_dataset(
        n_train=n_train, n_val=n_val, n_test=n_test,
        dim=input_dim, noise=0.05, rng=np.random.default_rng(seed))
    sts_data = build_synthetic_sts(n=800, dim=input_dim,
                                   rng=np.random.default_rng(seed+1))
    lc = np.bincount(train_data['label'])
    print(f"  Splits: train={n_train}, val={n_val}, test={n_test}")
    print(f"  Label dist: entail={lc[0]}  neutral={lc[1]}  contra={lc[2]}\n")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print('─'*60 + "\nBaselines\n" + '─'*60)
    baselines = [MajorityClassBaseline(), CosineSimilarityBaseline(),
                 DotProductBaseline(),    LinearSVMBaseline(n_epochs=150, lr=0.01),
                 OrderEmbeddingBaseline(box_dim=box_dim, n_epochs=100, lr=1e-3)]
    baseline_results = []
    for bl in baselines:
        t0 = time.time()
        bl.fit(train_data)
        m = bl.evaluate(test_data)
        m['model'] = bl.name; m['train_time'] = round(time.time()-t0, 2)
        baseline_results.append(m)
        print(f"  {bl.name:<34} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # ── Galois — no CEGAR (ablation) ─────────────────────────────────────────
    print('\n' + '─'*60 + "\nAblation: Galois WITHOUT CEGAR\n" + '─'*60)
    tr_nc = ImprovedTrainer(input_dim=input_dim, hidden_dim=hidden_dim,
                            box_dim=box_dim, lr=lr,
                            lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.0,
                            margin=1.0, scale=5.0, rng=np.random.default_rng(seed+20))
    t0 = time.time()
    tr_nc.train(train_data, val_data, n_epochs=n_epochs,
                batch_size=batch_size, verbose=verbose, patience=10)
    m_nc = tr_nc.evaluate(test_data)
    m_nc['model'] = 'Galois-NLP (no CEGAR)'; m_nc['train_time'] = round(time.time()-t0,2)

    # ── Galois — full model ───────────────────────────────────────────────────
    print('\n' + '─'*60 + "\nFull Galois-NLP (with CEGAR)\n" + '─'*60)
    trainer = ImprovedTrainer(input_dim=input_dim, hidden_dim=hidden_dim,
                              box_dim=box_dim, lr=lr,
                              lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
                              margin=1.0, scale=5.0, rng=np.random.default_rng(seed+10))
    t0 = time.time()
    history = trainer.train(train_data, val_data, n_epochs=n_epochs,
                            batch_size=batch_size, verbose=verbose, patience=10)
    t_galois = time.time() - t0

    l1,u1,_ = trainer.model.encode(test_data['emb1'])
    l2,u2,_ = trainer.model.encode(test_data['emb2'])
    p_ent   = box_containment_score(l1, u1, l2, u2, scale=trainer.scale)
    m_full  = compute_nli_metrics(p_ent, test_data['label'])
    box_st  = compute_box_stats(l1, u1, l2, u2)
    m_full['model'] = 'Galois-NLP (CEGAR)'; m_full['train_time'] = round(t_galois,2)

    all_results = baseline_results + [m_nc, m_full]

    # ── Print NLI table ───────────────────────────────────────────────────────
    print_table(all_results, "NLI Test-Set Comparison — All Models")

    # ── Box stats ─────────────────────────────────────────────────────────────
    print("\n  Box Representation Statistics (Galois full):")
    for k,v in box_st.items():
        print(f"    {k:<38}: {v:.4f}")
    print(f"    {'CEGAR buffer size':<38}: {len(trainer.cegar)}")

    # ── STS ───────────────────────────────────────────────────────────────────
    print('\n' + '─'*60 + "\nSTS Evaluation\n" + '─'*60)
    l1s,u1s,_ = trainer.model.encode(sts_data['emb1'])
    l2s,u2s,_ = trainer.model.encode(sts_data['emb2'])
    viol_sts  = box_containment_violation(l1s,u1s,l2s,u2s)
    gsim      = 5.0 / (1.0 + viol_sts)
    n1 = sts_data['emb1'] / (np.linalg.norm(sts_data['emb1'],axis=-1,keepdims=True)+1e-8)
    n2 = sts_data['emb2'] / (np.linalg.norm(sts_data['emb2'],axis=-1,keepdims=True)+1e-8)
    csim = (np.sum(n1*n2,axis=-1)+1.0)*2.5
    sts_cos = compute_sts_metrics(csim, sts_data['score'])
    sts_gal = compute_sts_metrics(gsim, sts_data['score'])
    print(f"  {'Model':<34}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'MSE':>8}")
    print('  ' + '─'*67)
    print(f"  {'Cosine Similarity (SBERT-style)':<34}  {sts_cos['pearson_r']:>10.4f}  "
          f"{sts_cos['spearman_rho']:>11.4f}  {sts_cos['mse']:>8.4f}")
    print(f"  {'Galois-NLP (CEGAR)':<34}  {sts_gal['pearson_r']:>10.4f}  "
          f"{sts_gal['spearman_rho']:>11.4f}  {sts_gal['mse']:>8.4f}")

    # ── CEGAR ablation (epoch-level) ──────────────────────────────────────────
    print('\n' + '─'*60 + "\nCEGAR Ablation — Training Curve\n" + '─'*60)
    print(f"  {'Epoch':>6}  {'ValAcc':>8}  {'ValF1':>8}  {'L_gal':>8}  {'L_task':>8}  {'L_ceg':>8}")
    step = max(1, len(history)//12)
    for h in history[::step]:
        print(f"  {h['epoch']:>6}  {h['val_acc']:>8.4f}  {h['val_f1']:>8.4f}  "
              f"{h['galois']:>8.4f}  {h['task']:>8.4f}  {h['cegar']:>8.4f}")

    # ── Lattice axioms ────────────────────────────────────────────────────────
    print('\n' + '─'*60 + "\nLattice Axiom Verification\n" + '─'*60)
    ref_ok   = check_reflexivity(l1, u1)
    anti_ok  = check_antisymmetry(l1, u1, l2, u2)
    trans_ok = check_transitivity(l1, u1, l2, u2, l1, u1)
    print(f"  Reflexivity  : {'PASS ✓' if ref_ok  else 'FAIL ✗'}")
    print(f"  Antisymmetry : {'PASS ✓' if anti_ok else 'FAIL ✗'}")
    print(f"  Transitivity : {'PASS ✓' if trans_ok else 'FAIL ✗'}")

    # ── Qualitative box widths ────────────────────────────────────────────────
    print('\n' + '─'*60 + "\nQualitative: Box Width as Uncertainty Proxy\n" + '─'*60)
    from src.dataset import build_synthetic_embeddings, SENTENCES
    base_emb = build_synthetic_embeddings(dim=input_dim, noise=0.0)
    hierarchy = ['s0','s1','s2','s3','s4','s5']
    print(f"  {'Sentence':<42}  {'MeanWidth':>10}  Interpretation")
    for sid in hierarchy:
        e = base_emb[sid][None,:]
        lx,ux,_ = trainer.model.encode(e)
        w = float(np.mean(ux-lx))
        tag = "← vague/abstract" if w>0.45 else ("← precise" if w<0.25 else "← moderate")
        print(f"  {SENTENCES[sid]:<42}  {w:>10.4f}  {tag}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        'config': dict(seed=seed, n_train=n_train, n_val=n_val, n_test=n_test,
                       input_dim=input_dim, box_dim=box_dim, hidden_dim=hidden_dim,
                       lr=lr, n_epochs=n_epochs),
        'nli_results':      all_results,
        'sts_results':      {'cosine': sts_cos, 'galois': sts_gal},
        'box_stats':        box_st,
        'training_history': history,
        'lattice_checks':   dict(reflexivity=bool(ref_ok),
                                 antisymmetry=bool(anti_ok),
                                 transitivity=bool(trans_ok)),
    }
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Saved → results/experiment_results.json")
    return output


if __name__ == '__main__':
    run_experiment()
