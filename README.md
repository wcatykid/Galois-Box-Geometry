# A Galois-Inspired Box Geometry for Ordered Semantic Relation

Code and experiments for the EACL 2027 paper *"A Galois-Inspired Box Geometry for Ordered Semantic Relation"* by Michael Thomas, Supratik Mukhopadhyay, and Chris Alvin.

## Overview

This repository implements a framework for learning approximate Galois connections between natural language and a complete lattice of axis-aligned boxes.
The abstraction function (alpha) maps sentences to boxes where containment models entailment and lattice joins compose meanings.
The concretization function (gamma) reconstructs embeddings from box representations, which enables round-trip fidelity evaluation.
The paper proves that bounded Galois consistency loss implies an approximate Galois connection, conditional on surrogate calibration, and verifies the bound empirically.

## Repository Structure

```
.
├── src/                          # Core library
│   ├── box_lattice.py            # Box lattice operations and axiom checks
│   ├── encoder.py                # Abstraction (alpha) and concretization (gamma) networks
│   ├── losses.py                 # L_recon, L_Galois, L_task, L_CEGAR, and CEGAR buffer
│   ├── dataset.py                # Synthetic NLI dataset generator ("Tom goes" hierarchy)
│   ├── improved_trainer.py       # Training loop with analytic backprop (NumPy)
│   ├── metrics.py                # NLI and STS metrics plus baseline models
│   ├── trainer.py                # Trainer (legacy reference)
│   └── webui_service.py          # Web UI backend service
│
├── run_experiment.py             # Synthetic NLI experiment (Section 4.1, Tables 2-3)
├── run_sbert_snli.py             # SBERT-based SNLI experiment
├── torch_train.py                # PyTorch training on SNLI (Galois v1)
├── torch_train_v2.py             # PyTorch training on SNLI with fine-tuned SBERT (Galois v2)
├── torch_baseline.py             # SNLI baseline comparisons
│
├── eval_hyperlex.py              # HyperLex evaluation (Section 4.2)
├── eval_hyperlex_embeddings.py   # HyperLex across GloVe, Word2Vec, and SBERT (Table 4)
├── eval_hyperlex_fullwn.py       # HyperLex with full WordNet training (Table 4)
├── eval_multinli.py              # MultiNLI evaluation (supplementary; not reported in the paper)
├── eval_multinli_calibrate.py    # MultiNLI calibrated evaluation (supplementary)
│
├── cegar_experiment.py           # Training-time CEGAR experiment (Section 4.4, Appendix H)
├── cegar_corrected.py            # Corrected CEGAR with confidence filtering (Appendix H)
├── inference_cegar.py            # Inference-time repair, in-domain nouns (Appendix C)
├── inference_cegar_verb.py       # Inference-time repair under noun-to-verb domain shift (Appendix C)
├── inference_cegar_noise.py      # Noise robustness experiments
├── cegar_chunk1_fastpass.py      # Repair triage sweep (Appendix C)
├── cegar_chunk2_focus.py         # Repair focused evaluation (Appendix C)
├── cegar_chunk2.py               # Full-scale repair evaluation (Appendix C, Tables 12-13)
│
├── eval_joins.py                 # Lattice join containment on the synthetic hierarchy
├── eval_join_loss.py             # Join consistency loss on the synthetic hierarchy
├── eval_joins_wordnet.py         # Join consistency on WordNet + GloVe (Appendix L)
├── eval_joins_snli.py            # Join consistency on SNLI + SBERT (Appendix L)
├── eval_concretization.py        # Concretization quality on the synthetic task
├── eval_concretization_real.py   # Concretization quality on WordNet and SNLI (Appendix M)
├── viz_boxes.py                  # 2D PCA box visualization (Appendix N)
│
├── webui_server.py               # Local web interface server
├── webui/static/index.html       # Web UI frontend
├── start_web_interface.bat       # Windows launcher for the web UI
├── tests/test_galois.py          # Unit tests
│
├── data/hyperlex/                # HyperLex dataset
├── results/                      # Result JSONs and small trained models
└── results_cegar_chunk2/         # Full-scale inference-time repair results
```

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/wcatykid/Galois-Box-Geometry.git
cd Galois-Box-Geometry
pip install -r requirements.txt
```

All experiments in the paper were run on a CPU (Intel i7-1365U) without a GPU.
The PyTorch experiments additionally require the packages listed (commented out) in `requirements.txt`.

### 2. Run the synthetic experiment (CPU-only, no internet)

```bash
python run_experiment.py
```

This reproduces the synthetic NLI results in Table 2 and the ablation in Table 3.

### 3. Run the HyperLex evaluation (Section 4.2)

```bash
python eval_hyperlex_fullwn.py
python eval_hyperlex_embeddings.py
```

GloVe and Word2Vec embeddings are downloaded automatically through gensim.

### 4. Run inference-time repair (Appendix C, Tables 12-13)

```bash
python cegar_chunk2.py
```

This reproduces the full-scale coordinate descent result: a 100% fix rate on 2,538 verb IS-A constraints with zero new violations.

### 5. Run SNLI training

```bash
pip install torch transformers datasets sentence-transformers
python torch_train_v2.py
```

### 6. Run join consistency and concretization experiments (Appendices L and M)

```bash
python eval_joins_wordnet.py
python eval_joins_snli.py
python eval_concretization_real.py
```

### 7. Web interface

```bash
python webui_server.py
```

## Key Results

### Synthetic NLI (Table 2)

| Model | Binary Acc. | 3-class Acc. | Neutral Recall |
|---|---|---|---|
| Cosine similarity | 0.494 | 0.609 | 0.126 |
| Order Embeddings | 0.832 | 0.683 | 0.258 |
| **Galois + CEGAR (ours)** | **0.941** | **0.870** | **0.682** |

### HyperLex Directionality (Table 4)

The model achieves 91-93% directionality accuracy across GloVe, Word2Vec, and SBERT.
The best asymmetric projection baseline reaches 78.7%, so the box geometry adds 12 percentage points.

### SNLI (Table 6)

| Model (frozen encoder) | Binary Acc. | Macro-F1 | Neutral Recall |
|---|---|---|---|
| Cosine (SBERT) | 0.506 | 0.345 | 0.240 |
| Order Embeddings | 0.690 | 0.445 | 0.191 |
| Probabilistic Box | 0.661 | 0.085 | 0.220 |
| **Galois + CEGAR (ours)** | **0.744** | **0.498** | **0.656** |

### Inference-Time Repair (Table 12)

| Method | Fixed (%) | New Violations | Net |
|---|---|---|---|
| No refinement | 0 (0.0%) | 0 | 0 |
| Random perturbation | 0 (0.0%) | 400 | -400 |
| Selective CEGAR | 466 (41.8%) | 89 | +377 |
| Hybrid repair | 1,109 (99.6%) | 2 | +1,107 |
| **Coordinate descent** | **1,114 (100.0%)** | **0** | **+1,114** |

## Architecture

**Abstraction (alpha):** input embedding -> two-layer MLP (input -> 512 -> 512) -> lower-corner and width heads (512 -> 128 each).
A softplus on the width enforces box validity (l <= u).

**Concretization (gamma):** box centroid -> MLP decoder (128 -> 512 -> input) -> reconstructed embedding.

**Loss:** L_total = lambda_1 * L_recon + lambda_2 * L_Galois + lambda_3 * L_task + lambda_4 * L_CEGAR + lambda_5 * L_3class

## Tests

```bash
pytest tests/
```

## Citation

```bibtex
@inproceedings{thomas-etal-2027-galois,
  title     = {A Galois-Inspired Box Geometry for Ordered Semantic Relation},
  author    = {Thomas, Michael and Mukhopadhyay, Supratik and Alvin, Chris},
  booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics},
  year      = {2027}
}
```

## License

This code is released under the MIT License. See `LICENSE` for details.
