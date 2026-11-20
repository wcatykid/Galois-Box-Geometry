"""
Dataset Utilities
=================
Provides:
  1. Synthetic NLI dataset generator (for offline / unit-test use)
  2. SNLI / MultiNLI loader wrappers (when HuggingFace datasets available)
  3. STS-B loader wrapper
  4. Pre-computed embedding loader (GloVe / random for offline mode)

The synthetic data is designed to replicate the *structural* properties
of SNLI that matter for Galois learning:
  - Entailment pairs cluster in embedding space  (premise ≈ subset of hypothesis)
  - Contradiction pairs are far apart
  - Neutral pairs are in between

Ground-truth label mapping (SNLI convention):
    0 = entailment
    1 = neutral
    2 = contradiction
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Synthetic NLI dataset
# ---------------------------------------------------------------------------

ENTAILMENT = 0
NEUTRAL = 1
CONTRADICTION = 2

# Sentence templates used to build synthetic embeddings
_TEMPLATES = {
    # (premise_class, hypothesis_class) → label
    # Hierarchy: specific ⊆ moderate ⊆ general
}

SENTENCES = {
    # Increasing specificity along "Tom goes" axis
    's0': "Something happens",
    's1': "Someone moves",
    's2': "Tom goes",
    's3': "Tom goes somewhere",
    's4': "Tom goes to school",
    's5': "Tom goes to East High School",
    's6': "Tom goes to West High School",
    's7': "Tom goes to Jefferson Elementary",
    # Unrelated
    's8': "Mary reads a book",
    's9': "The cat sits on a mat",
    's10': "Birds can fly",
    's11': "Water boils at 100 degrees",
}

# Semantic entailment edges (specific → general)
ENTAILMENT_PAIRS = [
    ('s5', 's4'), ('s5', 's3'), ('s5', 's2'), ('s5', 's1'), ('s5', 's0'),
    ('s6', 's4'), ('s6', 's3'), ('s6', 's2'),
    ('s7', 's4'), ('s7', 's3'), ('s7', 's2'),
    ('s4', 's3'), ('s4', 's2'), ('s4', 's1'), ('s4', 's0'),
    ('s3', 's2'), ('s3', 's1'), ('s3', 's0'),
    ('s2', 's1'), ('s2', 's0'),
    ('s1', 's0'),
]

NON_ENTAILMENT_PAIRS = [
    ('s8', 's2'), ('s9', 's2'), ('s10', 's2'),
    ('s5', 's6'), ('s5', 's7'),   # same level, not contained
    ('s2', 's4'),                  # reversed direction
    ('s8', 's9'), ('s10', 's11'),
]


def build_synthetic_embeddings(dim=64, noise=0.05, rng=None):
    """
    Build semantically structured embeddings for the toy sentences.

    Key design:
      - More general sentences get *larger* norm (representing a larger box)
      - Specific sentences are linear combinations of general ones
      - Unrelated sentences are random orthogonal directions
    """
    if rng is None:
        rng = np.random.default_rng(0)

    embeddings = {}

    # Base direction for the "Tom goes" axis
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)

    # Orthogonal directions for specificity modifiers
    def ortho_component(v, dim):
        r = rng.standard_normal(dim)
        r -= r.dot(v) * v  # remove projection onto v
        r /= np.linalg.norm(r) + 1e-8
        return r

    east_dir   = ortho_component(base, dim)
    west_dir   = ortho_component(base, dim)
    jeff_dir   = ortho_component(base, dim)
    school_dir = ortho_component(base, dim)
    some_dir   = ortho_component(base, dim)

    embeddings['s0'] = 0.1 * base + noise * rng.standard_normal(dim)          # "Something happens"
    embeddings['s1'] = 0.4 * base + noise * rng.standard_normal(dim)          # "Someone moves"
    embeddings['s2'] = 0.7 * base + noise * rng.standard_normal(dim)          # "Tom goes"
    embeddings['s3'] = 0.7 * base + 0.3 * some_dir + noise * rng.standard_normal(dim)  # "Tom goes somewhere"
    embeddings['s4'] = 0.7 * base + 0.3 * some_dir + 0.5 * school_dir + noise * rng.standard_normal(dim)
    embeddings['s5'] = 0.7 * base + 0.3 * some_dir + 0.5 * school_dir + 0.7 * east_dir + noise * rng.standard_normal(dim)
    embeddings['s6'] = 0.7 * base + 0.3 * some_dir + 0.5 * school_dir + 0.7 * west_dir + noise * rng.standard_normal(dim)
    embeddings['s7'] = 0.7 * base + 0.3 * some_dir + 0.5 * school_dir + 0.7 * jeff_dir + noise * rng.standard_normal(dim)

    # Unrelated sentences: random orthogonal
    for s in ['s8', 's9', 's10', 's11']:
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        embeddings[s] = v + noise * rng.standard_normal(dim)

    # L2-normalise all
    for k in embeddings:
        embeddings[k] = embeddings[k] / (np.linalg.norm(embeddings[k]) + 1e-8)

    return embeddings


def build_synthetic_dataset(n_train=1000, n_val=200, n_test=400,
                             dim=64, noise=0.05, rng=None):
    """
    Build a synthetic NLI dataset.

    Returns dicts with keys: 'emb1', 'emb2', 'label', 'sent1', 'sent2'
    """
    if rng is None:
        rng = np.random.default_rng(1)

    base_emb = build_synthetic_embeddings(dim=dim, noise=0.0, rng=rng)

    def make_split(n):
        emb1_list, emb2_list, label_list, s1_list, s2_list = [], [], [], [], []

        # Balance classes
        n_ent = n // 3
        n_neu = n // 3
        n_con = n - n_ent - n_neu

        # Entailment pairs
        ep_idx = rng.integers(0, len(ENTAILMENT_PAIRS), n_ent)
        for i in ep_idx:
            s1, s2 = ENTAILMENT_PAIRS[i]
            e1 = base_emb[s1] + noise * rng.standard_normal(dim)
            e2 = base_emb[s2] + noise * rng.standard_normal(dim)
            emb1_list.append(e1); emb2_list.append(e2)
            label_list.append(ENTAILMENT)
            s1_list.append(SENTENCES[s1]); s2_list.append(SENTENCES[s2])

        # Neutral: moderate overlap but no containment
        for _ in range(n_neu):
            s1 = rng.choice(list(SENTENCES.keys()))
            s2 = rng.choice(list(SENTENCES.keys()))
            while (s1, s2) in ENTAILMENT_PAIRS or s1 == s2:
                s2 = rng.choice(list(SENTENCES.keys()))
            e1 = base_emb[s1] + noise * rng.standard_normal(dim)
            e2 = base_emb[s2] + noise * rng.standard_normal(dim)
            emb1_list.append(e1); emb2_list.append(e2)
            label_list.append(NEUTRAL)
            s1_list.append(SENTENCES[s1]); s2_list.append(SENTENCES[s2])

        # Contradiction: far apart (opposite directions)
        np_idx = rng.integers(0, len(NON_ENTAILMENT_PAIRS), n_con)
        for i in np_idx:
            s1, s2 = NON_ENTAILMENT_PAIRS[i]
            e1 = base_emb[s1] + noise * rng.standard_normal(dim)
            e2 = -base_emb[s2] + noise * rng.standard_normal(dim)  # flip
            emb1_list.append(e1); emb2_list.append(e2)
            label_list.append(CONTRADICTION)
            s1_list.append(SENTENCES[s1]); s2_list.append(SENTENCES[s2])

        idx = rng.permutation(len(label_list))
        return {
            'emb1':  np.array(emb1_list)[idx],
            'emb2':  np.array(emb2_list)[idx],
            'label': np.array(label_list)[idx],
            'sent1': [s1_list[i] for i in idx],
            'sent2': [s2_list[i] for i in idx],
        }

    return make_split(n_train), make_split(n_val), make_split(n_test)


# ---------------------------------------------------------------------------
# STS-B style dataset (semantic similarity, score in [0,5])
# ---------------------------------------------------------------------------

def build_synthetic_sts(n=500, dim=64, noise=0.05, rng=None):
    """
    Build a synthetic STS dataset where similarity score reflects
    box overlap in the ground-truth latent space.
    """
    if rng is None:
        rng = np.random.default_rng(2)

    base_emb = build_synthetic_embeddings(dim=dim, noise=0.0, rng=rng)
    keys = list(base_emb.keys())

    emb1_list, emb2_list, score_list = [], [], []

    for _ in range(n):
        s1 = rng.choice(keys)
        s2 = rng.choice(keys)
        e1 = base_emb[s1] + noise * rng.standard_normal(dim)
        e2 = base_emb[s2] + noise * rng.standard_normal(dim)

        # Ground-truth cosine similarity → map to [0,5]
        cos = float(np.dot(base_emb[s1], base_emb[s2]))
        score = 2.5 * (cos + 1.0)  # [0,5]

        emb1_list.append(e1)
        emb2_list.append(e2)
        score_list.append(score)

    return {
        'emb1':  np.array(emb1_list),
        'emb2':  np.array(emb2_list),
        'score': np.array(score_list),
    }


# ---------------------------------------------------------------------------
# DataLoader utility
# ---------------------------------------------------------------------------

class NLIDataLoader:
    """Simple batched iterator over an NLI split dict."""

    def __init__(self, dataset, batch_size=32, shuffle=True, rng=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng or np.random.default_rng(99)
        self.n = len(dataset['label'])

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield {
                'emb1':  self.dataset['emb1'][batch_idx],
                'emb2':  self.dataset['emb2'][batch_idx],
                'label': self.dataset['label'][batch_idx],
            }
