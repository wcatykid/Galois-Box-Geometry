"""
Encoder / Abstraction Function  αθ : C → A
============================================
Maps a sentence (represented as a pre-computed embedding vector) to a
box [l, u] in R^d.

Architecture:
  h  = MLP(embedding)           # shared trunk
  l  = W_l h + b_l              # lower bound
  δ  = softplus(W_u h + b_u)    # non-negative offset
  u  = l + δ                    # upper bound  (guarantees l ≤ u)

For the full PyTorch version with a Transformer backbone see
`encoder_torch.py`.  This NumPy version is used for fast experimentation
and CPU-only environments.
"""

import numpy as np
from src.box_lattice import box_midpoint


def softplus(x, beta=1.0):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0.0, x)


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mu) / (std + eps)


# ---------------------------------------------------------------------------
# Parameter initialisation helpers
# ---------------------------------------------------------------------------

def glorot_uniform(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def init_encoder_params(input_dim, hidden_dim, box_dim, rng=None):
    """
    Returns a dict of weight matrices and biases for the encoder MLP.

    Network structure:
        input_dim  →  hidden_dim  →  hidden_dim  →  2*box_dim
    Then split into lower head (box_dim) and upper-offset head (box_dim).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    params = {
        # Trunk layers
        'W1': glorot_uniform(input_dim, hidden_dim, rng),
        'b1': np.zeros(hidden_dim),
        'W2': glorot_uniform(hidden_dim, hidden_dim, rng),
        'b2': np.zeros(hidden_dim),
        # Lower bound head
        'Wl': glorot_uniform(hidden_dim, box_dim, rng),
        'bl': np.zeros(box_dim),
        # Upper offset head (passed through softplus)
        'Wu': glorot_uniform(hidden_dim, box_dim, rng),
        'bu': np.full(box_dim, -1.0),   # slight negative bias → small initial boxes
    }
    return params


def encode(x, params, dropout=0.0, rng=None):
    """
    Forward pass of αθ.

    Args:
        x:      np.ndarray  (..., input_dim)  input embedding(s)
        params: dict of weight matrices from init_encoder_params
        dropout: float, dropout rate (applied during training only)
        rng:    np.random.Generator (needed when dropout > 0)

    Returns:
        l: np.ndarray (..., box_dim)  lower bounds
        u: np.ndarray (..., box_dim)  upper bounds
        h: np.ndarray (..., hidden_dim) intermediate representation
    """
    # Layer 1
    h = x @ params['W1'] + params['b1']
    h = layer_norm(relu(h))
    if dropout > 0 and rng is not None:
        mask = (rng.random(h.shape) > dropout).astype(float)
        h = h * mask / (1.0 - dropout)

    # Layer 2
    h = h @ params['W2'] + params['b2']
    h = layer_norm(relu(h))
    if dropout > 0 and rng is not None:
        mask = (rng.random(h.shape) > dropout).astype(float)
        h = h * mask / (1.0 - dropout)

    # Lower bound
    l = h @ params['Wl'] + params['bl']

    # Non-negative offset → upper bound
    delta = softplus(h @ params['Wu'] + params['bu'])
    u = l + delta

    return l, u, h


# ---------------------------------------------------------------------------
# Concretization function  γϕ : A → embedding space
# ---------------------------------------------------------------------------

def init_decoder_params(box_dim, hidden_dim, output_dim, rng=None):
    """
    Simple MLP decoder: box midpoint → reconstructed embedding.
    """
    if rng is None:
        rng = np.random.default_rng(43)

    params = {
        'W1': glorot_uniform(box_dim, hidden_dim, rng),
        'b1': np.zeros(hidden_dim),
        'W2': glorot_uniform(hidden_dim, output_dim, rng),
        'b2': np.zeros(output_dim),
    }
    return params


def decode(l, u, params):
    """
    Forward pass of γϕ.

    Args:
        l, u: lower/upper bounds (..., box_dim)
        params: dict from init_decoder_params

    Returns:
        x_hat: reconstructed embedding (..., output_dim)
    """
    z = box_midpoint(l, u)
    h = relu(z @ params['W1'] + params['b1'])
    x_hat = h @ params['W2'] + params['b2']
    return x_hat
