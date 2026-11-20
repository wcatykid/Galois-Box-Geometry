"""
Box Lattice Implementation
===========================
Implements the abstract domain A as a set of axis-aligned boxes in R^d.
Boxes form a complete lattice under the containment order:
  [l1,u1] ⊑ [l2,u2]  ⟺  l2 ≤ l1  and  u1 ≤ u2

This module is backend-agnostic: it works with plain NumPy arrays.
The PyTorch version (box_lattice_torch.py) wraps these operations as
differentiable functions.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Box operations (NumPy)
# ---------------------------------------------------------------------------

def box_containment_violation(l1, u1, l2, u2):
    """
    Compute the violation I([l1,u1], [l2,u2]) measuring how far
    [l1,u1] is from being contained in [l2,u2].

    Containment holds when l2 ≤ l1 and u1 ≤ u2.
    Violation = ||max(0, l2 - l1)|| + ||max(0, u1 - u2)||

    Args:
        l1, u1: lower/upper of inner box  shape (..., d)
        l2, u2: lower/upper of outer box  shape (..., d)
    Returns:
        scalar violation ≥ 0;  0 means l1,u1 ⊑ l2,u2
    """
    lower_viol = np.maximum(0.0, l2 - l1)
    upper_viol = np.maximum(0.0, u1 - u2)
    return np.sum(lower_viol + upper_viol, axis=-1)


def box_containment_score(l1, u1, l2, u2, scale=10.0):
    """
    Soft containment score in [0,1].
    score → 1  when [l1,u1] ⊆ [l2,u2]
    score → 0  when heavily violated
    """
    viol = box_containment_violation(l1, u1, l2, u2)
    return 1.0 / (1.0 + scale * viol)


def box_join(l1, u1, l2, u2):
    """Lattice join (least upper bound): smallest box containing both."""
    return np.minimum(l1, l2), np.maximum(u1, u2)


def box_meet(l1, u1, l2, u2):
    """Lattice meet (greatest lower bound): largest box inside both."""
    l = np.maximum(l1, l2)
    u = np.minimum(u1, u2)
    valid = np.all(l <= u, axis=-1, keepdims=True)
    return l, u, valid  # valid=False means empty intersection


def box_volume(l, u):
    """Log-volume of a box (sum of log side lengths)."""
    widths = np.maximum(u - l, 1e-8)
    return np.sum(np.log(widths), axis=-1)


def box_overlap_ratio(l1, u1, l2, u2):
    """Intersection-over-union in box space (scalar or batched)."""
    meet_l, meet_u, valid = box_meet(l1, u1, l2, u2)
    join_l, join_u = box_join(l1, u1, l2, u2)
    inter = np.where(valid.squeeze(), np.exp(box_volume(meet_l, meet_u)), 0.0)
    union = np.exp(box_volume(join_l, join_u))
    return inter / (union + 1e-8)


def box_midpoint(l, u):
    """Centroid of a box."""
    return 0.5 * (l + u)


def box_width(l, u):
    """Per-dimension widths (proxy for uncertainty/vagueness)."""
    return u - l


def is_valid_box(l, u):
    """Check l ≤ u coordinate-wise."""
    return np.all(l <= u, axis=-1)


# ---------------------------------------------------------------------------
# Complete lattice axiom checks (for unit tests / validation)
# ---------------------------------------------------------------------------

def check_reflexivity(l, u, tol=1e-6):
    """[l,u] ⊑ [l,u]"""
    viol = box_containment_violation(l, u, l, u)
    return np.all(viol < tol)


def check_antisymmetry(l1, u1, l2, u2, tol=1e-6):
    """If [l1,u1] ⊑ [l2,u2] and [l2,u2] ⊑ [l1,u1] then they're equal."""
    v12 = box_containment_violation(l1, u1, l2, u2)
    v21 = box_containment_violation(l2, u2, l1, u1)
    both_contained = (v12 < tol) & (v21 < tol)
    equal = np.allclose(l1, l2, atol=tol) and np.allclose(u1, u2, atol=tol)
    return (not np.any(both_contained)) or equal


def check_transitivity(l1, u1, l2, u2, l3, u3, tol=1e-6):
    """If a ⊑ b and b ⊑ c then a ⊑ c."""
    v12 = box_containment_violation(l1, u1, l2, u2)
    v23 = box_containment_violation(l2, u2, l3, u3)
    v13 = box_containment_violation(l1, u1, l3, u3)
    # wherever 12 and 23 hold, 13 must hold
    mask = (v12 < tol) & (v23 < tol)
    return np.all(v13[mask] < tol)
