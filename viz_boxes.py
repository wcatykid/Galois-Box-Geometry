"""
Box Geometry Visualization
============================
2D visualization of the "Tom goes" hierarchy boxes.
Uses PCA on midpoints for positioning, and mean box widths for size.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from src.dataset import (build_synthetic_dataset, build_synthetic_embeddings,
                          SENTENCES)
from src.box_lattice import box_join, box_containment_violation
from eval_join_loss import JoinAwareTrainer


def train_model(seed=42):
    base_emb = build_synthetic_embeddings(dim=64, noise=0.0,
                                           rng=np.random.default_rng(0))
    train_data, val_data, _ = build_synthetic_dataset(
        n_train=3000, n_val=500, n_test=1000,
        dim=64, noise=0.05, rng=np.random.default_rng(seed))
    trainer = JoinAwareTrainer(
        input_dim=64, hidden_dim=256, box_dim=64, lr=1e-3,
        lambda1=0.5, lambda2=2.0, lambda3=1.0, lambda4=0.3,
        margin=1.0, scale=5.0,
        lambda6=0.5, hierarchy_embs=base_emb,
        rng=np.random.default_rng(seed + 10))
    trainer.train(train_data, val_data, n_epochs=40,
                  batch_size=64, verbose=False, patience=10)
    return trainer.model, base_emb


def make_figure():
    print("Training model with join loss...")
    model, base_emb = train_model()

    sids = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
    labels = {
        's0': 'Something\nhappens',
        's1': 'Someone\nmoves',
        's2': 'Tom goes',
        's3': 'Tom goes\nsomewhere',
        's4': 'Tom goes\nto school',
        's5': 'East High\nSchool',
        's6': 'West High\nSchool',
        's7': 'Jefferson\nElementary',
        's8': 'Mary reads\na book',
        's9': 'Cat sits\non a mat',
    }

    # Encode all
    midpoints, widths_mean = [], []
    boxes_raw = {}
    for sid in sids:
        e = base_emb[sid][None, :]
        l, u, _ = model.encode(e)
        l, u = l[0], u[0]
        boxes_raw[sid] = (l, u)
        midpoints.append(0.5 * (l + u))
        widths_mean.append(np.mean(u - l))

    midpoints = np.array(midpoints)
    widths_mean = np.array(widths_mean)

    # PCA on midpoints for 2D positioning
    pca = PCA(n_components=2)
    centers_2d = pca.fit_transform(midpoints)

    # Normalize widths relative to largest box
    max_w = widths_mean.max()
    rel_widths = widths_mean / max_w

    # Scale factor for visual box size
    x_range = centers_2d[:, 0].max() - centers_2d[:, 0].min()
    base_size = x_range * 0.18

    colors = {
        's0': '#F4E4C1',
        's1': '#E8D5A8',
        's2': '#D4B98C',
        's3': '#C09A6B',
        's4': '#A67B5B',
        's5': '#2E86AB',
        's6': '#A23B72',
        's7': '#F18F01',
        's8': '#7FB069',
        's9': '#59A14F',
    }
    edge_colors = {
        's0': '#8B7355', 's1': '#8B7355', 's2': '#8B5E3C',
        's3': '#8B5E3C', 's4': '#6B4226',
        's5': '#1B5172', 's6': '#6B1E4A', 's7': '#B86B01',
        's8': '#4A7A3B', 's9': '#3A7A2B',
    }

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Draw boxes: back-to-front (largest first)
    draw_order = ['s0', 's1', 's2', 's3', 's4', 's8', 's9', 's5', 's6', 's7']

    box_rects = {}
    for sid in draw_order:
        idx = sids.index(sid)
        cx, cy = centers_2d[idx]
        half = base_size * rel_widths[idx]
        x0, y0 = cx - half, cy - half
        w = h = 2 * half

        alpha = 0.20 if sid in ['s0', 's1'] else 0.30
        lw = 2.5

        rect = patches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.05",
            linewidth=lw,
            edgecolor=edge_colors[sid],
            facecolor=colors[sid],
            alpha=alpha,
            zorder=draw_order.index(sid) + 1
        )
        ax.add_patch(rect)
        box_rects[sid] = (x0, y0, w, h, cx, cy, half)

    # Labels — position so they don't overlap
    label_offsets = {
        's0': ('edge_bl', 8.5),
        's1': ('edge_tl', 8.5),
        's2': ('center', 9.0),
        's3': ('center', 9.0),
        's4': ('center', 9.0),
        's5': ('center', 8.0),
        's6': ('center', 8.0),
        's7': ('center', 8.0),
        's8': ('center', 9.0),
        's9': ('center', 9.0),
    }
    for sid in draw_order:
        x0, y0, w, h, cx, cy, half = box_rects[sid]
        mode, fontsize = label_offsets[sid]

        if mode == 'edge_bl':
            lx, ly = x0 + 0.08, y0 + 0.12
            ha, va = 'left', 'bottom'
        elif mode == 'edge_tl':
            lx, ly = x0 + 0.08, y0 + h - 0.08
            ha, va = 'left', 'top'
        else:
            lx, ly = cx, cy
            ha, va = 'center', 'center'

        ax.text(lx, ly, labels[sid],
                ha=ha, va=va, fontsize=fontsize,
                fontweight='bold', color='#1a1a1a',
                zorder=20,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.85, edgecolor='none'))

    # Join visualization: dashed red box for join(s5, s6)
    idx5 = sids.index('s5')
    idx6 = sids.index('s6')
    # Join box spans both children
    c5x, c5y, h5 = centers_2d[idx5, 0], centers_2d[idx5, 1], base_size * rel_widths[idx5]
    c6x, c6y, h6 = centers_2d[idx6, 0], centers_2d[idx6, 1], base_size * rel_widths[idx6]
    jx0 = min(c5x - h5, c6x - h6)
    jy0 = min(c5y - h5, c6y - h6)
    jx1 = max(c5x + h5, c6x + h6)
    jy1 = max(c5y + h5, c6y + h6)

    join_rect = patches.FancyBboxPatch(
        (jx0, jy0), jx1 - jx0, jy1 - jy0,
        boxstyle="round,pad=0.05",
        linewidth=2.5, linestyle='--',
        edgecolor='#E63946', facecolor='#E63946',
        alpha=0.08, zorder=15
    )
    ax.add_patch(join_rect)

    ax.text((jx0 + jx1) / 2, jy1 + 0.25,
            'join(East HS, West HS) $\\sqsubseteq$ school',
            ha='center', va='bottom', fontsize=10,
            fontstyle='italic', color='#E63946', fontweight='bold',
            zorder=25)

    # Entailment arrows: specific → general
    entail_edges = [
        ('s5', 's4'), ('s6', 's4'), ('s7', 's4'),
        ('s4', 's3'), ('s3', 's2'), ('s2', 's1'), ('s1', 's0'),
    ]
    for child, parent in entail_edges:
        ci = sids.index(child)
        pi = sids.index(parent)
        ax.annotate('',
                    xy=centers_2d[pi],
                    xytext=centers_2d[ci],
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                    lw=1.2, alpha=0.5,
                                    connectionstyle='arc3,rad=0.1'),
                    zorder=0)

    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title('Learned Box Geometry: "Tom Goes" Hierarchy\n'
                 'Larger boxes = more general concepts; containment = entailment',
                 fontsize=14, fontweight='bold', pad=15)

    legend_elements = [
        patches.Patch(facecolor='#C09A6B', alpha=0.35, edgecolor='#8B5E3C',
                      linewidth=2, label='Hierarchy chain (s0-s4)'),
        patches.Patch(facecolor='#2E86AB', alpha=0.35, edgecolor='#1B5172',
                      linewidth=2, label='Leaf: East High School'),
        patches.Patch(facecolor='#A23B72', alpha=0.35, edgecolor='#6B1E4A',
                      linewidth=2, label='Leaf: West High School'),
        patches.Patch(facecolor='#F18F01', alpha=0.35, edgecolor='#B86B01',
                      linewidth=2, label='Leaf: Jefferson Elementary'),
        patches.Patch(facecolor='#7FB069', alpha=0.35, edgecolor='#4A7A3B',
                      linewidth=2, label='Unrelated sentences'),
        Line2D([0], [0], color='#E63946', linewidth=2.5, linestyle='--',
               label='Lattice join'),
        Line2D([0], [0], color='#555555', linewidth=1.2, alpha=0.6,
               marker='>', markersize=5,
               label='Entailment direction'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.92, edgecolor='#cccccc')

    margin_x = base_size * 0.5
    margin_y = base_size * 0.5
    all_xs = [box_rects[s][0] for s in sids] + [box_rects[s][0] + box_rects[s][2] for s in sids]
    all_ys = [box_rects[s][1] for s in sids] + [box_rects[s][1] + box_rects[s][3] for s in sids]
    ax.set_xlim(min(all_xs) - margin_x, max(all_xs) + margin_x)
    ax.set_ylim(min(all_ys) - margin_y, max(all_ys) + margin_y + 0.5)

    ax.grid(True, alpha=0.1, linestyle='-')
    ax.set_facecolor('#FAFAFA')
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'box_hierarchy_viz.png')
    fig.savefig(out, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {out}")

    pdf = out.replace('.png', '.pdf')
    fig.savefig(pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {pdf}")
    plt.close()


if __name__ == '__main__':
    make_figure()
