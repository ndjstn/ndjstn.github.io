from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_DIR = Path('assets/img/posts/neural-network-components')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'bg': '#f8fafc',
    'panel': '#ffffff',
    'ink': '#0f172a',
    'muted': '#475569',
    'line': '#94a3b8',
    'blue': '#0ea5e9',
    'green': '#16a34a',
    'amber': '#d97706',
    'red': '#dc2626',
    'purple': '#7c3aed',
    'soft_blue': '#e8f4fb',
    'soft_red': '#fdecec',
    'soft_green': '#eefbf3',
    'soft_amber': '#fff7ed',
}

plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'axes.facecolor': COLORS['panel'],
        'figure.facecolor': COLORS['bg'],
        'axes.edgecolor': '#e2e8f0',
        'axes.labelcolor': COLORS['muted'],
        'xtick.color': COLORS['muted'],
        'ytick.color': COLORS['muted'],
        'text.color': COLORS['ink'],
    }
)


def add_card(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle='round,pad=0.02,rounding_size=0.03',
            facecolor=COLORS['panel'],
            edgecolor='#dbe4ee',
            linewidth=1.4,
            transform=ax.transAxes,
        )
    )


def variant_a(ax):
    add_card(ax)
    ax.text(0.08, 0.92, 'Variant A — Structural', fontsize=16, fontweight='bold')
    ax.text(0.08, 0.875, 'Best for: first-time readers', fontsize=10.5, color=COLORS['muted'])

    inputs = [
        (0.14, 0.68, 'x₁', '0.90'),
        (0.14, 0.51, 'x₂', '0.35'),
        (0.14, 0.34, 'x₃', '0.80'),
    ]
    for x, y, label, value in inputs:
        ax.add_patch(Circle((x, y), 0.055, facecolor=COLORS['soft_blue'], edgecolor=COLORS['blue'], linewidth=1.8))
        ax.text(x, y + 0.012, label, ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(x, y - 0.035, value, ha='center', va='center', fontsize=9.5, color=COLORS['muted'])
        ax.add_patch(FancyArrowPatch((x + 0.055, y), (0.42, y), arrowstyle='->', mutation_scale=12, linewidth=1.6, color=COLORS['line']))

    weight_labels = ['w₁ pulls up', 'w₂ pulls down', 'w₃ pulls up']
    for (_, y, _, _), label in zip(inputs, weight_labels):
        ax.text(0.30, y + 0.045, label, ha='center', va='center', fontsize=9.7, color=COLORS['muted'])

    ax.add_patch(Circle((0.50, 0.51), 0.11, facecolor='#f8fafc', edgecolor=COLORS['green'], linewidth=2.2))
    ax.text(0.50, 0.535, 'score', ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(0.50, 0.48, r'$z = \, \, \, + b$', ha='center', va='center', fontsize=12, color=COLORS['muted'])

    ax.add_patch(Circle((0.50, 0.83), 0.042, facecolor='#f3e8ff', edgecolor=COLORS['purple'], linewidth=1.8))
    ax.text(0.50, 0.83, 'b', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.add_patch(FancyArrowPatch((0.50, 0.79), (0.50, 0.62), arrowstyle='->', mutation_scale=12, linewidth=1.5, color=COLORS['purple']))
    ax.text(0.58, 0.75, 'bias shifts the whole score', fontsize=9.5, color=COLORS['muted'])

    ax.add_patch(FancyArrowPatch((0.61, 0.51), (0.77, 0.51), arrowstyle='->', mutation_scale=12, linewidth=1.6, color=COLORS['line']))
    ax.plot([0.82, 0.82], [0.30, 0.72], color=COLORS['amber'], linewidth=2.2)
    ax.text(0.85, 0.72, 'threshold', fontsize=9.5, color=COLORS['amber'])
    ax.scatter([0.88], [0.56], s=110, color=COLORS['green'])
    ax.text(0.88, 0.50, 'fires', ha='center', fontsize=10)
    ax.text(0.08, 0.11, 'Shows the parts and the order of operations clearly.', fontsize=10.8)
    ax.text(0.08, 0.07, 'Weak on exact magnitude and sign.', fontsize=10.1, color=COLORS['muted'])



def variant_b(ax):
    add_card(ax)
    ax.text(0.08, 0.92, 'Variant B — Quantitative', fontsize=16, fontweight='bold')
    ax.text(0.08, 0.875, 'Best for: readers who want the score to feel concrete', fontsize=10.5, color=COLORS['muted'])

    features = ['links', 'sender rep', 'urgent tone', 'known contact', 'bias']
    values = np.array([1.44, -0.42, 0.80, -0.08, -0.55])
    colors = [COLORS['blue'], COLORS['red'], COLORS['blue'], COLORS['red'], COLORS['purple']]
    y0 = 0.75
    for idx, (feature, value, color) in enumerate(zip(features, values, colors)):
        y = y0 - idx * 0.11
        ax.text(0.08, y, feature, va='center', fontsize=10.6)
        x_left = 0.34
        x_right = 0.74
        ax.plot([0.54, 0.54], [0.21, 0.79], color='#cbd5e1', linewidth=1)
        if value >= 0:
            ax.add_patch(FancyBboxPatch((0.54, y - 0.024), 0.16 * (value / 1.44), 0.048, boxstyle='round,pad=0.005,rounding_size=0.01', facecolor=color, edgecolor=color, linewidth=0))
            ax.text(0.54 + 0.16 * (value / 1.44) + 0.02, y, f'+{value:.2f}', va='center', fontsize=10.0)
        else:
            width = 0.16 * (abs(value) / 0.55)
            ax.add_patch(FancyBboxPatch((0.54 - width, y - 0.024), width, 0.048, boxstyle='round,pad=0.005,rounding_size=0.01', facecolor=color, edgecolor=color, linewidth=0))
            ax.text(0.54 - width - 0.02, y, f'{value:.2f}', va='center', ha='right', fontsize=10.0)

    total = values.sum()
    prob = 1 / (1 + np.exp(-total))
    ax.text(0.08, 0.22, rf'total score $z$ = {total:.2f}', fontsize=11.5, fontweight='bold')
    ax.text(0.08, 0.17, rf'activation $\sigma(z)$ = {prob:.2f}', fontsize=10.7, color=COLORS['muted'])

    x = np.linspace(-3.5, 3.5, 300)
    curve = 1 / (1 + np.exp(-x))
    x_norm = 0.74 + (x + 3.5) / 7 * 0.20
    y_norm = 0.18 + curve * 0.18
    ax.plot(x_norm, y_norm, color=COLORS['green'], linewidth=2)
    ax.plot([0.74, 0.94], [0.27, 0.27], color='#cbd5e1', linewidth=1, linestyle=':')
    marker_x = 0.74 + (total + 3.5) / 7 * 0.20
    marker_y = 0.18 + prob * 0.18
    ax.scatter([marker_x], [marker_y], s=45, color=COLORS['amber'])
    ax.text(0.74, 0.39, 'threshold shifts are visible here too', fontsize=9.5, color=COLORS['muted'])
    ax.text(0.08, 0.11, 'Shows sign, magnitude, and cumulative pull.', fontsize=10.8)
    ax.text(0.08, 0.07, 'Weak on the “same weights, moved boundary” intuition.', fontsize=10.1, color=COLORS['muted'])



def variant_c(ax):
    add_card(ax)
    ax.text(0.08, 0.92, 'Variant C — Dynamic', fontsize=16, fontweight='bold')
    ax.text(0.08, 0.875, 'Best for: people confused about what bias changes', fontsize=10.5, color=COLORS['muted'])

    frames = [(-1.2, 0.20), (0.0, 0.50), (1.2, 0.80)]
    for idx, (bias, x0) in enumerate(frames):
        left = 0.08 + idx * 0.28
        right = left + 0.22
        bottom, top = 0.25, 0.72
        ax.add_patch(FancyBboxPatch((left, bottom), 0.22, 0.47, boxstyle='round,pad=0.01,rounding_size=0.02', facecolor='white', edgecolor='#dbe4ee', linewidth=1.2))
        # same direction vector
        ax.arrow(left + 0.05, bottom + 0.07, 0.08, 0.12, width=0.004, color=COLORS['amber'], length_includes_head=True)
        ax.text(left + 0.14, bottom + 0.20, 'same\nweights', fontsize=8.8, color=COLORS['muted'])
        # parallel boundary line shift
        xs = np.array([left + 0.04, right - 0.03])
        ys = np.array([bottom + 0.05 + x0 * 0.18, top - 0.08 + x0 * 0.18])
        ax.plot(xs, ys, color=COLORS['ink'], linewidth=2.1)
        ax.text(left + 0.02, top + 0.02, f'b = {bias:+.1f}', fontsize=9.5)

    ax.text(0.08, 0.18, 'Three snapshots of the same rule with a different bias.', fontsize=10.8)
    ax.text(0.08, 0.13, 'The angle stays fixed. The boundary slides in parallel.', fontsize=10.8)
    ax.text(0.08, 0.07, 'Weak on the actual score buildup unless paired with another panel.', fontsize=10.1, color=COLORS['muted'])



def main():
    fig = plt.figure(figsize=(18, 7.8), facecolor=COLORS['bg'])
    fig.text(0.05, 0.955, 'Weights and bias: three ways to teach the same idea', fontsize=24, fontweight='bold', color=COLORS['ink'])
    fig.text(0.05, 0.918, 'Pick the version that makes the interaction clearest. We can also combine the best parts into a hybrid.', fontsize=12.2, color=COLORS['muted'])

    gs = fig.add_gridspec(1, 3, left=0.04, right=0.98, top=0.87, bottom=0.07, wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    variant_a(axes[0])
    variant_b(axes[1])
    variant_c(axes[2])

    out = OUT_DIR / 'weights-bias-variants-board.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print(f'generated {out}')


if __name__ == '__main__':
    main()
