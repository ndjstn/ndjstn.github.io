from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_DIR = Path('assets/img/posts/neural-network-components')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'bg': '#f8fafc',
    'panel': '#ffffff',
    'ink': '#0f172a',
    'muted': '#475569',
    'line': '#94a3b8',
    'blue': '#38bdf8',
    'blue_dark': '#0ea5e9',
    'green': '#22c55e',
    'green_dark': '#16a34a',
    'amber': '#f59e0b',
    'amber_dark': '#d97706',
    'red': '#ef4444',
    'red_dark': '#dc2626',
    'purple': '#a855f7',
}

sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.titlesize': 18,
    'axes.labelcolor': COLORS['muted'],
    'text.color': COLORS['ink'],
    'axes.edgecolor': '#e2e8f0',
    'axes.facecolor': COLORS['panel'],
    'figure.facecolor': COLORS['bg'],
    'xtick.color': COLORS['muted'],
    'ytick.color': COLORS['muted'],
})


def add_rounded_panel(ax, xy=(0.02, 0.05), width=0.96, height=0.9):
    panel = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle='round,pad=0.02,rounding_size=0.03',
        facecolor=COLORS['panel'],
        edgecolor='#dbe4ee',
        linewidth=1.5,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(panel)


def title_block(fig, title, subtitle):
    fig.text(0.06, 0.955, title, fontsize=24, fontweight='bold', color=COLORS['ink'])
    fig.text(0.06, 0.905, subtitle, fontsize=12.5, color=COLORS['muted'])


def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def image_explanation_problem():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    add_rounded_panel(ax)
    title_block(fig, 'Most explanations start in the wrong place', 'Vocabulary first creates confusion. Job-first explanations create intuition.')

    left = FancyBboxPatch((0.08, 0.18), 0.34, 0.60, boxstyle='round,pad=0.02,rounding_size=0.03',
                          facecolor='#fff1f2', edgecolor='#fecdd3', linewidth=1.5)
    right = FancyBboxPatch((0.58, 0.18), 0.34, 0.60, boxstyle='round,pad=0.02,rounding_size=0.03',
                           facecolor='#f0fdf4', edgecolor='#bbf7d0', linewidth=1.5)
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.25, 0.73, 'Vocabulary-first', ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['red_dark'])
    ax.text(0.75, 0.73, 'Job-first', ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['green_dark'])

    terms = ['neuron', 'bias', 'activation', 'backprop']
    for idx, term in enumerate(terms):
        y = 0.62 - idx * 0.11
        box = FancyBboxPatch((0.13, y), 0.18, 0.07, boxstyle='round,pad=0.02,rounding_size=0.03',
                             facecolor='white', edgecolor='#fda4af', linewidth=1.2)
        ax.add_patch(box)
        ax.text(0.22, y + 0.035, term, ha='center', va='center', fontsize=12.5)
        ax.add_patch(FancyArrowPatch((0.31, y + 0.035), (0.36, 0.47), arrowstyle='->', mutation_scale=12,
                                     linewidth=1.5, color='#fb7185', alpha=0.75))

    confusion = Circle((0.36, 0.47), 0.07, facecolor='#ffe4e6', edgecolor='#fb7185', linewidth=2)
    ax.add_patch(confusion)
    ax.text(0.36, 0.47, 'confusion', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['red_dark'])

    steps = [
        (0.67, 0.56, 'input', COLORS['blue']),
        (0.84, 0.56, 'score what matters', COLORS['purple']),
        (0.67, 0.38, 'build a better\nrepresentation', COLORS['green']),
        (0.84, 0.38, 'make a\nprediction', COLORS['amber']),
    ]
    for x, y, label, color in steps:
        box = FancyBboxPatch((x - 0.075, y - 0.055), 0.15, 0.11, boxstyle='round,pad=0.02,rounding_size=0.03',
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11.5, color=COLORS['ink'])

    arrows = [
        ((0.745, 0.56), (0.765, 0.56)),
        ((0.84, 0.50), (0.84, 0.44)),
        ((0.745, 0.38), (0.765, 0.38)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=13,
                                     linewidth=1.8, color=COLORS['line']))

    ax.text(0.75, 0.24, 'Same system. Much better entry point.', ha='center', fontsize=13, color=COLORS['green_dark'])
    save(fig, 'explanation-problem.png')


def image_neuron_scoring_rule():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    add_rounded_panel(ax)
    title_block(fig, 'What a neuron actually does', 'It is not magic. It is a small scoring rule repeated at scale.')

    inputs = [(0.14, 0.65, 'x₁'), (0.14, 0.50, 'x₂'), (0.14, 0.35, 'x₃')]
    for i, (x, y, label) in enumerate(inputs, start=1):
        ax.add_patch(Circle((x, y), 0.05, facecolor='#e0f2fe', edgecolor=COLORS['blue_dark'], linewidth=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=20, fontweight='bold')
        ax.add_patch(FancyArrowPatch((x + 0.05, y), (0.42, y), arrowstyle='->', mutation_scale=13,
                                     linewidth=2, color=COLORS['line']))
        ax.text(0.28, y + 0.04, f'w{i}', ha='center', va='center', fontsize=13, color=COLORS['muted'])

    ax.add_patch(Circle((0.50, 0.50), 0.10, facecolor='#dcfce7', edgecolor=COLORS['green_dark'], linewidth=2.5))
    ax.text(0.50, 0.53, 'Σ(wx)', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.50, 0.46, '+ b', ha='center', va='center', fontsize=18, color=COLORS['green_dark'])

    ax.add_patch(FancyArrowPatch((0.50, 0.78), (0.50, 0.60), arrowstyle='->', mutation_scale=13,
                                 linewidth=2, color=COLORS['purple']))
    ax.add_patch(Circle((0.50, 0.83), 0.04, facecolor='#f3e8ff', edgecolor=COLORS['purple'], linewidth=2))
    ax.text(0.50, 0.83, 'b', ha='center', va='center', fontsize=18, fontweight='bold')

    activation_box = FancyBboxPatch((0.66, 0.42), 0.15, 0.16, boxstyle='round,pad=0.02,rounding_size=0.03',
                                    facecolor='#fef3c7', edgecolor=COLORS['amber_dark'], linewidth=2)
    ax.add_patch(activation_box)
    ax.text(0.735, 0.51, 'activation', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.735, 0.45, 'ReLU / sigmoid / tanh', ha='center', va='center', fontsize=10.5, color=COLORS['muted'])
    ax.add_patch(FancyArrowPatch((0.60, 0.50), (0.66, 0.50), arrowstyle='->', mutation_scale=13,
                                 linewidth=2, color=COLORS['line']))

    ax.add_patch(FancyArrowPatch((0.81, 0.50), (0.89, 0.50), arrowstyle='->', mutation_scale=13,
                                 linewidth=2, color=COLORS['line']))
    ax.add_patch(Circle((0.92, 0.50), 0.045, facecolor='#fee2e2', edgecolor=COLORS['red_dark'], linewidth=2))
    ax.text(0.92, 0.50, 'y', ha='center', va='center', fontsize=18, fontweight='bold')

    ax.text(0.50, 0.20, 'Input signals come in, get weighted, shifted, activated, and passed on.',
            ha='center', fontsize=13, color=COLORS['muted'])
    save(fig, 'neuron-scoring-rule.png')


def image_weights_biases():
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['bg'])
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.97, bottom=0.10, top=0.84, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    title_block(fig, 'Weights decide what matters. Biases decide when it matters.', 'One controls influence. The other shifts the response threshold.')

    features = ['keyword', 'sender score', 'link count', 'weird formatting', 'known contact']
    values = np.array([0.92, 0.58, 0.71, 0.44, -0.62])
    colors = [COLORS['blue'], COLORS['green'], COLORS['amber'], COLORS['purple'], COLORS['red']]
    ax1.barh(features, values, color=colors)
    ax1.axvline(0, color=COLORS['line'], linewidth=1.5)
    ax1.set_title('Weights = influence', loc='left', pad=12)
    ax1.set_xlabel('how much the model cares')
    ax1.grid(axis='x', alpha=0.25)
    sns.despine(ax=ax1, left=True, bottom=True)

    x = np.linspace(-6, 6, 300)
    logistic = 1 / (1 + np.exp(-(x - 1.5)))
    shifted = 1 / (1 + np.exp(-(x + 0.7)))
    ax2.plot(x, logistic, color=COLORS['blue_dark'], linewidth=3, label='with one bias')
    ax2.plot(x, shifted, color=COLORS['green_dark'], linewidth=3, linestyle='--', label='bias shifted left')
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title('Bias = threshold shift', loc='left', pad=12)
    ax2.set_xlabel('weighted sum before activation')
    ax2.set_ylabel('response strength')
    ax2.legend(frameon=False, loc='lower right')
    ax2.grid(alpha=0.25)
    sns.despine(ax=ax2)

    save(fig, 'weights-biases.png')


def image_activation_functions():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.subplots_adjust(top=0.80, left=0.08, right=0.97, bottom=0.12)
    x = np.linspace(-4.5, 4.5, 400)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    ax.plot(x, relu, color=COLORS['amber_dark'], linewidth=3, label='ReLU')
    ax.plot(x, sigmoid, color=COLORS['blue_dark'], linewidth=3, label='Sigmoid')
    ax.plot(x, tanh, color=COLORS['green_dark'], linewidth=3, label='Tanh')
    ax.axhline(0, color=COLORS['line'], linewidth=1)
    ax.axvline(0, color=COLORS['line'], linewidth=1)
    ax.set_title('Activation functions are where the model stops being linear', loc='left', pad=15)
    ax.text(0.0, 1.06, 'Different curves, same big job: add non-linearity so the model can bend.',
            transform=ax.transAxes, fontsize=12.5, color=COLORS['muted'])
    ax.set_xlabel('input to the activation')
    ax.set_ylabel('output response')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.grid(alpha=0.22)
    sns.despine(ax=ax)
    save(fig, 'activation-functions.png')


def image_representation_building():
    rng = np.random.default_rng(7)
    n = 180
    theta = rng.uniform(0, np.pi, n // 2)
    moon1 = np.column_stack([np.cos(theta), np.sin(theta)]) + rng.normal(scale=0.12, size=(n // 2, 2))
    moon2 = np.column_stack([1 - np.cos(theta), -np.sin(theta) - 0.45]) + rng.normal(scale=0.12, size=(n // 2, 2))
    raw = np.vstack([moon1, moon2])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    layer1 = np.column_stack([raw[:, 0], raw[:, 1] ** 2 + 0.2 * raw[:, 0]])
    layer2 = np.column_stack([layer1[:, 0] + 0.7 * layer1[:, 1], layer1[:, 1] - 0.5 * layer1[:, 0]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=COLORS['bg'])
    fig.subplots_adjust(top=0.77, left=0.06, right=0.98, bottom=0.10, wspace=0.20)
    title_block(fig, 'Hidden layers turn raw input into usable features', 'The point of depth is not mystique. It is better internal representations.')
    stages = [
        ('raw signal', raw),
        ('after a hidden layer', layer1),
        ('later representation', layer2),
    ]
    palette = np.array([COLORS['blue_dark'], COLORS['red_dark']])

    for ax, (label, data) in zip(axes, stages):
        ax.scatter(data[:, 0], data[:, 1], c=palette[labels], s=32, alpha=0.82, edgecolor='white', linewidth=0.35)
        ax.set_title(label, fontsize=15, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.12)
        sns.despine(ax=ax, left=True, bottom=True)

    save(fig, 'representation-building.png')


def image_backprop_blame_assignment():
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['bg'])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], left=0.05, right=0.97, bottom=0.10, top=0.84, hspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    title_block(fig, 'Backpropagation is blame assignment', 'Predict, measure the miss, send that signal backward, and update the parts that caused it.')

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis('off')
    steps = [
        (0.12, 'predict', COLORS['blue']),
        (0.37, 'compare with truth', COLORS['amber']),
        (0.62, 'assign blame', COLORS['red']),
        (0.87, 'update weights', COLORS['green']),
    ]
    for i, (x, label, color) in enumerate(steps):
        box = FancyBboxPatch((x - 0.09, 0.38), 0.18, 0.20, boxstyle='round,pad=0.02,rounding_size=0.03',
                             facecolor='white', edgecolor=color, linewidth=2)
        ax0.add_patch(box)
        ax0.text(x, 0.48, label, ha='center', va='center', fontsize=14, fontweight='bold')
        if i < len(steps) - 1:
            ax0.add_patch(FancyArrowPatch((x + 0.10, 0.48), (steps[i + 1][0] - 0.10, 0.48),
                                          arrowstyle='->', mutation_scale=13, linewidth=2, color=COLORS['line']))

    epochs = np.arange(1, 31)
    loss = 1.15 * np.exp(-epochs / 9) + 0.06 * np.sin(epochs / 2.4) + 0.08
    ax1.plot(epochs, loss, color=COLORS['blue_dark'], linewidth=3)
    ax1.fill_between(epochs, loss, color=COLORS['blue'], alpha=0.18)
    ax1.set_title('Learning is iterative correction', loc='left', pad=10)
    ax1.set_xlabel('training step')
    ax1.set_ylabel('loss')
    ax1.grid(alpha=0.22)
    sns.despine(ax=ax1)
    save(fig, 'backprop-blame-assignment.png')


def image_debugging_checklist():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    add_rounded_panel(ax)
    title_block(fig, 'How I actually think about a model when it is failing', 'The checklist is usually more useful than memorizing more vocabulary.')

    items = [
        ('task', 'What is the model actually trying to predict?', COLORS['blue']),
        ('signals', 'Which inputs should matter most for this job?', COLORS['green']),
        ('capacity', 'Is the model too weak or too big for the data?', COLORS['amber']),
        ('feedback', 'Is the learning signal clean enough to help?', COLORS['red']),
        ('data', 'Is the real problem the data, not the architecture?', COLORS['purple']),
    ]
    y_positions = [0.74, 0.60, 0.46, 0.32, 0.18]
    for (label, question, color), y in zip(items, y_positions):
        ax.add_patch(FancyBboxPatch((0.10, y - 0.05), 0.80, 0.09, boxstyle='round,pad=0.02,rounding_size=0.03',
                                    facecolor='white', edgecolor=color, linewidth=2))
        ax.text(0.16, y, label, ha='left', va='center', fontsize=13, fontweight='bold', color=color)
        ax.text(0.29, y, question, ha='left', va='center', fontsize=12.5, color=COLORS['ink'])

    save(fig, 'debugging-checklist.png')


def main():
    image_explanation_problem()
    image_neuron_scoring_rule()
    image_weights_biases()
    image_activation_functions()
    image_representation_building()
    image_backprop_blame_assignment()
    image_debugging_checklist()
    print('Generated images in', OUT_DIR)


if __name__ == '__main__':
    main()
