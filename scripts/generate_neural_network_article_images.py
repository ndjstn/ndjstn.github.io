from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['bg'])
    outer = fig.add_gridspec(
        2,
        2,
        left=0.05,
        right=0.98,
        bottom=0.08,
        top=0.84,
        width_ratios=[1.15, 1.0],
        height_ratios=[1, 1],
        wspace=0.24,
        hspace=0.28,
    )
    left_top = outer[0, 0].subgridspec(1, 3, wspace=0.12)
    ax_input = fig.add_subplot(left_top[0, 0])
    ax_weight = fig.add_subplot(left_top[0, 1], sharey=ax_input)
    ax_contrib = fig.add_subplot(left_top[0, 2], sharey=ax_input)
    ax_score = fig.add_subplot(outer[1, 0])
    ax_curve = fig.add_subplot(outer[0, 1])
    ax_boundary = fig.add_subplot(outer[1, 1])

    title_block(
        fig,
        'Weights shape the rule. Bias shifts the whole rule.',
        'The useful picture is not “some bars and a curve.” It is feature values, signed influence, a total score, and a moved threshold.',
    )

    features = np.array(['contains links', 'sender reputation', 'urgent tone', 'known contact'])
    x_vals = np.array([0.90, 0.35, 0.80, 0.10])
    weights = np.array([1.60, -1.20, 1.00, -0.80])
    contributions = x_vals * weights
    bias = -0.55
    total = contributions.sum() + bias
    probability = 1 / (1 + np.exp(-total))

    order = np.arange(len(features))[::-1]
    features = features[order]
    x_vals = x_vals[order]
    weights = weights[order]
    contributions = contributions[order]
    y = np.arange(len(features))

    for ax, values, title, xlim in [
        (ax_input, x_vals, 'input $x_i$', (0, 1.05)),
        (ax_weight, weights, 'weight $w_i$', (-2.05, 2.05)),
        (ax_contrib, contributions, 'contribution $w_i x_i$', (-1.75, 1.75)),
    ]:
        bar_colors = [COLORS['blue_dark'] if value >= 0 else COLORS['red_dark'] for value in values]
        ax.barh(y, values, color=bar_colors, alpha=0.88)
        ax.set_xlim(*xlim)
        ax.set_title(title, loc='left', pad=10, fontsize=14)
        ax.grid(axis='x', alpha=0.20)
        ax.axvline(0, color=COLORS['line'], linewidth=1.3)
        sns.despine(ax=ax, left=True, bottom=True)
        for idx, value in enumerate(values):
            ha = 'left' if value >= 0 else 'right'
            offset = 0.03 * (xlim[1] - xlim[0]) * (1 if value >= 0 else -1)
            ax.text(value + offset, idx, f'{value:+.2f}' if xlim[0] < 0 else f'{value:.2f}', va='center', ha=ha, fontsize=10.5)

    ax_input.set_yticks(y, labels=features)
    ax_input.tick_params(axis='y', labelsize=11)
    ax_weight.tick_params(axis='y', left=False, labelleft=False)
    ax_contrib.tick_params(axis='y', left=False, labelleft=False)
    ax_input.text(
        0.0,
        1.08,
        'A concrete scoring example',
        transform=ax_input.transAxes,
        fontsize=13.5,
        fontweight='bold',
        color=COLORS['ink'],
    )

    steps = list(contributions[::-1]) + [bias]
    step_labels = list(features[::-1]) + ['bias']
    current = 0.0
    for idx, (label, delta) in enumerate(zip(step_labels, steps)):
        start = min(current, current + delta)
        color = COLORS['blue_dark'] if delta >= 0 else COLORS['red_dark']
        ax_score.bar(idx, abs(delta), bottom=start, color=color, width=0.62, alpha=0.82)
        ax_score.text(idx, current + delta + (0.08 if delta >= 0 else -0.14), f'{delta:+.2f}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=10.5)
        current += delta
    ax_score.bar(len(steps), total, color=COLORS['purple'], width=0.62, alpha=0.35, edgecolor=COLORS['purple'], linewidth=2)
    ax_score.text(len(steps), total + 0.10, f'$z$ = {total:.2f}\n$\\sigma(z)$ = {probability:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_score.axhline(0, color=COLORS['line'], linewidth=1.2)
    ax_score.set_xticks(np.arange(len(steps) + 1), [*step_labels, 'final score'], rotation=15, ha='right')
    ax_score.set_ylabel('pre-activation score $z$')
    ax_score.set_title('How the weighted evidence adds up', loc='left', pad=12, fontsize=14)
    ax_score.text(
        0.01,
        0.95,
        r'$z = \sum_i w_i x_i + b$',
        transform=ax_score.transAxes,
        fontsize=12.5,
        color=COLORS['muted'],
        va='top',
    )
    ax_score.grid(axis='y', alpha=0.18)
    sns.despine(ax=ax_score)

    raw_score = np.linspace(-4.0, 4.0, 400)
    bias_values = [(-1.2, COLORS['red_dark'], 'bias = -1.2'), (0.0, COLORS['ink'], 'bias = 0.0'), (1.2, COLORS['green_dark'], 'bias = +1.2')]
    ax_curve.axhline(0.5, color=COLORS['line'], linewidth=1.2, linestyle=':')
    sample_raw = 0.8
    for bias_value, color, label in bias_values:
        response = 1 / (1 + np.exp(-(raw_score + bias_value)))
        threshold = -bias_value
        sample_prob = 1 / (1 + np.exp(-(sample_raw + bias_value)))
        ax_curve.plot(raw_score, response, color=color, linewidth=3, label=label)
        ax_curve.axvline(threshold, color=color, linewidth=1.4, linestyle='--', alpha=0.75)
        ax_curve.scatter([sample_raw], [sample_prob], color=color, s=55, zorder=5)
    ax_curve.annotate(
        'Positive bias moves\nthe threshold left',
        xy=(-1.2, 0.52),
        xytext=(-3.4, 0.78),
        arrowprops={'arrowstyle': '->', 'color': COLORS['muted'], 'lw': 1.6},
        fontsize=11,
        color=COLORS['muted'],
    )
    ax_curve.set_title('Bias changes when the unit “switches on”', loc='left', pad=12, fontsize=14)
    ax_curve.set_xlabel(r'raw weighted sum $w^\top x$ before bias')
    ax_curve.set_ylabel(r'activation $\sigma(w^\top x + b)$')
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.legend(frameon=False, loc='lower right')
    ax_curve.grid(alpha=0.20)
    sns.despine(ax=ax_curve)

    boundary_weights = np.array([1.4, -1.0])
    xx, yy = np.meshgrid(np.linspace(-2.4, 2.4, 200), np.linspace(-2.1, 2.1, 200))
    mid_bias = 0.0
    surface = 1 / (1 + np.exp(-(boundary_weights[0] * xx + boundary_weights[1] * yy + mid_bias)))
    ax_boundary.contourf(xx, yy, surface, levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.28)
    boundary_biases = [(-0.8, COLORS['red_dark'], 'same weights, lower bias'), (0.8, COLORS['green_dark'], 'same weights, higher bias')]
    for bias_value, color, label in boundary_biases:
        ax_boundary.contour(xx, yy, boundary_weights[0] * xx + boundary_weights[1] * yy + bias_value, levels=[0], colors=[color], linewidths=3)
        ax_boundary.text(1.45, (boundary_weights[0] * 1.45 + bias_value) / (-boundary_weights[1]), label, color=color, fontsize=10.5, va='center')
    normal = boundary_weights / np.linalg.norm(boundary_weights)
    ax_boundary.annotate(
        'weight vector $w$\nsets boundary orientation',
        xy=(0.2 + 0.9 * normal[0], 0.2 + 0.9 * normal[1]),
        xytext=(-1.9, 1.4),
        arrowprops={'arrowstyle': '->', 'lw': 1.6, 'color': COLORS['muted']},
        fontsize=11,
        color=COLORS['muted'],
    )
    ax_boundary.annotate(
        'bias translates the boundary\nwithout rotating it',
        xy=(0.65, -0.15),
        xytext=(-1.95, -1.6),
        arrowprops={'arrowstyle': '->', 'lw': 1.6, 'color': COLORS['muted']},
        fontsize=11,
        color=COLORS['muted'],
    )
    ax_boundary.set_title('In 2D, weights rotate the split and bias slides it', loc='left', pad=12, fontsize=14)
    ax_boundary.set_xlabel('feature 1')
    ax_boundary.set_ylabel('feature 2')
    ax_boundary.grid(alpha=0.15)
    sns.despine(ax=ax_boundary)

    save(fig, 'weights-biases.png')


def image_weights_bias_animation():
    fig, (ax_curve, ax_boundary) = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=COLORS['bg'])
    fig.subplots_adjust(top=0.80, left=0.06, right=0.98, bottom=0.14, wspace=0.24)
    fig.text(0.06, 0.94, 'Bias is easier to understand when you watch it move', fontsize=22, fontweight='bold', color=COLORS['ink'])
    fig.text(0.06, 0.89, 'Weights stay fixed. The threshold and the decision boundary slide because the bias term changes the score everywhere at once.', fontsize=12.5, color=COLORS['muted'])

    raw_score = np.linspace(-4.0, 4.0, 400)
    sample_raw = 0.7
    boundary_weights = np.array([1.4, -1.0])
    xx, yy = np.meshgrid(np.linspace(-2.4, 2.4, 180), np.linspace(-2.1, 2.1, 180))
    frames = np.concatenate([np.linspace(-1.6, 1.3, 28), np.linspace(1.3, -1.6, 28)])

    def draw_frame(bias_value):
        ax_curve.clear()
        ax_boundary.clear()

        response = 1 / (1 + np.exp(-(raw_score + bias_value)))
        threshold = -bias_value
        sample_prob = 1 / (1 + np.exp(-(sample_raw + bias_value)))

        ax_curve.plot(raw_score, response, color=COLORS['blue_dark'], linewidth=3)
        ax_curve.axhline(0.5, color=COLORS['line'], linewidth=1.2, linestyle=':')
        ax_curve.axvline(threshold, color=COLORS['amber_dark'], linewidth=2.0, linestyle='--')
        ax_curve.scatter([sample_raw], [sample_prob], color=COLORS['red_dark'], s=90, zorder=5)
        ax_curve.annotate(
            f'bias = {bias_value:+.2f}\nthreshold = {-bias_value:+.2f}\nsample p = {sample_prob:.2f}',
            xy=(sample_raw, sample_prob),
            xytext=(-3.45, 0.80),
            arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': COLORS['muted']},
            bbox={'boxstyle': 'round,pad=0.35', 'fc': 'white', 'ec': '#cbd5e1'},
            fontsize=10.5,
            color=COLORS['ink'],
        )
        ax_curve.set_title('The activation threshold slides', loc='left', pad=10, fontsize=14)
        ax_curve.set_xlabel(r'raw weighted sum $w^\top x$')
        ax_curve.set_ylabel(r'activation $\sigma(w^\top x + b)$')
        ax_curve.set_ylim(-0.02, 1.02)
        ax_curve.grid(alpha=0.18)
        sns.despine(ax=ax_curve)

        surface = 1 / (1 + np.exp(-(boundary_weights[0] * xx + boundary_weights[1] * yy + bias_value)))
        ax_boundary.contourf(xx, yy, surface, levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.42)
        ax_boundary.contour(xx, yy, boundary_weights[0] * xx + boundary_weights[1] * yy + bias_value, levels=[0], colors=[COLORS['ink']], linewidths=3)
        normal = boundary_weights / np.linalg.norm(boundary_weights)
        ax_boundary.arrow(0.2, 0.2, 0.7 * normal[0], 0.7 * normal[1], width=0.02, color=COLORS['amber_dark'], length_includes_head=True)
        ax_boundary.text(-2.15, 1.65, 'same weights → same angle', fontsize=10.5, color=COLORS['muted'])
        ax_boundary.text(-2.15, 1.38, 'changing bias → translated line', fontsize=10.5, color=COLORS['muted'])
        ax_boundary.set_title('The decision boundary moves in parallel', loc='left', pad=10, fontsize=14)
        ax_boundary.set_xlabel('feature 1')
        ax_boundary.set_ylabel('feature 2')
        ax_boundary.grid(alpha=0.14)
        sns.despine(ax=ax_boundary)

    anim = animation.FuncAnimation(fig, draw_frame, frames=frames, interval=90)
    anim.save(OUT_DIR / 'weights-bias-threshold-shift.gif', writer=animation.PillowWriter(fps=8))
    plt.close(fig)


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
    raw, labels = make_moons(n_samples=220, noise=0.17, random_state=7)
    raw = StandardScaler().fit_transform(raw)
    model = MLPClassifier(
        hidden_layer_sizes=(2, 2),
        activation='tanh',
        solver='lbfgs',
        alpha=1e-2,
        max_iter=6000,
        random_state=7,
    )
    model.fit(raw, labels)

    activations = []
    current = raw
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        current = np.tanh(current @ weights + bias)
        activations.append(current.copy())

    stages = [
        ('raw input', raw, 'The classes still wrap around each other.'),
        ('after hidden layer 1', activations[0], 'The network starts to untangle the moons.'),
        ('later representation', activations[1], 'A simple linear split is now much more plausible.'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), facecolor=COLORS['bg'])
    fig.subplots_adjust(top=0.77, left=0.06, right=0.98, bottom=0.12, wspace=0.23)
    title_block(fig, 'Hidden layers turn raw input into usable features', 'Instead of a cartoon, this uses a tiny learned network and shows how a simple linear probe sees the data at each stage.')

    for ax, (label, data, note) in zip(axes, stages):
        probe = LogisticRegression(max_iter=4000)
        probe.fit(data, labels)
        accuracy = probe.score(data, labels)
        DecisionBoundaryDisplay.from_estimator(
            probe,
            data,
            response_method='predict_proba',
            class_of_interest=1,
            plot_method='contourf',
            levels=np.linspace(0, 1, 11),
            cmap='RdBu_r',
            alpha=0.30,
            grid_resolution=220,
            eps=0.9,
            ax=ax,
        )
        DecisionBoundaryDisplay.from_estimator(
            probe,
            data,
            response_method='predict_proba',
            class_of_interest=1,
            plot_method='contour',
            levels=[0.5],
            colors=[COLORS['ink']],
            linewidths=2,
            grid_resolution=220,
            eps=0.9,
            ax=ax,
        )
        ax.scatter(data[labels == 0, 0], data[labels == 0, 1], color=COLORS['blue_dark'], s=40, alpha=0.90, edgecolor='white', linewidth=0.4, label='class 0')
        ax.scatter(data[labels == 1, 0], data[labels == 1, 1], color=COLORS['red_dark'], s=40, alpha=0.90, edgecolor='white', linewidth=0.4, label='class 1')
        ax.set_title(f'{label}\nlinear probe accuracy: {accuracy:.2f}', fontsize=15, pad=12, fontweight='bold')
        ax.text(
            0.03,
            0.96,
            note,
            transform=ax.transAxes,
            va='top',
            fontsize=10.8,
            bbox={'boxstyle': 'round,pad=0.30', 'fc': 'white', 'ec': '#cbd5e1', 'alpha': 0.95},
        )
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.grid(alpha=0.12)
        sns.despine(ax=ax)

    axes[0].legend(frameon=False, loc='lower left')
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
    image_weights_bias_animation()
    image_activation_functions()
    image_representation_building()
    image_backprop_blame_assignment()
    image_debugging_checklist()
    print('Generated images in', OUT_DIR)


if __name__ == '__main__':
    main()
