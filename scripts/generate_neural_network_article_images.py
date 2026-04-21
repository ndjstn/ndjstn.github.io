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
    fig.text(0.06, 0.955, title, fontsize=22, fontweight='bold', color=COLORS['ink'])
    fig.text(0.06, 0.918, subtitle, fontsize=11.5, color=COLORS['muted'])

def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def image_explanation_problem():
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    left_x, left_w = 0.08, 0.22
    right_x, right_w = 0.46, 0.40
    rows = [
        ('neuron', 'score evidence for a pattern'),
        ('bias', 'shift the cutoff'),
        ('activation', 'decide what passes on'),
        ('backprop', 'assign blame and update'),
    ]
    y_positions = [0.73, 0.57, 0.41, 0.25]

    ax.text(left_x, 0.88, 'Technical term', ha='left', va='center', fontsize=17, fontweight='bold', color=COLORS['ink'])
    ax.text(right_x, 0.88, 'Plain-language job', ha='left', va='center', fontsize=17, fontweight='bold', color=COLORS['ink'])
    ax.plot([left_x, left_x + left_w], [0.845, 0.845], color='#fca5a5', linewidth=3, solid_capstyle='round')
    ax.plot([right_x, right_x + 0.28], [0.845, 0.845], color='#86efac', linewidth=3, solid_capstyle='round')

    for (term, job), y in zip(rows, y_positions):
        term_box = FancyBboxPatch(
            (left_x, y - 0.042),
            left_w,
            0.084,
            boxstyle='round,pad=0.012,rounding_size=0.03',
            facecolor='#fff7f7',
            edgecolor='#fca5a5',
            linewidth=1.5,
        )
        job_box = FancyBboxPatch(
            (right_x, y - 0.05),
            right_w,
            0.10,
            boxstyle='round,pad=0.012,rounding_size=0.03',
            facecolor='#f7fff9',
            edgecolor='#86efac',
            linewidth=1.5,
        )
        ax.add_patch(term_box)
        ax.add_patch(job_box)
        ax.text(left_x + left_w / 2, y, term, ha='center', va='center', fontsize=13.5)
        ax.text(right_x + right_w / 2, y, job, ha='center', va='center', fontsize=13.1)
        ax.add_patch(
            FancyArrowPatch(
                (left_x + left_w + 0.03, y),
                (right_x - 0.03, y),
                arrowstyle='->',
                mutation_scale=14,
                linewidth=1.6,
                color='#94a3b8',
            )
        )

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
    fig = plt.figure(figsize=(15.2, 8.0), facecolor=COLORS['bg'])
    outer = fig.add_gridspec(
        2, 2,
        left=0.05, right=0.98, bottom=0.10, top=0.84,
        width_ratios=[1.25, 0.82], height_ratios=[1.0, 0.78],
        wspace=0.24, hspace=0.26,
    )
    left_top = outer[0, 0].subgridspec(1, 3, wspace=0.10)
    ax_input = fig.add_subplot(left_top[0, 0])
    ax_weight = fig.add_subplot(left_top[0, 1], sharey=ax_input)
    ax_contrib = fig.add_subplot(left_top[0, 2], sharey=ax_input)
    ax_score = fig.add_subplot(outer[1, 0])
    ax_curve = fig.add_subplot(outer[:, 1])

    title_block(fig, 'Weights set the pull. Bias moves the threshold.', 'First read the score. Then watch the same score curve shift left or right when the bias changes.')

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

    specs = [
        (ax_input, x_vals, 'input $x_i$', (0, 1.05)),
        (ax_weight, weights, 'weight $w_i$', (-2.05, 2.05)),
        (ax_contrib, contributions, 'pull $w_i x_i$', (-1.75, 1.75)),
    ]
    for ax, values, label, xlim in specs:
        bar_colors = [COLORS['blue_dark'] if value >= 0 else COLORS['red_dark'] for value in values]
        ax.barh(y, values, color=bar_colors, alpha=0.88)
        ax.axvline(0, color='#cbd5e1', linewidth=1.1)
        ax.set_xlim(*xlim)
        ax.set_title(label, loc='left', fontsize=13.5, pad=8)
        ax.grid(axis='x', alpha=0.16)
        sns.despine(ax=ax, left=True, bottom=True)
        for idx, value in enumerate(values):
            if xlim[0] < 0:
                xpos = value + 0.07 * np.sign(value if value != 0 else 1)
                ha = 'left' if value >= 0 else 'right'
                txt = f'{value:+.2f}'
            else:
                xpos = value + 0.03
                ha = 'left'
                txt = f'{value:.2f}'
            ax.text(xpos, idx, txt, va='center', ha=ha, fontsize=10.3, color=COLORS['ink'])
    ax_input.set_yticks(y, labels=features)
    ax_input.tick_params(axis='y', labelsize=10.8)
    ax_weight.tick_params(axis='y', left=False, labelleft=False)
    ax_contrib.tick_params(axis='y', left=False, labelleft=False)

    step_labels = [*features[::-1], 'bias', 'total score']
    step_values = [*contributions[::-1], bias]
    current = 0.0
    for idx, delta in enumerate(step_values):
        start = min(current, current + delta)
        height = abs(delta)
        color = COLORS['blue_dark'] if delta >= 0 else COLORS['red_dark']
        ax_score.bar(idx, height, bottom=start, width=0.62, color=color, alpha=0.82)
        ax_score.text(idx, current + delta + (0.07 if delta >= 0 else -0.11), f'{delta:+.2f}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=9.8)
        current += delta
    ax_score.bar(len(step_values), total, width=0.62, color=COLORS['purple'], alpha=0.26, edgecolor=COLORS['purple'], linewidth=2)
    ax_score.text(len(step_values), total + 0.08, f'$z$ = {total:.2f}\n$p$ = {probability:.2f}', ha='center', va='bottom', fontsize=10.4, fontweight='bold')
    ax_score.axhline(0, color='#cbd5e1', linewidth=1.1)
    ax_score.set_xticks(np.arange(len(step_labels)), step_labels, rotation=15, ha='right')
    ax_score.set_ylabel('score before activation')
    ax_score.set_title('How the evidence adds up', loc='left', fontsize=13.5, pad=8)
    ax_score.text(0.01, 0.96, r'$z = \sum_i w_i x_i + b$', transform=ax_score.transAxes, va='top', fontsize=11.2, color=COLORS['muted'])
    ax_score.grid(axis='y', alpha=0.14)
    sns.despine(ax=ax_score)

    raw_score = np.linspace(-4.0, 4.0, 400)
    sample_raw = 0.8
    settings = [(-1.2, COLORS['red_dark'], 'bias = -1.2'), (0.0, COLORS['ink'], 'bias = 0.0'), (1.2, COLORS['green_dark'], 'bias = +1.2')]
    ax_curve.axhline(0.5, color='#cbd5e1', linewidth=1.0, linestyle=':')
    for bias_value, color, label in settings:
        response = 1 / (1 + np.exp(-(raw_score + bias_value)))
        threshold = -bias_value
        ax_curve.plot(raw_score, response, color=color, linewidth=2.8, label=label)
        ax_curve.axvline(threshold, color=color, linewidth=1.3, linestyle='--', alpha=0.8)
        sample_prob = 1 / (1 + np.exp(-(sample_raw + bias_value)))
        ax_curve.scatter([sample_raw], [sample_prob], color=color, s=42, zorder=5)
    ax_curve.text(-3.7, 0.94, 'same weights\ndifferent bias', fontsize=10.8, color=COLORS['muted'], va='top')
    ax_curve.text(1.6, 0.18, 'switch-on point\nslides left or right', fontsize=10.8, color=COLORS['muted'])
    ax_curve.set_title('Bias does not change what matters. It changes when the unit fires.', loc='left', fontsize=13.5, pad=8)
    ax_curve.set_xlabel(r'raw weighted sum $w^\top x$')
    ax_curve.set_ylabel(r'activation $\sigma(w^\top x + b)$')
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.legend(frameon=False, loc='lower right')
    ax_curve.grid(alpha=0.16)
    sns.despine(ax=ax_curve)

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
        ('raw input', raw, 'wrapped around each other'),
        ('after hidden layer 1', activations[0], 'starting to untangle'),
        ('later representation', activations[1], 'almost linearly separable'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), facecolor=COLORS['bg'])
    fig.subplots_adjust(top=0.78, left=0.055, right=0.985, bottom=0.12, wspace=0.23)
    title_block(fig, 'Hidden layers build easier-to-separate representations', 'The useful question is simple: how well can a straight line read the representation at each stage?')

    for ax, (label, data, note) in zip(axes, stages):
        probe = LogisticRegression(max_iter=4000)
        probe.fit(data, labels)
        accuracy = probe.score(data, labels)

        x_min, x_max = data[:, 0].min() - 0.45, data[:, 0].max() + 0.45
        y_min, y_max = data[:, 1].min() - 0.45, data[:, 1].max() + 0.45
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
        decision = probe.coef_[0, 0] * xx + probe.coef_[0, 1] * yy + probe.intercept_[0]
        ax.contourf(xx, yy, decision, levels=[-1e9, 0, 1e9], colors=['#e8f4fb', '#fdecec'], alpha=0.45)
        ax.contour(xx, yy, decision, levels=[0], colors=[COLORS['ink']], linewidths=1.8)
        ax.scatter(data[labels == 0, 0], data[labels == 0, 1], color=COLORS['blue_dark'], s=34, alpha=0.92, edgecolor='white', linewidth=0.35)
        ax.scatter(data[labels == 1, 0], data[labels == 1, 1], color=COLORS['red_dark'], s=34, alpha=0.92, edgecolor='white', linewidth=0.35)
        ax.set_title(label, fontsize=14.2, pad=10, fontweight='bold')
        ax.text(0.03, 0.05, f'{note}\nlinear readout acc = {accuracy:.2f}', transform=ax.transAxes, fontsize=10.3, color=COLORS['ink'], va='bottom', bbox={'boxstyle': 'round,pad=0.28', 'fc': 'white', 'ec': '#dbe4ee', 'alpha': 0.96})
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.08)
        sns.despine(ax=ax, left=True, bottom=True)

    save(fig, 'representation-building.png')

def image_backprop_blame_assignment():
    fig = plt.figure(figsize=(14.8, 6.8), facecolor=COLORS['bg'])
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.98, bottom=0.12, top=0.84, width_ratios=[1.1, 0.9], wspace=0.22)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_grad = fig.add_subplot(gs[0, 1])
    title_block(fig, 'Backpropagation is structured correction', 'Forward signals move right. Error signals move left. The biggest gradients mark the biggest corrections.')

    ax_net.set_xlim(0, 1)
    ax_net.set_ylim(0, 1)
    ax_net.axis('off')

    layers = {
        'input': [(0.12, 0.72), (0.12, 0.50), (0.12, 0.28)],
        'hidden': [(0.45, 0.63), (0.45, 0.37)],
        'output': [(0.78, 0.50)],
    }
    edge_pairs = [
        (layers['input'][0], layers['hidden'][0]),
        (layers['input'][1], layers['hidden'][0]),
        (layers['input'][2], layers['hidden'][0]),
        (layers['input'][0], layers['hidden'][1]),
        (layers['input'][1], layers['hidden'][1]),
        (layers['input'][2], layers['hidden'][1]),
    ]
    forward_strengths = [0.9, 0.35, 0.65, 0.55, 0.25, 0.75]
    backward_strengths = [0.18, 0.62, 0.42, 0.55, 0.85, 0.30]
    out_pairs = [(layers['hidden'][0], layers['output'][0]), (layers['hidden'][1], layers['output'][0])]
    out_forward = [0.85, 0.52]
    out_backward = [0.77, 0.33]

    for (start, end), strength, grad in zip(edge_pairs, forward_strengths, backward_strengths):
        ax_net.add_patch(FancyArrowPatch(start, end, arrowstyle='-', linewidth=1.1 + 2.5 * strength, color=COLORS['blue_dark'], alpha=0.34))
        ax_net.add_patch(FancyArrowPatch((end[0] - 0.02, end[1]), (start[0] + 0.02, start[1]), arrowstyle='->', linewidth=0.8 + 2.1 * grad, color=COLORS['red_dark'], alpha=0.72, mutation_scale=10))
    for (start, end), strength, grad in zip(out_pairs, out_forward, out_backward):
        ax_net.add_patch(FancyArrowPatch(start, end, arrowstyle='-', linewidth=1.1 + 2.7 * strength, color=COLORS['blue_dark'], alpha=0.34))
        ax_net.add_patch(FancyArrowPatch((end[0] - 0.02, end[1]), (start[0] + 0.02, start[1]), arrowstyle='->', linewidth=0.8 + 2.1 * grad, color=COLORS['red_dark'], alpha=0.72, mutation_scale=10))

    for x, y in layers['input']:
        ax_net.add_patch(Circle((x, y), 0.038, facecolor='#e8f4fb', edgecolor=COLORS['blue_dark'], linewidth=1.8))
    for x, y in layers['hidden']:
        ax_net.add_patch(Circle((x, y), 0.043, facecolor='#f8fafc', edgecolor='#94a3b8', linewidth=1.8))
    ax_net.add_patch(Circle(layers['output'][0], 0.048, facecolor='#fef3c7', edgecolor=COLORS['amber_dark'], linewidth=1.9))
    ax_net.add_patch(FancyBboxPatch((0.85, 0.43), 0.10, 0.14, boxstyle='round,pad=0.02,rounding_size=0.03', facecolor='#fff1f2', edgecolor='#fca5a5', linewidth=1.6))
    ax_net.text(0.90, 0.525, 'loss', ha='center', va='center', fontsize=12.5, fontweight='bold')
    ax_net.text(0.90, 0.475, '$L = 0.31$', ha='center', va='center', fontsize=11.0, color=COLORS['red_dark'])
    ax_net.add_patch(FancyArrowPatch((0.83, 0.50), (0.85, 0.50), arrowstyle='->', linewidth=1.6, color=COLORS['line'], mutation_scale=10))
    ax_net.text(0.08, 0.89, 'blue = forward activity', color=COLORS['blue_dark'], fontsize=11.2, fontweight='bold')
    ax_net.text(0.08, 0.83, 'red = backward gradient', color=COLORS['red_dark'], fontsize=11.2, fontweight='bold')
    ax_net.text(0.08, 0.10, 'thicker red arrows mean a larger correction', color=COLORS['muted'], fontsize=10.4)

    params = ['w_h1→y', 'w_x2→h2', 'w_x1→h1', 'w_h2→y', 'b_h1', 'b_y']
    grads = np.array([-0.84, 0.62, -0.41, 0.28, -0.19, 0.11])
    grad_colors = [COLORS['red_dark'] if g < 0 else COLORS['blue_dark'] for g in grads]
    y = np.arange(len(params))
    ax_grad.barh(y, grads, color=grad_colors, alpha=0.84)
    ax_grad.axvline(0, color='#cbd5e1', linewidth=1.1)
    ax_grad.set_yticks(y, labels=params)
    ax_grad.set_xlabel('signed gradient')
    ax_grad.set_title('Which parameters get the largest update?', loc='left', fontsize=13.5, pad=8)
    ax_grad.text(0.02, 0.96, r'$w \leftarrow w - \eta \nabla_w L$', transform=ax_grad.transAxes, va='top', fontsize=11.4, color=COLORS['muted'])
    for idx, grad in enumerate(grads):
        ax_grad.text(grad + (0.03 if grad >= 0 else -0.03), idx, f'{grad:+.2f}', va='center', ha='left' if grad >= 0 else 'right', fontsize=10.1)
    ax_grad.grid(axis='x', alpha=0.16)
    sns.despine(ax=ax_grad, left=True, bottom=True)

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
