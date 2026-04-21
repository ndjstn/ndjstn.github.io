from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
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
    feature_names = ['cold room', 'someone home', 'afternoon sun']
    values = np.array([0.80, 1.00, 0.60])
    baseline_weights = np.array([1.10, 0.50, -0.70])
    baseline_feature_score = np.dot(values, baseline_weights)

    fig, (ax_weights, ax_bias) = plt.subplots(1, 2, figsize=(13.8, 5.2), facecolor='white')
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.78, wspace=0.22)
    fig.text(0.06, 0.93, 'Thermostat example: weight vs bias', fontsize=18.5, fontweight='bold', color=COLORS['ink'])
    fig.text(0.06, 0.88, 'Room stays fixed: cold 0.80, home 1.00, sun 0.60. Left changes one weight; right changes only bias.', fontsize=11.2, color=COLORS['muted'])

    frames = list(range(48))

    def draw_frame(frame_index):
        ax_weights.clear()
        ax_bias.clear()

        weight_phase = frame_index < 24
        current_weights = baseline_weights.copy()
        if weight_phase:
            current_weights[0] = np.interp(frame_index, [0, 23], [0.40, 1.10])
            current_bias = -0.70
            bias_demo = -0.70
            feature_score = np.dot(values, baseline_weights)
            phase_label = 'phase 1: change one weight'
        else:
            current_bias = -0.70
            bias_demo = np.interp(frame_index, [24, 47], [-1.20, -0.60])
            feature_score = baseline_feature_score
            phase_label = 'phase 2: change only bias'

        contributions = values * current_weights
        left_score = contributions.sum()
        left_threshold = -current_bias
        left_z = left_score + current_bias

        ax_weights.set_title('left knob: weight', loc='left', fontsize=13.5, pad=10, fontweight='bold')
        y = np.arange(len(feature_names))
        colors = [COLORS['blue_dark'], COLORS['green_dark'], COLORS['red_dark']]
        ax_weights.barh(y, contributions, color=colors, alpha=0.88)
        ax_weights.axvline(0, color='#cbd5e1', linewidth=1.2)
        ax_weights.set_yticks(y, labels=feature_names)
        ax_weights.set_xlim(-0.55, 1.15)
        ax_weights.invert_yaxis()
        ax_weights.grid(axis='x', alpha=0.12)
        sns.despine(ax=ax_weights, left=True, bottom=False)
        for idx, value in enumerate(contributions):
            ax_weights.text(value + (0.03 if value >= 0 else -0.03), idx, f'{value:+.2f}', va='center', ha='left' if value >= 0 else 'right', fontsize=10.2)
        left_summary = '\n'.join([
            'moving now' if weight_phase else 'fixed now',
            f'cold-room weight = {current_weights[0]:+.2f}',
            f'room score = {left_score:.2f}   cutoff = {left_threshold:.2f}',
        ])
        ax_weights.text(
            0.03,
            0.95,
            left_summary,
            transform=ax_weights.transAxes,
            va='top',
            fontsize=10.7,
            color=COLORS['ink'],
            linespacing=1.35,
            bbox={'boxstyle': 'round,pad=0.20', 'fc': 'white', 'ec': 'none', 'alpha': 0.88},
        )
        ax_weights.text(0.03, 0.67, 'heat on' if left_z > 0 else 'stay off', transform=ax_weights.transAxes, va='top', fontsize=11.2, fontweight='bold', color=COLORS['green_dark'] if left_z > 0 else COLORS['red_dark'])

        ax_bias.set_title('right knob: bias', loc='left', fontsize=13.5, pad=10, fontweight='bold')
        ax_bias.set_xlim(0.0, 1.35)
        ax_bias.set_ylim(0, 1)
        ax_bias.set_yticks([])
        ax_bias.set_xticks(np.arange(0.0, 1.4, 0.25))
        ax_bias.grid(axis='x', alpha=0.12)
        sns.despine(ax=ax_bias, left=True, bottom=False)
        ax_bias.hlines(0.45, 0.0, 1.35, color='#cbd5e1', linewidth=2)
        threshold = -bias_demo
        z = feature_score + bias_demo
        ax_bias.axvline(threshold, color=COLORS['ink'], linewidth=2)
        ax_bias.scatter([feature_score], [0.45], s=84, color=COLORS['blue_dark'], zorder=3)
        right_summary = '\n'.join([
            'fixed now' if weight_phase else 'moving now',
            f'same room score = {feature_score:.2f}',
            f'bias = {bias_demo:+.2f}   cutoff = {threshold:.2f}',
        ])
        ax_bias.text(
            0.03,
            0.95,
            right_summary,
            transform=ax_bias.transAxes,
            va='top',
            fontsize=10.7,
            color=COLORS['ink'],
            linespacing=1.35,
            bbox={'boxstyle': 'round,pad=0.20', 'fc': 'white', 'ec': 'none', 'alpha': 0.88},
        )
        ax_bias.text(0.03, 0.67, 'heat on' if z > 0 else 'stay off', transform=ax_bias.transAxes, va='top', fontsize=11.2, fontweight='bold', color=COLORS['green_dark'] if z > 0 else COLORS['red_dark'])
        ax_bias.text(threshold, 0.74, 'cutoff', ha='center', fontsize=10.2, color=COLORS['muted'])
        ax_bias.text(feature_score, 0.18, 'same room score', ha='center', fontsize=10.0, color=COLORS['blue_dark'])

    anim = animation.FuncAnimation(fig, draw_frame, frames=frames, interval=100)
    anim.save(OUT_DIR / 'weights-bias-threshold-shift.gif', writer=animation.PillowWriter(fps=8))
    plt.close(fig)

def image_activation_functions():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.subplots_adjust(top=0.83, left=0.08, right=0.97, bottom=0.14)
    x = np.linspace(-4.5, 4.5, 400)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    fig.text(0.08, 0.95, 'Activation functions break the straight-line limit', fontsize=20.5, fontweight='bold', color=COLORS['ink'])
    fig.text(0.08, 0.905, 'Each curve reshapes the weighted sum differently after the same scoring step.', fontsize=11.8, color=COLORS['muted'])

    ax.plot(x, relu, color=COLORS['amber_dark'], linewidth=3, label='ReLU')
    ax.plot(x, sigmoid, color=COLORS['blue_dark'], linewidth=3, label='Sigmoid')
    ax.plot(x, tanh, color=COLORS['green_dark'], linewidth=3, label='Tanh')
    ax.axhline(0, color=COLORS['line'], linewidth=1)
    ax.axvline(0, color=COLORS['line'], linewidth=1)
    ax.set_xlabel('input to the activation')
    ax.set_ylabel('output response')
    ax.legend(frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0)
    ax.grid(alpha=0.22)
    sns.despine(ax=ax)
    save(fig, 'activation-functions.png')


def image_representation_building():
    raw, labels = make_moons(n_samples=220, noise=0.12, random_state=1)
    raw = StandardScaler().fit_transform(raw)
    model = MLPClassifier(
        hidden_layer_sizes=(2, 2),
        activation='tanh',
        solver='lbfgs',
        alpha=1e-2,
        max_iter=6000,
        random_state=1,
    )
    model.fit(raw, labels)

    activations = []
    current = raw
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        current = np.tanh(current @ weights + bias)
        activations.append(current.copy())

    stages = [
        ('raw input', raw),
        ('after hidden layer 1', activations[0]),
        ('later representation', activations[1]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.9), facecolor='white')
    fig.text(0.05, 0.95, 'Hidden layers try to make the next readout simpler', fontsize=21, fontweight='bold', color=COLORS['ink'])
    fig.text(0.05, 0.905, 'Same data, same labels. The only question is: how many mistakes does one straight line make at each stage?', fontsize=11.7, color=COLORS['muted'])
    fig.subplots_adjust(top=0.82, left=0.05, right=0.985, bottom=0.14, wspace=0.24)

    for ax, (label, data) in zip(axes, stages):
        probe = LogisticRegression(max_iter=4000)
        probe.fit(data, labels)
        accuracy = probe.score(data, labels)
        predictions = probe.predict(data)
        mistakes = predictions != labels
        mistake_count = int(mistakes.sum())

        x_min, x_max = data[:, 0].min() - 0.45, data[:, 0].max() + 0.45
        y_min, y_max = data[:, 1].min() - 0.45, data[:, 1].max() + 0.45
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
        decision = probe.coef_[0, 0] * xx + probe.coef_[0, 1] * yy + probe.intercept_[0]
        ax.contourf(xx, yy, decision, levels=[-1e9, 0, 1e9], colors=['#e8f4fb', '#fdecec'], alpha=0.38)
        ax.contour(xx, yy, decision, levels=[0], colors=[COLORS['ink']], linewidths=1.8)
        ax.scatter(data[labels == 0, 0], data[labels == 0, 1], color=COLORS['blue_dark'], s=38, alpha=0.94, edgecolor='white', linewidth=0.4)
        ax.scatter(data[labels == 1, 0], data[labels == 1, 1], color=COLORS['red_dark'], s=38, alpha=0.94, edgecolor='white', linewidth=0.4)
        if mistake_count:
            ax.scatter(data[mistakes, 0], data[mistakes, 1], s=88, facecolors='none', edgecolors=COLORS['ink'], linewidth=1.5)
        ax.set_title(label, fontsize=14.2, pad=10, fontweight='bold')
        ax.text(0.04, 0.05, f'straight-line mistakes: {mistake_count}\naccuracy: {accuracy:.2f}', transform=ax.transAxes, fontsize=10.6, color=COLORS['ink'], va='bottom', bbox={'boxstyle': 'round,pad=0.30', 'fc': 'white', 'ec': '#dbe4ee', 'alpha': 0.96})
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


def image_neuron_scoring_rule():
    feature_names = ['smoke density', 'room heat', 'shower steam']
    inputs = np.array([0.90, 0.55, 0.30])
    weights = np.array([1.40, 0.80, -1.10])
    contributions = inputs * weights
    bias = -0.60
    z = contributions.sum() + bias
    activation = 1 / (1 + np.exp(-z))

    fig = plt.figure(figsize=(13.8, 5.9), facecolor='white')
    fig.text(0.05, 0.95, 'A neuron is just a scoring rule plus a switch', fontsize=21, fontweight='bold', color=COLORS['ink'])
    fig.text(0.05, 0.905, 'Smoke and heat push the alarm score up. Shower steam pushes it back down. The activation says how strongly the unit fires.', fontsize=11.7, color=COLORS['muted'])

    gs = fig.add_gridspec(1, 3, left=0.05, right=0.98, bottom=0.14, top=0.84, width_ratios=[1.05, 1.18, 0.92], wspace=0.22)
    ax_table = fig.add_subplot(gs[0, 0])
    ax_sum = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[0, 2])

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')
    ax_table.text(0.04, 0.92, 'current signals', fontsize=13.8, fontweight='bold', color=COLORS['ink'])
    ax_table.text(0.04, 0.82, 'signal', fontsize=12.0, fontweight='bold')
    ax_table.text(0.45, 0.82, 'reading', fontsize=11.2, fontweight='bold', ha='center')
    ax_table.text(0.68, 0.82, 'weight', fontsize=11.2, fontweight='bold', ha='center')
    ax_table.text(0.93, 0.82, 'pull', fontsize=12.0, fontweight='bold', ha='center')
    zero_x = 0.84
    ax_table.plot([zero_x, zero_x], [0.17, 0.76], color='#cbd5e1', linewidth=1)
    row_y = [0.67, 0.47, 0.27]
    scale = 0.11 / np.max(np.abs(contributions))
    row_colors = [COLORS['blue_dark'], COLORS['green_dark'], COLORS['red_dark']]
    for name, value, weight, contribution, y, color in zip(feature_names, inputs, weights, contributions, row_y, row_colors):
        ax_table.text(0.04, y, name, fontsize=12.3, va='center')
        ax_table.text(0.50, y, f'{value:.2f}', fontsize=12.3, va='center', ha='center')
        ax_table.text(0.69, y, f'{weight:+.2f}', fontsize=12.3, va='center', ha='center')
        width = contribution * scale
        left = zero_x if contribution >= 0 else zero_x + width
        ax_table.add_patch(Rectangle((left, y - 0.045), abs(width), 0.09, facecolor=color, alpha=0.90, edgecolor='none', transform=ax_table.transAxes))
        ax_table.text(0.98, y, f'{contribution:+.2f}', fontsize=11.8, va='center', ha='right')
        ax_table.plot([0.04, 0.98], [y - 0.10, y - 0.10], color='#eef2f7', linewidth=1)
    ax_table.text(0.04, 0.08, f'bias = {bias:+.2f}', fontsize=12.2, color=COLORS['muted'])

    ax_sum.set_xlim(-0.45, 1.45)
    ax_sum.set_ylim(0, 1)
    ax_sum.set_yticks([])
    ax_sum.set_xticks(np.arange(-0.25, 1.51, 0.25))
    ax_sum.grid(axis='x', alpha=0.12)
    sns.despine(ax=ax_sum, left=True, bottom=False)
    ax_sum.text(0.00, 0.93, 'alarm score before activation', transform=ax_sum.transAxes, fontsize=13.8, fontweight='bold')
    ax_sum.text(0.00, 0.84, f'{contributions[0]:.2f} + {contributions[1]:.2f} - {abs(contributions[2]):.2f} - {abs(bias):.2f} = {z:.2f}', transform=ax_sum.transAxes, fontsize=11.4, color=COLORS['muted'])
    ax_sum.hlines(0.46, -0.45, 1.45, color='#cbd5e1', linewidth=2)
    ax_sum.axvline(0, color=COLORS['ink'], linewidth=1.5)
    current = 0.0
    labels = ['smoke', 'heat', 'steam', 'bias']
    deltas = [*contributions, bias]
    colors = [COLORS['blue_dark'], COLORS['green_dark'], COLORS['red_dark'], COLORS['muted']]
    for delta, label, color in zip(deltas, labels, colors):
        nxt = current + delta
        ax_sum.plot([current, nxt], [0.46, 0.46], color=color, linewidth=14, solid_capstyle='round', alpha=0.90)
        ax_sum.scatter([nxt], [0.46], s=58, color=color, zorder=3)
        if label != 'bias':
            ax_sum.text((current + nxt) / 2, 0.70 if delta >= 0 else 0.22, f'{label}\n{delta:+.2f}', ha='center', va='center', fontsize=10.4)
        current = nxt
    ax_sum.text(z, 0.78, f'score z = {z:.2f}', ha='center', fontsize=12.6, fontweight='bold', color=COLORS['ink'])

    raw = np.linspace(-4, 4, 400)
    curve = 1 / (1 + np.exp(-raw))
    ax_act.plot(raw, curve, color=COLORS['purple'], linewidth=2.8)
    ax_act.scatter([z], [activation], s=72, color=COLORS['purple'], zorder=3)
    ax_act.axvline(z, color='#cbd5e1', linewidth=1.2, linestyle='--')
    ax_act.axhline(activation, color='#cbd5e1', linewidth=1.2, linestyle='--')
    ax_act.axhline(0.5, color='#e2e8f0', linewidth=1.0, linestyle=':')
    ax_act.set_xlim(-4, 4)
    ax_act.set_ylim(0, 1.02)
    ax_act.set_title('how strongly the unit fires', loc='left', fontsize=13.8, pad=8, fontweight='bold')
    ax_act.set_xlabel('pre-activation score z')
    ax_act.set_ylabel('activation a')
    ax_act.grid(alpha=0.14)
    sns.despine(ax=ax_act)
    ax_act.text(0.05, 0.90, f'alarm activity = {activation:.2f}', transform=ax_act.transAxes, fontsize=12.0, fontweight='bold', color=COLORS['ink'])

    save(fig, 'neuron-scoring-rule.png')

def image_weights_biases():
    feature_names = ['cold room', 'someone home', 'afternoon sun']
    values = np.array([0.80, 1.00, 0.60])
    weights = np.array([1.10, 0.50, -0.70])
    contributions = values * weights
    feature_score = contributions.sum()
    modes = [('comfort mode', -0.70, 'heat turns on'), ('eco mode', -1.20, 'heat stays off')]

    fig = plt.figure(figsize=(13.8, 6.2), facecolor='white')
    fig.text(0.05, 0.95, 'Same room, different thermostat mode', fontsize=22, fontweight='bold', color=COLORS['ink'])
    fig.text(0.05, 0.905, 'Weights build the room score. Bias decides how much score is enough to heat.', fontsize=12.0, color=COLORS['muted'])

    outer = fig.add_gridspec(2, 1, left=0.05, right=0.98, bottom=0.10, top=0.88, height_ratios=[1.02, 0.88], hspace=0.38)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.40, 0.60], wspace=0.18)
    bottom = outer[1].subgridspec(1, 2, wspace=0.18)
    ax_table = fig.add_subplot(top[0, 0])
    ax_score = fig.add_subplot(top[0, 1])
    ax_mode_a = fig.add_subplot(bottom[0, 0])
    ax_mode_b = fig.add_subplot(bottom[0, 1])

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')
    ax_table.text(0.02, 0.93, 'current signals', fontsize=14.0, fontweight='bold', color=COLORS['ink'])
    ax_table.text(0.02, 0.84, 'signal', fontsize=12.2, fontweight='bold')
    ax_table.text(0.54, 0.84, 'reading', fontsize=12.2, fontweight='bold', ha='right')
    ax_table.text(0.69, 0.84, 'weight', fontsize=12.2, fontweight='bold', ha='right')
    ax_table.text(0.90, 0.84, 'pull', fontsize=12.2, fontweight='bold', ha='center')

    zero_x = 0.85
    ax_table.plot([zero_x, zero_x], [0.18, 0.77], color='#cbd5e1', linewidth=1)
    row_y = [0.67, 0.47, 0.27]
    scale = 0.10 / np.max(np.abs(contributions))
    row_colors = [COLORS['blue_dark'], COLORS['green_dark'], COLORS['red_dark']]

    for name, value, weight, contribution, y, color in zip(feature_names, values, weights, contributions, row_y, row_colors):
        ax_table.text(0.02, y, name, fontsize=12.4, va='center')
        ax_table.text(0.54, y, f'{value:.2f}', fontsize=12.4, va='center', ha='right')
        ax_table.text(0.69, y, f'{weight:+.2f}', fontsize=12.4, va='center', ha='right')
        width = contribution * scale
        left = zero_x if contribution >= 0 else zero_x + width
        ax_table.add_patch(Rectangle((left, y - 0.045), abs(width), 0.09, facecolor=color, edgecolor='none', alpha=0.92, transform=ax_table.transAxes))
        ax_table.text(0.98, y, f'{contribution:+.2f}', fontsize=12.2, va='center', ha='right')
        ax_table.plot([0.02, 0.98], [y - 0.10, y - 0.10], color='#eef2f7', linewidth=1)

    ax_score.set_xlim(-0.08, 1.18)
    ax_score.set_ylim(0, 1)
    ax_score.set_yticks([])
    ax_score.set_xticks(np.arange(0.0, 1.21, 0.25))
    ax_score.grid(axis='x', alpha=0.10)
    sns.despine(ax=ax_score, left=True, bottom=False)
    ax_score.text(0.00, 0.95, 'room score before bias', transform=ax_score.transAxes, fontsize=14.2, fontweight='bold')
    ax_score.text(0.00, 0.86, f'{contributions[0]:.2f} + {contributions[1]:.2f} - {abs(contributions[2]):.2f} = {feature_score:.2f}', transform=ax_score.transAxes, fontsize=11.5, color=COLORS['muted'])
    ax_score.hlines(0.45, -0.08, 1.18, color='#cbd5e1', linewidth=2)
    ax_score.axvline(0, color=COLORS['ink'], linewidth=1.6)

    current = 0.0
    labels = ['cold room', 'someone home', 'sunlight']
    for delta, label, color in zip(contributions, labels, row_colors):
        nxt = current + delta
        ax_score.plot([current, nxt], [0.45, 0.45], color=color, linewidth=16, solid_capstyle='round', alpha=0.92)
        ax_score.scatter([nxt], [0.45], s=62, color=color, zorder=3)
        ax_score.text((current + nxt) / 2, 0.69 if delta >= 0 else 0.22, f'{label}\n{delta:+.2f}', ha='center', va='center', fontsize=10.5)
        current = nxt
    ax_score.text(feature_score, 0.82, f'room score = {feature_score:.2f}', ha='center', fontsize=12.6, fontweight='bold', color=COLORS['blue_dark'])

    for axis, (mode_name, bias, outcome) in zip((ax_mode_a, ax_mode_b), modes):
        threshold = -bias
        z = feature_score + bias
        heat_on = z > 0
        axis.set_xlim(0.0, 1.35)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xticks(np.arange(0.0, 1.36, 0.25))
        axis.grid(axis='x', alpha=0.10)
        sns.despine(ax=axis, left=True, bottom=False)
        axis.set_title(mode_name, fontsize=14.0, pad=8, fontweight='bold')
        axis.hlines(0.43, 0.0, 1.35, color='#cbd5e1', linewidth=2)
        axis.axvline(threshold, color=COLORS['ink'], linewidth=2)
        axis.scatter([feature_score], [0.43], s=82, color=COLORS['blue_dark'], zorder=3)
        axis.text(0.03, 0.87, f'bias = {bias:+.2f}', transform=axis.transAxes, fontsize=11.0, color=COLORS['muted'])
        axis.text(0.03, 0.73, outcome, transform=axis.transAxes, fontsize=12.2, fontweight='bold', color=COLORS['green_dark'] if heat_on else COLORS['red_dark'])
        axis.text(0.74, 0.87, f'z = {z:+.2f}', transform=axis.transAxes, fontsize=11.4, fontweight='bold', color=COLORS['ink'])
        axis.text(threshold, 0.74, f'cutoff {threshold:.2f}', ha='center', fontsize=10.3, color=COLORS['muted'])
        axis.text(feature_score, 0.18, f'same room score {feature_score:.2f}', ha='center', fontsize=10.3, color=COLORS['blue_dark'])

    save(fig, 'weights-biases.png')

def image_backprop_blame_assignment():
    fig = plt.figure(figsize=(13.8, 6.3), facecolor='white')
    fig.text(0.05, 0.95, 'Backprop is error credit assignment', fontsize=21, fontweight='bold', color=COLORS['ink'])
    fig.text(0.05, 0.905, 'The model predicts, compares with the target, then sends blame backward. Bigger gradients mean bigger corrections.', fontsize=11.7, color=COLORS['muted'])

    gs = fig.add_gridspec(1, 2, left=0.05, right=0.98, top=0.86, bottom=0.13, width_ratios=[1.15, 0.85], wspace=0.16)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_net.set_xlim(0, 1)
    ax_net.set_ylim(0, 1)
    ax_net.axis('off')

    ax_net.text(0.00, 0.96, '1. predict, compare, send error back', fontsize=14.0, fontweight='bold', color=COLORS['ink'])
    ax_net.plot([0.00, 0.08], [0.90, 0.90], color=COLORS['blue_dark'], linewidth=3)
    ax_net.text(0.09, 0.90, 'forward signal', va='center', fontsize=10.8, color=COLORS['muted'])
    ax_net.add_patch(FancyArrowPatch((0.27, 0.90), (0.35, 0.90), arrowstyle='->', linewidth=2.4, color=COLORS['purple'], mutation_scale=12))
    ax_net.text(0.36, 0.90, 'error signal', va='center', fontsize=10.8, color=COLORS['muted'])

    inputs = [(0.10, 0.72, r'$x_1$', 0.80), (0.10, 0.42, r'$x_2$', 0.40)]
    hidden = [(0.38, 0.80, r'$h_1$', 0.71), (0.38, 0.34, r'$h_2$', 0.29)]
    output = (0.67, 0.57, r'$\hat{y}$', 0.63)
    target = 1.00

    forward_edges = [
        (inputs[0][:2], hidden[0][:2], 2.6),
        (inputs[1][:2], hidden[0][:2], 1.4),
        (inputs[0][:2], hidden[1][:2], 1.8),
        (inputs[1][:2], hidden[1][:2], 2.9),
        (hidden[0][:2], output[:2], 3.2),
        (hidden[1][:2], output[:2], 1.9),
    ]
    for start_xy, end_xy, width in forward_edges:
        ax_net.add_patch(FancyArrowPatch(start_xy, end_xy, arrowstyle='-', linewidth=width, color=COLORS['blue_dark'], alpha=0.42))

    for x, y, label, value in inputs:
        ax_net.add_patch(Circle((x, y), 0.050, facecolor='#e0f2fe', edgecolor=COLORS['blue_dark'], linewidth=2.2))
        ax_net.text(x, y + 0.002, label, ha='center', va='center', fontsize=18, fontweight='bold')
        ax_net.text(x, y - 0.090, f'{value:.2f}', ha='center', va='center', fontsize=11.0, color=COLORS['muted'])
    for x, y, label, value in hidden:
        ax_net.add_patch(Circle((x, y), 0.056, facecolor='white', edgecolor='#94a3b8', linewidth=2.2))
        ax_net.text(x, y + 0.002, label, ha='center', va='center', fontsize=18, fontweight='bold')
        ax_net.text(x, y - 0.100, f'{value:.2f}', ha='center', va='center', fontsize=11.0, color=COLORS['muted'])
    ax_net.add_patch(Circle(output[:2], 0.058, facecolor='#fef3c7', edgecolor=COLORS['amber_dark'], linewidth=2.2))
    ax_net.text(output[0], output[1] + 0.002, output[2], ha='center', va='center', fontsize=18, fontweight='bold')
    ax_net.text(output[0], output[1] - 0.100, f'{output[3]:.2f}', ha='center', va='center', fontsize=11.0, color=COLORS['muted'])

    loss_box = FancyBboxPatch((0.79, 0.47), 0.17, 0.18, boxstyle='round,pad=0.02,rounding_size=0.03', facecolor='#fff7ed', edgecolor=COLORS['amber_dark'], linewidth=1.8)
    ax_net.add_patch(loss_box)
    ax_net.add_patch(FancyArrowPatch((0.73, 0.57), (0.79, 0.57), arrowstyle='->', linewidth=1.7, color=COLORS['line'], mutation_scale=12))
    ax_net.text(0.875, 0.61, f'target = {target:.2f}', ha='center', va='center', fontsize=11.4)
    ax_net.text(0.875, 0.565, f'prediction = {output[3]:.2f}', ha='center', va='center', fontsize=11.4)
    ax_net.text(0.875, 0.515, r'$L = (\hat{y} - y)^2$', ha='center', va='center', fontsize=11.2, color=COLORS['muted'])

    backward_paths = [
        ((0.82, 0.47), (0.71, 0.49), -0.10, 2.8),
        ((0.64, 0.49), (0.43, 0.73), -0.13, 3.0),
        ((0.64, 0.49), (0.43, 0.37), 0.05, 2.0),
        ((0.30, 0.70), (0.14, 0.67), 0.10, 1.9),
        ((0.30, 0.36), (0.14, 0.44), -0.10, 2.5),
    ]
    for start_xy, end_xy, rad, width in backward_paths:
        ax_net.add_patch(FancyArrowPatch(start_xy, end_xy, arrowstyle='->', linewidth=width, color=COLORS['purple'], alpha=0.82, mutation_scale=12, connectionstyle=f'arc3,rad={rad}'))
    ax_net.text(0.52, 0.74, 'prediction missed target → send blame back', fontsize=11.0, color=COLORS['purple'], fontweight='bold')

    ax_bar.set_title('2. larger gradients mean larger corrections', loc='left', fontsize=14.0, pad=10, fontweight='bold')
    bars = [
        ('x1→h1', -0.41),
        ('x2→h2', 0.62),
        ('h1→ŷ', -0.84),
        ('h2→ŷ', 0.28),
        ('b_h1', -0.19),
        ('b_ŷ', 0.11),
    ]
    labels = [label for label, _ in bars]
    values = np.array([value for _, value in bars])
    y = np.arange(len(bars))
    colors = np.where(values >= 0, COLORS['purple'], '#6d28d9')
    ax_bar.barh(y, values, color=colors, alpha=0.88, height=0.72)
    ax_bar.axvline(0, color='#cbd5e1', linewidth=1.3)
    ax_bar.set_yticks(y, labels=labels)
    ax_bar.set_xlabel('signed gradient')
    ax_bar.grid(axis='x', alpha=0.12)
    sns.despine(ax=ax_bar, left=True, bottom=False)
    ax_bar.invert_yaxis()
    max_abs = np.max(np.abs(values))
    ax_bar.set_xlim(-max_abs * 1.18, max_abs * 1.18)
    for idx, value in enumerate(values):
        ax_bar.text(value + (0.03 if value >= 0 else -0.03), idx, f'{value:+.2f}', va='center', ha='left' if value >= 0 else 'right', fontsize=10.4)
    ax_bar.text(0.02, 0.96, r'update rule: $w \leftarrow w - \eta \nabla_w L$', transform=ax_bar.transAxes, va='top', fontsize=11.0, color=COLORS['muted'])

    save(fig, 'backprop-blame-assignment.png')

def image_debugging_checklist():
    fig, ax = plt.subplots(figsize=(13.8, 6.0), facecolor='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.text(0.05, 0.95, 'A better debugging checklist than more buzzwords', fontsize=21, fontweight='bold', color=COLORS['ink'])
    fig.text(0.05, 0.905, 'When a model is failing, I try to diagnose the job, the signals, the capacity, the feedback, and the data in that order.', fontsize=11.7, color=COLORS['muted'])

    items = [
        ('01', 'task', 'What is the model actually trying to predict?', COLORS['blue_dark']),
        ('02', 'signals', 'Which inputs should matter most for this job?', COLORS['green_dark']),
        ('03', 'capacity', 'Is the model too weak or too large for the data?', COLORS['amber_dark']),
        ('04', 'feedback', 'Is the learning signal clean enough to help?', COLORS['red_dark']),
        ('05', 'data', 'Is the real problem the data, not the architecture?', COLORS['purple']),
    ]
    y_positions = [0.76, 0.61, 0.46, 0.31, 0.16]
    for index, (num, label, question, color), y in zip(range(len(items)), items, y_positions):
        if index > 0:
            ax.plot([0.05, 0.95], [y + 0.075, y + 0.075], color='#e2e8f0', linewidth=1)
        ax.add_patch(Rectangle((0.05, y - 0.045), 0.012, 0.09, facecolor=color, edgecolor='none'))
        ax.text(0.08, y + 0.022, num, ha='left', va='center', fontsize=10.5, color=COLORS['muted'])
        ax.text(0.14, y + 0.022, label, ha='left', va='center', fontsize=14.0, fontweight='bold', color=color)
        ax.text(0.14, y - 0.020, question, ha='left', va='center', fontsize=14.0, color=COLORS['ink'])

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
