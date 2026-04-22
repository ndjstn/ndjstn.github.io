from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("assets/img/posts/neural-network-components")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "bg": "#f8fafc",
    "panel": "#ffffff",
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#94a3b8",
    "blue": "#38bdf8",
    "blue_dark": "#0ea5e9",
    "green": "#22c55e",
    "green_dark": "#16a34a",
    "amber": "#f59e0b",
    "amber_dark": "#d97706",
    "red": "#ef4444",
    "red_dark": "#dc2626",
    "purple": "#a855f7",
}

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelcolor": COLORS["muted"],
        "text.color": COLORS["ink"],
        "axes.edgecolor": "#e2e8f0",
        "axes.facecolor": COLORS["panel"],
        "figure.facecolor": COLORS["bg"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
    }
)


def add_rounded_panel(ax, xy=(0.02, 0.05), width=0.96, height=0.9):
    panel = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=COLORS["panel"],
        edgecolor="#dbe4ee",
        linewidth=1.5,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(panel)


def title_block(fig, title, subtitle):
    fig.text(0.06, 0.955, title, fontsize=22, fontweight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.918, subtitle, fontsize=11.5, color=COLORS["muted"])


def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def image_hero_editorial():
    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.08, 0.83, r"$x$", fontsize=26, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.36,
        0.83,
        r"$w^\top x + b$",
        fontsize=26,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.68, 0.83, r"$\phi(z)$", fontsize=26, fontweight="bold", color=COLORS["ink"]
    )
    ax.text(
        0.86, 0.83, r"$\hat{y}$", fontsize=26, fontweight="bold", color=COLORS["ink"]
    )

    features = [
        ("x1", 0.82, COLORS["blue_dark"]),
        ("x2", 0.52, COLORS["green_dark"]),
        ("x3", -0.36, COLORS["red_dark"]),
    ]
    y_positions = [0.66, 0.52, 0.38]
    zero_x = 0.20
    ax.plot([zero_x, zero_x], [0.30, 0.72], color="#cbd5e1", linewidth=2)
    for (label, value, color), y in zip(features, y_positions, strict=False):
        ax.text(
            0.08, y, label, ha="center", va="center", fontsize=18, color=COLORS["muted"]
        )
        width = 0.16 * abs(value)
        left = zero_x if value >= 0 else zero_x - width
        ax.add_patch(
            Rectangle(
                (left, y - 0.032),
                width,
                0.064,
                facecolor=color,
                edgecolor="none",
                alpha=0.88,
            )
        )
        value_x = left + width + 0.018 if value >= 0 else zero_x + 0.018
        ax.text(
            value_x,
            y,
            f"{value:+.2f}",
            ha="left",
            va="center",
            fontsize=15,
            color=COLORS["ink"],
        )

    ax.add_patch(
        FancyArrowPatch(
            (0.32, 0.52),
            (0.40, 0.52),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2,
            color=COLORS["line"],
        )
    )

    score_start = 0.44
    score_y = 0.52
    current = score_start
    scale = 0.16
    deltas = [0.82, 0.52, -0.36, -0.42]
    colors = [
        COLORS["blue_dark"],
        COLORS["green_dark"],
        COLORS["red_dark"],
        COLORS["muted"],
    ]
    labels = [r"$w_1x_1$", r"$w_2x_2$", r"$w_3x_3$", r"$b$"]
    ax.plot(
        [score_start - 0.02, score_start + 0.24],
        [score_y, score_y],
        color="#cbd5e1",
        linewidth=2,
    )
    for delta, color, label in zip(deltas, colors, labels, strict=False):
        nxt = current + delta * scale
        ax.plot(
            [current, nxt],
            [score_y, score_y],
            color=color,
            linewidth=16,
            solid_capstyle="round",
            alpha=0.9,
        )
        ax.scatter([nxt], [score_y], s=80, color=color, zorder=3)
        ax.text(
            (current + nxt) / 2,
            score_y + (0.095 if delta >= 0 else -0.11),
            label,
            ha="center",
            fontsize=15,
            color=COLORS["muted"],
        )
        current = nxt
    ax.text(
        current, score_y - 0.18, r"$z$", ha="center", fontsize=22, color=COLORS["ink"]
    )

    ax.add_patch(
        FancyArrowPatch(
            (0.61, 0.52),
            (0.67, 0.52),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2,
            color=COLORS["line"],
        )
    )

    curve_x = np.linspace(-3.5, 3.5, 200)
    curve_y = 1 / (1 + np.exp(-curve_x))
    plot_x = 0.68 + (curve_x + 3.5) / 7.0 * 0.16
    plot_y = 0.35 + curve_y * 0.34
    ax.plot(plot_x, plot_y, color=COLORS["purple"], linewidth=4)
    sample = 0.72
    sample_y = 1 / (1 + np.exp(-sample))
    sx = 0.68 + (sample + 3.5) / 7.0 * 0.16
    sy = 0.35 + sample_y * 0.34
    ax.plot([sx, sx], [0.35, sy], color="#cbd5e1", linewidth=1.6, linestyle="--")
    ax.scatter([sx], [sy], s=120, color=COLORS["purple"], zorder=4)

    ax.add_patch(
        FancyArrowPatch(
            (0.85, sy),
            (0.90, sy),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2,
            color=COLORS["line"],
        )
    )
    output = Circle(
        (0.93, sy),
        0.045,
        facecolor="#fef3c7",
        edgecolor=COLORS["amber_dark"],
        linewidth=2.4,
    )
    ax.add_patch(output)

    layer_xs = [0.11, 0.45, 0.74, 0.93]
    for x in layer_xs:
        ax.plot([x, x], [0.18, 0.24], color="#cbd5e1", linewidth=1.4)
    ax.plot([0.11, 0.93], [0.21, 0.21], color="#e2e8f0", linewidth=1.2)
    ax.text(0.11, 0.14, "signals", ha="center", fontsize=15, color=COLORS["muted"])
    ax.text(0.45, 0.14, "score", ha="center", fontsize=15, color=COLORS["muted"])
    ax.text(0.74, 0.14, "activation", ha="center", fontsize=15, color=COLORS["muted"])
    ax.text(0.93, 0.14, "prediction", ha="center", fontsize=15, color=COLORS["muted"])

    save(fig, "hero.png")


def image_explanation_problem():
    fig, ax = plt.subplots(figsize=(13.8, 6.0), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.05,
        0.95,
        "The label is not the idea",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "These terms only became useful once I attached each one to the question it answers inside the system.",
        fontsize=11.7,
        color=COLORS["muted"],
    )

    ax.text(
        0.05,
        0.83,
        "technical term",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.30,
        0.83,
        "plain-language job",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.72,
        0.83,
        "question it answers",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )

    rows = [
        (
            "neuron",
            "score local evidence",
            "Does this pattern show up here?",
            COLORS["blue_dark"],
        ),
        (
            "bias",
            "move the trigger point",
            "How much evidence is enough?",
            COLORS["green_dark"],
        ),
        (
            "activation",
            "reshape the response",
            "How hard should it fire?",
            COLORS["amber_dark"],
        ),
        (
            "backprop",
            "route correction backward",
            "Who should change most?",
            COLORS["purple"],
        ),
    ]
    y_positions = [0.68, 0.52, 0.36, 0.20]

    for (term, job, question, color), y in zip(rows, y_positions, strict=False):
        ax.plot([0.05, 0.95], [y - 0.09, y - 0.09], color="#e2e8f0", linewidth=1)
        ax.add_patch(
            Rectangle((0.05, y - 0.045), 0.012, 0.09, facecolor=color, edgecolor="none")
        )
        ax.text(
            0.08,
            y,
            term,
            fontsize=14.5,
            fontweight="bold",
            va="center",
            color=COLORS["ink"],
        )
        ax.text(0.30, y, job, fontsize=14.5, va="center", color=COLORS["ink"])
        ax.text(0.72, y, question, fontsize=13.8, va="center", color=COLORS["muted"])

    save(fig, "explanation-problem.png")


def image_weights_bias_animation():
    feature_names = ["cold room", "someone home", "afternoon sun"]
    values = np.array([0.80, 1.00, 0.60])
    baseline_weights = np.array([1.10, 0.50, -0.70])
    baseline_feature_score = np.dot(values, baseline_weights)

    fig, (ax_weights, ax_bias) = plt.subplots(
        1, 2, figsize=(13.8, 5.2), facecolor="white"
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.78, wspace=0.22)
    fig.text(
        0.06,
        0.93,
        "Thermostat example: weight vs bias",
        fontsize=18.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.06,
        0.88,
        "Room stays fixed: cold 0.80, home 1.00, sun 0.60. Left changes one weight; right changes only bias.",
        fontsize=11.2,
        color=COLORS["muted"],
    )

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
        else:
            current_bias = -0.70
            bias_demo = np.interp(frame_index, [24, 47], [-1.20, -0.60])
            feature_score = baseline_feature_score

        contributions = values * current_weights
        left_score = contributions.sum()
        left_threshold = -current_bias
        left_z = left_score + current_bias

        ax_weights.set_title(
            "left knob: weight", loc="left", fontsize=13.5, pad=10, fontweight="bold"
        )
        y = np.arange(len(feature_names))
        colors = [COLORS["blue_dark"], COLORS["green_dark"], COLORS["red_dark"]]
        ax_weights.barh(y, contributions, color=colors, alpha=0.88)
        ax_weights.axvline(0, color="#cbd5e1", linewidth=1.2)
        ax_weights.set_yticks(y, labels=feature_names)
        ax_weights.set_xlim(-0.55, 1.15)
        ax_weights.invert_yaxis()
        ax_weights.grid(axis="x", alpha=0.12)
        sns.despine(ax=ax_weights, left=True, bottom=False)
        for idx, value in enumerate(contributions):
            ax_weights.text(
                value + (0.03 if value >= 0 else -0.03),
                idx,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=10.2,
            )
        left_summary = "\n".join(
            [
                "moving now" if weight_phase else "fixed now",
                f"cold-room weight = {current_weights[0]:+.2f}",
                f"room score = {left_score:.2f}   cutoff = {left_threshold:.2f}",
            ]
        )
        ax_weights.text(
            0.03,
            0.95,
            left_summary,
            transform=ax_weights.transAxes,
            va="top",
            fontsize=10.7,
            color=COLORS["ink"],
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "none",
                "alpha": 0.88,
            },
        )
        ax_weights.text(
            0.03,
            0.67,
            "heat on" if left_z > 0 else "stay off",
            transform=ax_weights.transAxes,
            va="top",
            fontsize=11.2,
            fontweight="bold",
            color=COLORS["green_dark"] if left_z > 0 else COLORS["red_dark"],
        )

        ax_bias.set_title(
            "right knob: bias", loc="left", fontsize=13.5, pad=10, fontweight="bold"
        )
        ax_bias.set_xlim(0.0, 1.35)
        ax_bias.set_ylim(0, 1)
        ax_bias.set_yticks([])
        ax_bias.set_xticks(np.arange(0.0, 1.4, 0.25))
        ax_bias.grid(axis="x", alpha=0.12)
        sns.despine(ax=ax_bias, left=True, bottom=False)
        ax_bias.hlines(0.45, 0.0, 1.35, color="#cbd5e1", linewidth=2)
        threshold = -bias_demo
        z = feature_score + bias_demo
        ax_bias.axvline(threshold, color=COLORS["ink"], linewidth=2)
        ax_bias.scatter(
            [feature_score], [0.45], s=84, color=COLORS["blue_dark"], zorder=3
        )
        right_summary = "\n".join(
            [
                "fixed now" if weight_phase else "moving now",
                f"same room score = {feature_score:.2f}",
                f"bias = {bias_demo:+.2f}   cutoff = {threshold:.2f}",
            ]
        )
        ax_bias.text(
            0.03,
            0.95,
            right_summary,
            transform=ax_bias.transAxes,
            va="top",
            fontsize=10.7,
            color=COLORS["ink"],
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "none",
                "alpha": 0.88,
            },
        )
        ax_bias.text(
            0.03,
            0.67,
            "heat on" if z > 0 else "stay off",
            transform=ax_bias.transAxes,
            va="top",
            fontsize=11.2,
            fontweight="bold",
            color=COLORS["green_dark"] if z > 0 else COLORS["red_dark"],
        )
        ax_bias.text(
            threshold, 0.74, "cutoff", ha="center", fontsize=10.2, color=COLORS["muted"]
        )
        ax_bias.text(
            feature_score,
            0.18,
            "same room score",
            ha="center",
            fontsize=10.0,
            color=COLORS["blue_dark"],
        )

    anim = animation.FuncAnimation(fig, draw_frame, frames=frames, interval=100)
    anim.save(
        OUT_DIR / "weights-bias-threshold-shift.gif",
        writer=animation.PillowWriter(fps=8),
    )
    plt.close(fig)


def image_activation_functions():
    fig, ax = plt.subplots(figsize=(13.2, 5.8), facecolor="white")
    fig.subplots_adjust(top=0.86, left=0.08, right=0.97, bottom=0.15)
    fig.text(
        0.08,
        0.94,
        "Activation functions reshape the same score",
        fontsize=19.5,
        fontweight="bold",
        color=COLORS["ink"],
    )

    z = np.linspace(-4.5, 4.5, 400)
    specs = [
        ("ReLU", np.maximum(0, z), COLORS["amber_dark"]),
        ("Sigmoid", 1 / (1 + np.exp(-z)), COLORS["blue_dark"]),
        ("Tanh", np.tanh(z), COLORS["green_dark"]),
    ]
    for label, values, color in specs:
        ax.plot(z, values, color=color, linewidth=2.7, label=label)

    sample_z = 1.2
    ax.axvline(sample_z, color="#cbd5e1", linewidth=1.3, linestyle="--")
    ax.text(
        sample_z + 0.08,
        4.18,
        "same incoming score",
        fontsize=10.6,
        color=COLORS["muted"],
    )
    ax.axhline(0, color=COLORS["line"], linewidth=1.0)
    ax.axvline(0, color=COLORS["line"], linewidth=1.0)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-1.15, 4.75)
    ax.set_xlabel("incoming score z")
    ax.set_ylabel("output response")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
    )
    ax.grid(alpha=0.16)
    sns.despine(ax=ax)
    save(fig, "activation-functions.png")


def image_representation_building():
    raw, labels = make_moons(n_samples=220, noise=0.12, random_state=1)
    raw = StandardScaler().fit_transform(raw)
    model = MLPClassifier(
        hidden_layer_sizes=(2, 2),
        activation="tanh",
        solver="lbfgs",
        alpha=1e-2,
        max_iter=6000,
        random_state=1,
    )
    model.fit(raw, labels)

    activations = []
    current = raw
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1], strict=False):
        current = np.tanh(current @ weights + bias)
        activations.append(current.copy())

    stages = [
        ("raw input", raw, "line still cuts through both classes"),
        ("after hidden layer 1", activations[0], "much easier for one line to read"),
        ("later representation", activations[1], "one line now separates them cleanly"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), facecolor="white")
    fig.text(
        0.05,
        0.95,
        "Hidden layers try to make the next readout simpler",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "Same data, same labels. Watch how many mistakes one straight line makes after each transformation.",
        fontsize=11.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(top=0.82, left=0.05, right=0.985, bottom=0.14, wspace=0.24)

    for ax, (label, data, note) in zip(axes, stages, strict=False):
        probe = LogisticRegression(max_iter=4000)
        probe.fit(data, labels)
        accuracy = probe.score(data, labels)
        predictions = probe.predict(data)
        mistakes = predictions != labels
        mistake_count = int(mistakes.sum())

        x_min, x_max = data[:, 0].min() - 0.45, data[:, 0].max() + 0.45
        y_min, y_max = data[:, 1].min() - 0.45, data[:, 1].max() + 0.45
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220)
        )
        decision = probe.coef_[0, 0] * xx + probe.coef_[0, 1] * yy + probe.intercept_[0]
        ax.contour(xx, yy, decision, levels=[0], colors=[COLORS["ink"]], linewidths=2.0)
        ax.scatter(
            data[labels == 0, 0],
            data[labels == 0, 1],
            color=COLORS["blue_dark"],
            s=38,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.scatter(
            data[labels == 1, 0],
            data[labels == 1, 1],
            color=COLORS["red_dark"],
            s=38,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.4,
        )
        if mistake_count:
            ax.scatter(
                data[mistakes, 0],
                data[mistakes, 1],
                s=92,
                facecolors="none",
                edgecolors=COLORS["ink"],
                linewidth=1.5,
            )
        ax.set_title(label, fontsize=14.2, pad=10, fontweight="bold")
        ax.text(
            0.04,
            0.92,
            note,
            transform=ax.transAxes,
            fontsize=10.4,
            color=COLORS["muted"],
        )
        ax.text(
            0.04,
            0.05,
            f"{mistake_count} straight-line mistakes\naccuracy: {accuracy:.2f}",
            transform=ax.transAxes,
            fontsize=10.6,
            color=COLORS["ink"],
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#dbe4ee",
                "alpha": 0.96,
            },
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        sns.despine(ax=ax, left=True, bottom=True)

    save(fig, "representation-building.png")


def image_neuron_scoring_rule():
    feature_names = ["smoke density", "room heat", "shower steam"]
    inputs = np.array([0.90, 0.55, 0.30])
    weights = np.array([1.40, 0.80, -1.10])
    contributions = inputs * weights
    bias = -0.60
    z = contributions.sum() + bias
    activation = 1 / (1 + np.exp(-z))

    fig = plt.figure(figsize=(13.8, 5.9), facecolor="white")
    fig.text(
        0.05,
        0.95,
        "A neuron is just a scoring rule plus a switch",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "Smoke and heat push the alarm score up. Shower steam pushes it back down. The activation says how strongly the unit fires.",
        fontsize=11.7,
        color=COLORS["muted"],
    )

    gs = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.98,
        bottom=0.14,
        top=0.84,
        width_ratios=[1.05, 1.18, 0.92],
        wspace=0.22,
    )
    ax_table = fig.add_subplot(gs[0, 0])
    ax_sum = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[0, 2])

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")
    ax_table.text(
        0.04,
        0.92,
        "current signals",
        fontsize=13.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax_table.text(0.04, 0.82, "signal", fontsize=12.0, fontweight="bold")
    ax_table.text(0.45, 0.82, "reading", fontsize=11.2, fontweight="bold", ha="center")
    ax_table.text(0.68, 0.82, "weight", fontsize=11.2, fontweight="bold", ha="center")
    ax_table.text(0.93, 0.82, "pull", fontsize=12.0, fontweight="bold", ha="center")
    zero_x = 0.84
    ax_table.plot([zero_x, zero_x], [0.17, 0.76], color="#cbd5e1", linewidth=1)
    row_y = [0.67, 0.47, 0.27]
    scale = 0.11 / np.max(np.abs(contributions))
    row_colors = [COLORS["blue_dark"], COLORS["green_dark"], COLORS["red_dark"]]
    for name, value, weight, contribution, y, color in zip(
        feature_names, inputs, weights, contributions, row_y, row_colors, strict=False
    ):
        ax_table.text(0.04, y, name, fontsize=12.3, va="center")
        ax_table.text(0.50, y, f"{value:.2f}", fontsize=12.3, va="center", ha="center")
        ax_table.text(
            0.69, y, f"{weight:+.2f}", fontsize=12.3, va="center", ha="center"
        )
        width = contribution * scale
        left = zero_x if contribution >= 0 else zero_x + width
        ax_table.add_patch(
            Rectangle(
                (left, y - 0.045),
                abs(width),
                0.09,
                facecolor=color,
                alpha=0.90,
                edgecolor="none",
                transform=ax_table.transAxes,
            )
        )
        ax_table.text(
            0.98, y, f"{contribution:+.2f}", fontsize=11.8, va="center", ha="right"
        )
        ax_table.plot([0.04, 0.98], [y - 0.10, y - 0.10], color="#eef2f7", linewidth=1)
    ax_table.text(
        0.04, 0.08, f"bias = {bias:+.2f}", fontsize=12.2, color=COLORS["muted"]
    )

    ax_sum.set_xlim(-0.45, 1.45)
    ax_sum.set_ylim(0, 1)
    ax_sum.set_yticks([])
    ax_sum.set_xticks(np.arange(-0.25, 1.51, 0.25))
    ax_sum.grid(axis="x", alpha=0.12)
    sns.despine(ax=ax_sum, left=True, bottom=False)
    ax_sum.text(
        0.00,
        0.93,
        "alarm score before activation",
        transform=ax_sum.transAxes,
        fontsize=13.8,
        fontweight="bold",
    )
    ax_sum.text(
        0.00,
        0.84,
        f"{contributions[0]:.2f} + {contributions[1]:.2f} - {abs(contributions[2]):.2f} - {abs(bias):.2f} = {z:.2f}",
        transform=ax_sum.transAxes,
        fontsize=11.4,
        color=COLORS["muted"],
    )
    ax_sum.hlines(0.46, -0.45, 1.45, color="#cbd5e1", linewidth=2)
    ax_sum.axvline(0, color=COLORS["ink"], linewidth=1.5)
    current = 0.0
    labels = ["smoke", "heat", "steam", "bias"]
    deltas = [*contributions, bias]
    colors = [
        COLORS["blue_dark"],
        COLORS["green_dark"],
        COLORS["red_dark"],
        COLORS["muted"],
    ]
    for delta, label, color in zip(deltas, labels, colors, strict=False):
        nxt = current + delta
        ax_sum.plot(
            [current, nxt],
            [0.46, 0.46],
            color=color,
            linewidth=14,
            solid_capstyle="round",
            alpha=0.90,
        )
        ax_sum.scatter([nxt], [0.46], s=58, color=color, zorder=3)
        if label != "bias":
            ax_sum.text(
                (current + nxt) / 2,
                0.70 if delta >= 0 else 0.22,
                f"{label}\n{delta:+.2f}",
                ha="center",
                va="center",
                fontsize=10.4,
            )
        current = nxt
    ax_sum.text(
        z,
        0.78,
        f"score z = {z:.2f}",
        ha="center",
        fontsize=12.6,
        fontweight="bold",
        color=COLORS["ink"],
    )

    raw = np.linspace(-4, 4, 400)
    curve = 1 / (1 + np.exp(-raw))
    ax_act.plot(raw, curve, color=COLORS["purple"], linewidth=2.8)
    ax_act.scatter([z], [activation], s=72, color=COLORS["purple"], zorder=3)
    ax_act.axvline(z, color="#cbd5e1", linewidth=1.2, linestyle="--")
    ax_act.axhline(activation, color="#cbd5e1", linewidth=1.2, linestyle="--")
    ax_act.axhline(0.5, color="#e2e8f0", linewidth=1.0, linestyle=":")
    ax_act.set_xlim(-4, 4)
    ax_act.set_ylim(0, 1.02)
    ax_act.set_title(
        "how strongly the unit fires",
        loc="left",
        fontsize=13.8,
        pad=8,
        fontweight="bold",
    )
    ax_act.set_xlabel("pre-activation score z")
    ax_act.set_ylabel("activation a")
    ax_act.grid(alpha=0.14)
    sns.despine(ax=ax_act)
    ax_act.text(
        0.05,
        0.90,
        f"alarm activity = {activation:.2f}",
        transform=ax_act.transAxes,
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["ink"],
    )

    save(fig, "neuron-scoring-rule.png")


def image_weights_biases():
    feature_names = ["cold room", "someone home", "afternoon sun"]
    values = np.array([0.80, 1.00, 0.60])
    weights = np.array([1.10, 0.50, -0.70])
    contributions = values * weights
    feature_score = contributions.sum()
    modes = [
        ("comfort mode", -0.70, "heat turns on"),
        ("eco mode", -1.20, "heat stays off"),
    ]

    fig = plt.figure(figsize=(13.8, 6.2), facecolor="white")
    fig.text(
        0.05,
        0.95,
        "Same room, different thermostat mode",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "Weights build the room score. Bias decides how much score is enough to heat.",
        fontsize=12.0,
        color=COLORS["muted"],
    )

    outer = fig.add_gridspec(
        2,
        1,
        left=0.05,
        right=0.98,
        bottom=0.10,
        top=0.88,
        height_ratios=[1.02, 0.88],
        hspace=0.38,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.40, 0.60], wspace=0.18)
    bottom = outer[1].subgridspec(1, 2, wspace=0.18)
    ax_table = fig.add_subplot(top[0, 0])
    ax_score = fig.add_subplot(top[0, 1])
    ax_mode_a = fig.add_subplot(bottom[0, 0])
    ax_mode_b = fig.add_subplot(bottom[0, 1])

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")
    ax_table.text(
        0.02,
        0.93,
        "current signals",
        fontsize=14.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax_table.text(0.02, 0.84, "signal", fontsize=12.2, fontweight="bold")
    ax_table.text(0.54, 0.84, "reading", fontsize=12.2, fontweight="bold", ha="right")
    ax_table.text(0.69, 0.84, "weight", fontsize=12.2, fontweight="bold", ha="right")
    ax_table.text(0.90, 0.84, "pull", fontsize=12.2, fontweight="bold", ha="center")

    zero_x = 0.85
    ax_table.plot([zero_x, zero_x], [0.18, 0.77], color="#cbd5e1", linewidth=1)
    row_y = [0.67, 0.47, 0.27]
    scale = 0.10 / np.max(np.abs(contributions))
    row_colors = [COLORS["blue_dark"], COLORS["green_dark"], COLORS["red_dark"]]

    for name, value, weight, contribution, y, color in zip(
        feature_names, values, weights, contributions, row_y, row_colors, strict=False
    ):
        ax_table.text(0.02, y, name, fontsize=12.4, va="center")
        ax_table.text(0.54, y, f"{value:.2f}", fontsize=12.4, va="center", ha="right")
        ax_table.text(0.69, y, f"{weight:+.2f}", fontsize=12.4, va="center", ha="right")
        width = contribution * scale
        left = zero_x if contribution >= 0 else zero_x + width
        ax_table.add_patch(
            Rectangle(
                (left, y - 0.045),
                abs(width),
                0.09,
                facecolor=color,
                edgecolor="none",
                alpha=0.92,
                transform=ax_table.transAxes,
            )
        )
        ax_table.text(
            0.98, y, f"{contribution:+.2f}", fontsize=12.2, va="center", ha="right"
        )
        ax_table.plot([0.02, 0.98], [y - 0.10, y - 0.10], color="#eef2f7", linewidth=1)

    ax_score.set_xlim(-0.08, 1.18)
    ax_score.set_ylim(0, 1)
    ax_score.set_yticks([])
    ax_score.set_xticks(np.arange(0.0, 1.21, 0.25))
    ax_score.grid(axis="x", alpha=0.10)
    sns.despine(ax=ax_score, left=True, bottom=False)
    ax_score.text(
        0.00,
        0.95,
        "room score before bias",
        transform=ax_score.transAxes,
        fontsize=14.2,
        fontweight="bold",
    )
    ax_score.text(
        0.00,
        0.86,
        f"{contributions[0]:.2f} + {contributions[1]:.2f} - {abs(contributions[2]):.2f} = {feature_score:.2f}",
        transform=ax_score.transAxes,
        fontsize=11.5,
        color=COLORS["muted"],
    )
    ax_score.hlines(0.45, -0.08, 1.18, color="#cbd5e1", linewidth=2)
    ax_score.axvline(0, color=COLORS["ink"], linewidth=1.6)

    current = 0.0
    labels = ["cold room", "someone home", "sunlight"]
    for delta, label, color in zip(contributions, labels, row_colors, strict=False):
        nxt = current + delta
        ax_score.plot(
            [current, nxt],
            [0.45, 0.45],
            color=color,
            linewidth=16,
            solid_capstyle="round",
            alpha=0.92,
        )
        ax_score.scatter([nxt], [0.45], s=62, color=color, zorder=3)
        ax_score.text(
            (current + nxt) / 2,
            0.69 if delta >= 0 else 0.22,
            f"{label}\n{delta:+.2f}",
            ha="center",
            va="center",
            fontsize=10.5,
        )
        current = nxt
    ax_score.text(
        feature_score,
        0.82,
        f"room score = {feature_score:.2f}",
        ha="center",
        fontsize=12.6,
        fontweight="bold",
        color=COLORS["blue_dark"],
    )

    for axis, (mode_name, bias, outcome) in zip(
        (ax_mode_a, ax_mode_b), modes, strict=False
    ):
        threshold = -bias
        z = feature_score + bias
        heat_on = z > 0
        axis.set_xlim(0.0, 1.35)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xticks(np.arange(0.0, 1.36, 0.25))
        axis.grid(axis="x", alpha=0.10)
        sns.despine(ax=axis, left=True, bottom=False)
        axis.set_title(mode_name, fontsize=14.0, pad=8, fontweight="bold")
        axis.hlines(0.43, 0.0, 1.35, color="#cbd5e1", linewidth=2)
        axis.axvline(threshold, color=COLORS["ink"], linewidth=2)
        axis.scatter([feature_score], [0.43], s=82, color=COLORS["blue_dark"], zorder=3)
        axis.text(
            0.03,
            0.87,
            f"bias = {bias:+.2f}",
            transform=axis.transAxes,
            fontsize=11.0,
            color=COLORS["muted"],
        )
        axis.text(
            0.03,
            0.73,
            outcome,
            transform=axis.transAxes,
            fontsize=12.2,
            fontweight="bold",
            color=COLORS["green_dark"] if heat_on else COLORS["red_dark"],
        )
        axis.text(
            0.74,
            0.87,
            f"z = {z:+.2f}",
            transform=axis.transAxes,
            fontsize=11.4,
            fontweight="bold",
            color=COLORS["ink"],
        )
        axis.text(
            threshold,
            0.74,
            f"cutoff {threshold:.2f}",
            ha="center",
            fontsize=10.3,
            color=COLORS["muted"],
        )
        axis.text(
            feature_score,
            0.18,
            f"same room score {feature_score:.2f}",
            ha="center",
            fontsize=10.3,
            color=COLORS["blue_dark"],
        )

    save(fig, "weights-biases.png")


def image_backprop_blame_assignment():
    fig, ax = plt.subplots(figsize=(13.6, 5.6), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.05,
        0.95,
        "Backprop in one still frame",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "Forward pass makes the prediction. Backward pass sends correction back through the same path.",
        fontsize=11.7,
        color=COLORS["muted"],
    )

    positions = {
        "x1": (0.10, 0.72),
        "x2": (0.10, 0.36),
        "h1": (0.36, 0.80),
        "h2": (0.36, 0.28),
        "yhat": (0.64, 0.54),
        "loss": (0.86, 0.54),
    }
    forward_edges = [
        ("x1", "h1"),
        ("x1", "h2"),
        ("x2", "h1"),
        ("x2", "h2"),
        ("h1", "yhat"),
        ("h2", "yhat"),
    ]
    for a, b in forward_edges:
        ax.add_patch(
            FancyArrowPatch(
                positions[a],
                positions[b],
                arrowstyle="-",
                linewidth=2.8,
                color=COLORS["blue_dark"],
                alpha=0.38,
            )
        )
    for a, b in [
        ("loss", "yhat"),
        ("yhat", "h1"),
        ("yhat", "h2"),
        ("h1", "x1"),
        ("h2", "x2"),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                positions[a],
                positions[b],
                arrowstyle="->",
                linewidth=2.6,
                color=COLORS["purple"],
                alpha=0.82,
                mutation_scale=12,
                connectionstyle="arc3,rad=0.08",
            )
        )

    for key, label, value, face, edge in [
        ("x1", r"$x_1$", "0.80", "#e0f2fe", COLORS["blue_dark"]),
        ("x2", r"$x_2$", "0.40", "#e0f2fe", COLORS["blue_dark"]),
        ("h1", r"$h_1$", "0.71", "white", "#94a3b8"),
        ("h2", r"$h_2$", "0.29", "white", "#94a3b8"),
        ("yhat", r"$\hat{y}$", "0.63", "#fef3c7", COLORS["amber_dark"]),
    ]:
        x, y = positions[key]
        ax.add_patch(
            Circle(
                (x, y),
                0.050 if key.startswith("x") else 0.056,
                facecolor=face,
                edgecolor=edge,
                linewidth=2.1,
            )
        )
        ax.text(
            x,
            y + 0.002,
            label,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
        ax.text(
            x,
            y - 0.090,
            value,
            ha="center",
            va="center",
            fontsize=11.0,
            color=COLORS["muted"],
        )

    loss_box = FancyBboxPatch(
        (0.80, 0.44),
        0.13,
        0.18,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#fff7ed",
        edgecolor=COLORS["amber_dark"],
        linewidth=1.8,
    )
    ax.add_patch(loss_box)
    ax.text(0.865, 0.58, "target = 1.00", ha="center", fontsize=11.2)
    ax.text(0.865, 0.53, "prediction = 0.63", ha="center", fontsize=11.2)
    ax.text(
        0.865,
        0.47,
        r"$L = (\hat{y} - y)^2$",
        ha="center",
        fontsize=11.2,
        color=COLORS["muted"],
    )
    ax.text(
        0.05,
        0.15,
        "blue = forward prediction path",
        fontsize=11.0,
        color=COLORS["blue_dark"],
    )
    ax.text(
        0.05,
        0.10,
        "purple = backward correction path",
        fontsize=11.0,
        color=COLORS["purple"],
    )

    save(fig, "backprop-blame-assignment.png")


def image_backprop_animation():
    fig, (ax_net, ax_bar) = plt.subplots(1, 2, figsize=(13.8, 5.8), facecolor="white")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.84, wspace=0.18)
    fig.text(
        0.05,
        0.95,
        "Backprop is one forward pass, then one backward correction pass",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "Blue shows the prediction being built. Purple shows the error signal coming back and turning into parameter updates.",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    positions = {
        "x1": (0.12, 0.72),
        "x2": (0.12, 0.38),
        "h1": (0.40, 0.80),
        "h2": (0.40, 0.28),
        "yhat": (0.68, 0.54),
        "loss": (0.89, 0.54),
    }
    forward_edges = [
        ("x1", "h1"),
        ("x1", "h2"),
        ("x2", "h1"),
        ("x2", "h2"),
        ("h1", "yhat"),
        ("h2", "yhat"),
    ]
    back_edges = [
        ("loss", "yhat"),
        ("yhat", "h1"),
        ("yhat", "h2"),
        ("h1", "x1"),
        ("h2", "x2"),
    ]
    bars = [
        ("x1→h1", -0.41),
        ("x2→h2", 0.62),
        ("h1→ŷ", -0.84),
        ("h2→ŷ", 0.28),
        ("b_h1", -0.19),
        ("b_ŷ", 0.11),
    ]
    max_abs = max(abs(v) for _, v in bars)
    frames = list(range(28))

    def node(ax, xy, label, value, face, edge, alpha=1.0, radius=0.052):
        ax.add_patch(
            Circle(
                xy, radius, facecolor=face, edgecolor=edge, linewidth=2.0, alpha=alpha
            )
        )
        ax.text(
            xy[0],
            xy[1] + 0.002,
            label,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            alpha=alpha,
        )
        ax.text(
            xy[0],
            xy[1] - 0.090,
            value,
            ha="center",
            va="center",
            fontsize=10.6,
            color=COLORS["muted"],
            alpha=alpha,
        )

    def draw(frame):
        ax_net.clear()
        ax_bar.clear()
        ax_net.set_xlim(0, 1)
        ax_net.set_ylim(0, 1)
        ax_net.axis("off")

        forward_phase = min(frame / 10.0, 1.0)
        backward_phase = max(0.0, min((frame - 10) / 12.0, 1.0))

        ax_net.text(
            0.00,
            0.95,
            "prediction path",
            fontsize=13.5,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax_net.text(
            0.00,
            0.89,
            "forward" if backward_phase == 0 else "forward complete",
            fontsize=10.6,
            color=COLORS["muted"],
        )

        for a, b in forward_edges:
            ax_net.add_patch(
                FancyArrowPatch(
                    positions[a],
                    positions[b],
                    arrowstyle="-",
                    linewidth=2.4,
                    color="#dbe4ee",
                    alpha=0.9,
                )
            )
        for idx, (a, b) in enumerate(forward_edges):
            local = max(0.0, min(forward_phase * len(forward_edges) - idx, 1.0))
            if local > 0:
                ax_net.add_patch(
                    FancyArrowPatch(
                        positions[a],
                        positions[b],
                        arrowstyle="-",
                        linewidth=3.0,
                        color=COLORS["blue_dark"],
                        alpha=0.25 + 0.55 * local,
                    )
                )

        node(ax_net, positions["x1"], r"$x_1$", "0.80", "#e0f2fe", COLORS["blue_dark"])
        node(ax_net, positions["x2"], r"$x_2$", "0.40", "#e0f2fe", COLORS["blue_dark"])
        node(
            ax_net,
            positions["h1"],
            r"$h_1$",
            "0.71",
            "white",
            "#94a3b8",
            alpha=0.35 + 0.65 * min(forward_phase * 1.5, 1.0),
            radius=0.056,
        )
        node(
            ax_net,
            positions["h2"],
            r"$h_2$",
            "0.29",
            "white",
            "#94a3b8",
            alpha=0.35 + 0.65 * min(forward_phase * 1.5, 1.0),
            radius=0.056,
        )
        node(
            ax_net,
            positions["yhat"],
            r"$\hat{y}$",
            "0.63",
            "#fef3c7",
            COLORS["amber_dark"],
            alpha=0.25 + 0.75 * max(0.0, min((forward_phase - 0.55) / 0.45, 1.0)),
            radius=0.060,
        )

        loss_alpha = max(0.0, min((forward_phase - 0.75) / 0.25, 1.0))
        loss_box = FancyBboxPatch(
            (0.82, 0.44),
            0.14,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#fff7ed",
            edgecolor=COLORS["amber_dark"],
            linewidth=1.8,
            alpha=max(loss_alpha, 0.12),
        )
        ax_net.add_patch(loss_box)
        ax_net.text(
            0.89, 0.58, "target = 1.00", ha="center", fontsize=11.0, alpha=loss_alpha
        )
        ax_net.text(
            0.89,
            0.53,
            "prediction = 0.63",
            ha="center",
            fontsize=11.0,
            alpha=loss_alpha,
        )
        ax_net.text(
            0.89,
            0.47,
            r"$L = (\hat{y} - y)^2$",
            ha="center",
            fontsize=11.0,
            color=COLORS["muted"],
            alpha=loss_alpha,
        )

        if backward_phase > 0:
            ax_net.text(
                0.42,
                0.71,
                "correction signal comes back through the same graph",
                fontsize=10.8,
                color=COLORS["purple"],
                fontweight="bold",
            )
        for idx, (a, b) in enumerate(back_edges):
            local = max(0.0, min(backward_phase * len(back_edges) - idx, 1.0))
            if local > 0:
                ax_net.add_patch(
                    FancyArrowPatch(
                        positions[a],
                        positions[b],
                        arrowstyle="->",
                        linewidth=2.4 + local,
                        color=COLORS["purple"],
                        alpha=0.30 + 0.55 * local,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=0.08",
                    )
                )

        ax_bar.set_title(
            "parameter updates", loc="left", fontsize=13.5, pad=10, fontweight="bold"
        )
        labels = [name for name, _ in bars]
        finals = np.array([value for _, value in bars])
        y = np.arange(len(bars))
        shown = np.zeros_like(finals)
        if backward_phase > 0:
            for idx, value in enumerate(finals):
                local = max(0.0, min(backward_phase * len(finals) - idx, 1.0))
                shown[idx] = value * local
        colors = np.where(shown >= 0, COLORS["purple"], "#6d28d9")
        ax_bar.barh(y, shown, color=colors, alpha=0.88, height=0.72)
        ax_bar.axvline(0, color="#cbd5e1", linewidth=1.3)
        ax_bar.set_yticks(y, labels=labels)
        ax_bar.set_xlabel("signed gradient")
        ax_bar.grid(axis="x", alpha=0.12)
        sns.despine(ax=ax_bar, left=True, bottom=False)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(-max_abs * 1.18, max_abs * 1.18)
        for idx, value in enumerate(shown):
            if abs(value) > 1e-3:
                ax_bar.text(
                    value + (0.03 if value >= 0 else -0.03),
                    idx,
                    f"{value:+.2f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=10.2,
                )
        ax_bar.text(
            0.02,
            0.96,
            "update rule: w ← w - η·gradient",
            transform=ax_bar.transAxes,
            va="top",
            fontsize=10.8,
            color=COLORS["muted"],
        )
        ax_bar.text(
            0.02,
            0.88,
            "bars stay near zero until the backward pass starts",
            transform=ax_bar.transAxes,
            va="top",
            fontsize=10.2,
            color=COLORS["muted"],
        )

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=120)
    anim.save(
        OUT_DIR / "backprop-blame-assignment.gif", writer=animation.PillowWriter(fps=8)
    )
    plt.close(fig)


def image_debugging_checklist():
    fig, ax = plt.subplots(figsize=(12.8, 5.0), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.06,
        0.92,
        "A practical debugging pass",
        fontsize=19.5,
        fontweight="bold",
        color=COLORS["ink"],
    )

    steps = [
        ("01", "task", "What exactly should the model predict?", COLORS["blue_dark"]),
        ("02", "signals", "Which inputs should matter most?", COLORS["green_dark"]),
        ("03", "capacity", "Is the model too weak or too large?", COLORS["amber_dark"]),
        ("04", "feedback", "Is the correction signal useful?", COLORS["red_dark"]),
        ("05", "data", "Is the dataset the real bottleneck?", COLORS["purple"]),
    ]
    y_positions = np.linspace(0.74, 0.22, len(steps))
    for y, (num, label, question, color) in zip(y_positions, steps, strict=False):
        ax.plot([0.16, 0.90], [y - 0.065, y - 0.065], color="#e2e8f0", linewidth=1)
        ax.add_patch(
            Rectangle((0.16, y - 0.035), 0.012, 0.07, facecolor=color, edgecolor="none")
        )
        ax.text(0.19, y, num, va="center", fontsize=9.5, color=COLORS["muted"])
        ax.text(
            0.25,
            y + 0.018,
            label,
            va="center",
            fontsize=12.3,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.25, y - 0.022, question, va="center", fontsize=10.8, color=COLORS["ink"]
        )

    ax.text(
        0.16,
        0.11,
        "If a step is broken, stop there and fix it before moving on.",
        fontsize=10.8,
        color=COLORS["muted"],
    )
    save(fig, "debugging-checklist.png")


def main():
    image_hero_editorial()
    image_explanation_problem()
    image_neuron_scoring_rule()
    image_weights_biases()
    image_weights_bias_animation()
    image_activation_functions()
    image_representation_building()
    image_backprop_blame_assignment()
    image_backprop_animation()
    image_debugging_checklist()
    print("Generated images in", OUT_DIR)


if __name__ == "__main__":
    main()
