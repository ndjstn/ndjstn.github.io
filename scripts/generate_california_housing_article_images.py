from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("assets/img/posts/california-housing-prices")
CACHE_DIR = Path(".cache/california-housing")
ARCHIVE = CACHE_DIR / "cal_housing.tgz"
FIGSHARE_URL = "https://ndownloader.figshare.com/files/5976036"

COLORS = {
    "bg": "#f7f5ef",
    "panel": "#ffffff",
    "ink": "#17201b",
    "muted": "#69736d",
    "grid": "#dedbd2",
    "coast": "#0f6b7a",
    "gold": "#c38a22",
    "terracotta": "#bf5f45",
    "green": "#3f7f5f",
    "blue": "#2f6f96",
    "red": "#b64b3f",
}

VALUE_CMAP = LinearSegmentedColormap.from_list(
    "california_value",
    ["#233b53", "#2f6f96", "#4f936c", "#d9a441", "#d66b45", "#8f2f2a"],
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["muted"],
        "axes.titlecolor": COLORS["ink"],
        "axes.facecolor": COLORS["panel"],
        "figure.facecolor": COLORS["bg"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "savefig.facecolor": COLORS["bg"],
    }
)


def ensure_data() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        return

    subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", FIGSHARE_URL, "-o", str(ARCHIVE)],
        check=True,
    )


def load_data() -> pd.DataFrame:
    ensure_data()
    columns = [
        "Longitude",
        "Latitude",
        "HouseAge",
        "TotalRooms",
        "TotalBedrooms",
        "Population",
        "Households",
        "MedInc",
        "MedHouseValUSD",
    ]
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        member = archive.extractfile("CaliforniaHousing/cal_housing.data")
        if member is None:
            raise FileNotFoundError("CaliforniaHousing/cal_housing.data not found in archive")
        data = member.read()

    df = pd.read_csv(io.BytesIO(data), header=None, names=columns)
    df["MedHouseVal"] = df["MedHouseValUSD"] / 100_000
    df["AveRooms"] = df["TotalRooms"] / df["Households"]
    df["AveBedrms"] = df["TotalBedrooms"] / df["Households"]
    df["AveOccup"] = df["Population"] / df["Households"]
    df["IsCapped"] = df["MedHouseValUSD"] >= df["MedHouseValUSD"].max()
    return df


def save(fig: plt.Figure, name: str, *, pad: float = 0.18) -> None:
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def money_tick(value: float, _position: int | None = None) -> str:
    return f"${value / 1000:.0f}k"


def format_map(ax: plt.Axes, *, axes: bool = True) -> None:
    ax.set_aspect(1.18)
    ax.set_xlim(-124.7, -113.7)
    ax.set_ylim(32.1, 42.3)
    if axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.grid(True, alpha=0.55)
    else:
        ax.axis("off")


def add_note(ax: plt.Axes, text: str, x: float, y: float) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["muted"],
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.08",
            "facecolor": "#fbfaf5",
            "edgecolor": "#e6e1d7",
            "linewidth": 0.9,
        },
    )


def image_hero(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.06, 0.08, 0.58, 0.84])
    ax.set_facecolor("#f4f1e8")
    format_map(ax, axes=True)

    order = np.argsort(df["MedHouseValUSD"].to_numpy())
    sc = ax.scatter(
        df["Longitude"].to_numpy()[order],
        df["Latitude"].to_numpy()[order],
        c=df["MedHouseValUSD"].to_numpy()[order],
        s=8,
        cmap=VALUE_CMAP,
        alpha=0.82,
        linewidths=0,
    )
    ax.set_title("1990 median house value by census block group", loc="left", pad=12, fontsize=16, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.044, pad=0.022)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(money_tick)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("median value", color=COLORS["muted"], labelpad=8)

    ax_stats = fig.add_axes([0.72, 0.16, 0.22, 0.68])
    bins = np.arange(0, 525_001, 25_000)
    ax_stats.hist(df["MedHouseValUSD"], bins=bins, orientation="horizontal", color=COLORS["blue"], alpha=0.84)
    cap = df["MedHouseValUSD"].max()
    ax_stats.axhline(cap, color=COLORS["red"], linewidth=2.0)
    ax_stats.set_ylim(0, 525_000)
    ax_stats.set_xlabel("block groups")
    ax_stats.set_ylabel("median value")
    ax_stats.yaxis.set_major_formatter(money_tick)
    ax_stats.set_title("Target distribution", loc="left", pad=12, fontsize=14, fontweight="bold")
    ax_stats.grid(axis="x", alpha=0.5)
    ax_stats.grid(axis="y", visible=False)
    for spine in ("top", "right"):
        ax_stats.spines[spine].set_visible(False)

    fig.text(0.72, 0.88, "20,640 rows", fontsize=18, fontweight="bold", color=COLORS["ink"])
    fig.text(0.72, 0.84, "Each row is a census block group, not a sale.", fontsize=10.5, color=COLORS["muted"])
    fig.text(0.72, 0.07, "Source: StatLib / scikit-learn California housing dataset", fontsize=9.5, color=COLORS["muted"])
    fig.savefig(OUT_DIR / "hero.png", dpi=180)
    plt.close(fig)


def image_value_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    bins = np.arange(0, 525_001, 20_000)
    ax.hist(df["MedHouseValUSD"], bins=bins, color=COLORS["blue"], alpha=0.86, edgecolor="white")
    cap = df["MedHouseValUSD"].max()
    capped = int(df["IsCapped"].sum())
    pct = df["IsCapped"].mean() * 100
    ax.axvline(cap, color=COLORS["red"], linewidth=2.6)
    ax.annotate(
        f"{capped:,} block groups\nhit the ${cap / 1000:.0f}k ceiling",
        xy=(cap, 840),
        xytext=(365_000, 1_280),
        arrowprops={"arrowstyle": "->", "color": COLORS["red"], "lw": 1.4},
        color=COLORS["ink"],
        fontsize=11,
    )
    ax.set_title("The target has a hard ceiling", loc="left", pad=14, fontweight="bold")
    ax.text(
        0,
        1.01,
        f"{pct:.1f}% of rows are censored at the maximum value, so the right tail is not fully observed.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=10.5,
    )
    ax.set_xlabel("median house value")
    ax.set_ylabel("block groups")
    ax.xaxis.set_major_formatter(money_tick)
    ax.grid(axis="y", alpha=0.65)
    ax.grid(axis="x", visible=False)
    save(fig, "value-distribution.png")


def image_income_vs_value(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.7))
    hb = ax.hexbin(
        df["MedInc"],
        df["MedHouseValUSD"],
        gridsize=54,
        mincnt=1,
        cmap=VALUE_CMAP,
        linewidths=0,
        alpha=0.95,
    )
    cbar = fig.colorbar(hb, ax=ax, fraction=0.038, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("block groups in bin", color=COLORS["muted"])
    cap = df["MedHouseValUSD"].max()
    ax.axhline(cap, color=COLORS["red"], linewidth=1.8, linestyle="--")
    ax.text(0.3, cap - 22_000, "capped target", color=COLORS["red"], fontsize=10.5)
    corr = df[["MedInc", "MedHouseValUSD"]].corr().iloc[0, 1]
    ax.set_title("Median income explains the first pass", loc="left", pad=14, fontweight="bold")
    ax.text(
        0,
        1.01,
        f"Correlation with median house value: {corr:.3f}. The relationship is strong, but it flattens against the ceiling.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=10.5,
    )
    ax.set_xlabel("median income, in tens of thousands of dollars")
    ax.set_ylabel("median house value")
    ax.yaxis.set_major_formatter(money_tick)
    ax.grid(True, alpha=0.55)
    save(fig, "income-vs-value.png")


def image_location_price_map(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.3), gridspec_kw={"width_ratios": [1.05, 1]})
    ax_map, ax_rank = axes

    sc = ax_map.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedHouseValUSD"],
        s=5.7,
        cmap=VALUE_CMAP,
        alpha=0.78,
        linewidths=0,
    )
    format_map(ax_map)
    ax_map.set_title("Location is not a side variable", loc="left", pad=14, fontweight="bold")
    ax_map.text(
        0,
        1.01,
        "Latitude and longitude pull in coastal access, metro density, and regional labor markets.",
        transform=ax_map.transAxes,
        color=COLORS["muted"],
        fontsize=10.2,
    )
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.045, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(money_tick)

    features = ["MedInc", "Latitude", "AveRooms", "HouseAge", "AveOccup", "Population", "Longitude", "AveBedrms"]
    labels = [
        "median income",
        "latitude",
        "avg. rooms",
        "house age",
        "avg. occupancy",
        "population",
        "longitude",
        "avg. bedrooms",
    ]
    corr = df[features + ["MedHouseValUSD"]].corr(numeric_only=True)["MedHouseValUSD"].drop("MedHouseValUSD")
    corr = corr.reindex(features)
    y = np.arange(len(features))
    colors = [COLORS["green"] if value > 0 else COLORS["terracotta"] for value in corr]
    ax_rank.barh(y, corr, color=colors, alpha=0.9)
    ax_rank.axvline(0, color="#bdb7ab", linewidth=1)
    ax_rank.set_yticks(y, labels)
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(-0.22, 0.76)
    ax_rank.set_xlabel("correlation with median house value")
    ax_rank.set_title("A simple correlation read", loc="left", pad=14, fontweight="bold")
    for i, value in enumerate(corr):
        ax_rank.text(
            value + (0.018 if value >= 0 else -0.018),
            i,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color=COLORS["ink"],
            fontsize=10,
        )
    ax_rank.grid(axis="x", alpha=0.55)
    ax_rank.grid(axis="y", visible=False)
    save(fig, "location-and-correlations.png")


def image_model_check(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    X = df[features]
    y = df["MedHouseValUSD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=17,
        ),
    }
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        results[name] = {
            "mae": mean_absolute_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "r2": r2_score(y_test, pred),
        }

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4), gridspec_kw={"width_ratios": [0.82, 1.18]})
    ax_bar, ax_scatter = axes

    names = list(results)
    mae = [results[name]["mae"] for name in names]
    rmse = [results[name]["rmse"] for name in names]
    y_pos = np.arange(len(names))
    ax_bar.barh(y_pos - 0.16, mae, height=0.32, label="MAE", color=COLORS["green"])
    ax_bar.barh(y_pos + 0.16, rmse, height=0.32, label="RMSE", color=COLORS["blue"])
    ax_bar.set_yticks(y_pos, names)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("dollars")
    ax_bar.xaxis.set_major_formatter(money_tick)
    ax_bar.set_title("A baseline check", loc="left", pad=14, fontweight="bold")
    for idx, name in enumerate(names):
        ax_bar.text(
            max(mae[idx], rmse[idx]) + 3_000,
            idx,
            f"R2 {results[name]['r2']:.3f}",
            va="center",
            color=COLORS["muted"],
            fontsize=10.5,
        )
    ax_bar.legend(frameon=False, loc="lower right")
    ax_bar.grid(axis="x", alpha=0.55)
    ax_bar.grid(axis="y", visible=False)

    pred = predictions["Gradient boosting"]
    sample = np.random.default_rng(17).choice(len(y_test), size=1_700, replace=False)
    ax_scatter.scatter(y_test.to_numpy()[sample], pred[sample], s=11, color=COLORS["blue"], alpha=0.42, linewidths=0)
    lims = [0, 525_000]
    ax_scatter.plot(lims, lims, color=COLORS["red"], linewidth=1.5)
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel("actual median house value")
    ax_scatter.set_ylabel("predicted median house value")
    ax_scatter.xaxis.set_major_formatter(money_tick)
    ax_scatter.yaxis.set_major_formatter(money_tick)
    ax_scatter.set_title("Predictions still miss the edges", loc="left", pad=14, fontweight="bold")
    add_note(
        ax_scatter,
        "The cap makes expensive areas\nhard to read as a real tail.",
        0.58,
        0.18,
    )
    ax_scatter.grid(True, alpha=0.55)
    save(fig, "model-check.png")
    return results


def image_income_animation(df: pd.DataFrame) -> None:
    bands = pd.qcut(df["MedInc"], 4, labels=["lowest", "lower-middle", "upper-middle", "highest"])
    df = df.assign(IncomeBand=bands)
    band_names = ["lowest", "lower-middle", "upper-middle", "highest"]
    band_labels = {
        "lowest": "lowest income quartile",
        "lower-middle": "lower-middle income quartile",
        "upper-middle": "upper-middle income quartile",
        "highest": "highest income quartile",
    }

    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    ax.set_facecolor("#f4f1e8")
    format_map(ax, axes=False)
    ax.scatter(df["Longitude"], df["Latitude"], s=3.6, color="#c8c2b7", alpha=0.22, linewidths=0)
    highlight = ax.scatter([], [], s=7.2, c=[], cmap=VALUE_CMAP, vmin=0, vmax=df["MedHouseValUSD"].max(), alpha=0.92, linewidths=0)
    label = ax.text(0.04, 0.93, "", transform=ax.transAxes, fontsize=17, fontweight="bold", color=COLORS["ink"])
    sublabel = ax.text(0.04, 0.885, "", transform=ax.transAxes, fontsize=10.8, color=COLORS["muted"])

    def update(frame: int):
        band = band_names[frame]
        subset = df[df["IncomeBand"] == band]
        points = subset[["Longitude", "Latitude"]].to_numpy()
        highlight.set_offsets(points)
        highlight.set_array(subset["MedHouseValUSD"].to_numpy())
        median_value = subset["MedHouseValUSD"].median()
        label.set_text(band_labels[band])
        sublabel.set_text(f"median block-group value: ${median_value / 1000:.0f}k")
        return highlight, label, sublabel

    ani = animation.FuncAnimation(fig, update, frames=len(band_names), interval=1050, blit=False, repeat=True)
    ani.save(OUT_DIR / "income-map-animation.gif", writer=animation.PillowWriter(fps=1), dpi=135)
    plt.close(fig)


def print_summary(df: pd.DataFrame, model_results: dict[str, dict[str, float]]) -> None:
    cap = df["MedHouseValUSD"].max()
    capped = int(df["IsCapped"].sum())
    print(f"rows: {len(df):,}")
    print(f"target median: ${df['MedHouseValUSD'].median():,.0f}")
    print(f"target mean: ${df['MedHouseValUSD'].mean():,.0f}")
    print(f"target cap: ${cap:,.0f}; capped rows: {capped:,} ({df['IsCapped'].mean() * 100:.1f}%)")
    print(f"MedInc correlation: {df[['MedInc', 'MedHouseValUSD']].corr().iloc[0, 1]:.3f}")
    for name, values in model_results.items():
        print(
            f"{name}: MAE ${values['mae']:,.0f}; "
            f"RMSE ${values['rmse']:,.0f}; R2 {values['r2']:.3f}"
        )


def main() -> None:
    df = load_data()
    image_hero(df)
    image_value_distribution(df)
    image_income_vs_value(df)
    image_location_price_map(df)
    model_results = image_model_check(df)
    image_income_animation(df)
    print_summary(df, model_results)


if __name__ == "__main__":
    main()
