from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyArrowPatch, Rectangle

warnings.filterwarnings("ignore", message="facecolor will have no effect.*")


OUT_DIR = Path("assets/img/posts/kawasaki-wind-patterns")
CACHE_DIR = Path(".cache/kawasaki-wind")
CACHE_FILE = CACHE_DIR / "ncep_pacific_1996_2006.nc"

U_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/uwnd.mon.mean.nc"
V_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/vwnd.mon.mean.nc"

PERIOD = slice("1996-01-01", "2006-12-31")
LEVELS = [300.0, 850.0]

COLORS = {
    "bg": "#f4f6f4",
    "panel": "#ffffff",
    "ink": "#19221f",
    "muted": "#66736d",
    "grid": "#d7ddd9",
    "water": "#ddebf0",
    "land": "#efe7d7",
    "coast": "#65706b",
    "blue": "#2b6f91",
    "deep_blue": "#174a68",
    "green": "#4f8761",
    "gold": "#c8912d",
    "red": "#b6533f",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["muted"],
        "axes.titlecolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
    }
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LOCATIONS = {
    "Japan": (139.7, 35.7),
    "Hawaii": (202.2, 21.3),
    "San Diego": (242.8, 32.7),
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_winds() -> xr.Dataset:
    ensure_dirs()
    if CACHE_FILE.exists():
        return xr.open_dataset(CACHE_FILE).load()

    u = xr.open_dataset(U_URL)["uwnd"].sel(
        time=PERIOD,
        level=LEVELS,
        lat=slice(65, 5),
        lon=slice(100, 260),
    )
    v = xr.open_dataset(V_URL)["vwnd"].sel(
        time=PERIOD,
        level=LEVELS,
        lat=slice(65, 5),
        lon=slice(100, 260),
    )
    ds = xr.Dataset({"u": u, "v": v}).load()
    ds.to_netcdf(CACHE_FILE, engine="netcdf4")
    return ds


def save(fig: plt.Figure, name: str, *, pad: float = 0.10) -> None:
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def stroke_text(text_obj, lw: float = 3.0) -> None:
    text_obj.set_path_effects([pe.withStroke(linewidth=lw, foreground=COLORS["panel"])])


def add_pacific_context(ax, *, grid_labels: bool = False) -> None:
    ax.set_extent([110, 255, 15, 60], crs=ccrs.PlateCarree())
    ax.set_facecolor(COLORS["water"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=COLORS["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, color=COLORS["coast"], zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, color="#9b9a90", alpha=0.7, zorder=4)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=grid_labels,
        linewidth=0.55,
        color="#9fb0ac",
        alpha=0.38,
        linestyle="-",
    )
    if grid_labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8.5, "color": COLORS["muted"]}
        gl.ylabel_style = {"size": 8.5, "color": COLORS["muted"]}


def add_locations(ax) -> None:
    for label, (lon, lat) in LOCATIONS.items():
        ax.scatter(
            lon,
            lat,
            s=56,
            color=COLORS["red"],
            edgecolor="white",
            linewidth=0.9,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
        dx = 3.0 if label != "San Diego" else -18.5
        dy = 1.6 if label != "Hawaii" else -3.2
        text = ax.text(
            lon + dx,
            lat + dy,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=11.3,
            color=COLORS["ink"],
            weight="bold",
            zorder=8,
        )
        stroke_text(text, lw=3.0)


def jan_300_wind(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(level=300).where(ds.time.dt.month == 1, drop=True).mean("time")


def monthly_climatology(ds: xr.Dataset, level: float) -> xr.Dataset:
    return ds.sel(level=level).groupby("time.month").mean("time")


def p_wind(ds: xr.Dataset) -> xr.DataArray:
    u = ds.sel(level=300)["u"].sel(lat=35, method="nearest").sel(lon=slice(140, 240))
    return u.mean("lon").groupby("time.month").mean("time")


def nw_wind_japan(ds: xr.Dataset) -> xr.DataArray:
    low = ds.sel(level=850).sel(lat=slice(45, 30), lon=slice(130, 145))
    component = np.cos(np.deg2rad(45)) * low["u"] - np.sin(np.deg2rad(45)) * low["v"]
    weights = np.cos(np.deg2rad(component["lat"]))
    return component.weighted(weights).mean(("lat", "lon")).groupby("time.month").mean("time")


def image_hero(ds: xr.Dataset) -> None:
    wind = jan_300_wind(ds)
    speed = np.hypot(wind["u"], wind["v"])
    q_lon = wind["lon"].to_numpy()[::4]
    q_lat = wind["lat"].to_numpy()[::3]
    q_x, q_y = np.meshgrid(q_lon, q_lat)

    fig = plt.figure(figsize=(14.8, 7.9))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.90], projection=ccrs.PlateCarree(central_longitude=180))
    add_pacific_context(ax, grid_labels=False)

    ax.contourf(
        wind["lon"],
        wind["lat"],
        speed,
        levels=np.arange(10, 58, 4),
        cmap="YlGnBu",
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    ax.quiver(
        q_x,
        q_y,
        wind["u"].to_numpy()[::3, ::4],
        wind["v"].to_numpy()[::3, ::4],
        transform=ccrs.PlateCarree(),
        color="#14384d",
        scale=820,
        width=0.0024,
        alpha=0.82,
        zorder=5,
    )
    ax.plot([140, 240], [35, 35], color=COLORS["red"], linewidth=3.0, transform=ccrs.PlateCarree(), zorder=6)
    pwind = ax.text(
        170,
        36.8,
        "P-WIND line",
        color=COLORS["red"],
        fontsize=12.2,
        weight="bold",
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    stroke_text(pwind, lw=3.0)
    jet = ax.annotate(
        "winter jet",
        xy=(171, 46),
        xytext=(153, 53),
        xycoords=ccrs.PlateCarree(),
        textcoords=ccrs.PlateCarree(),
        color=COLORS["deep_blue"],
        fontsize=12.0,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["deep_blue"], "lw": 1.8},
        zorder=8,
    )
    stroke_text(jet, lw=3.0)
    add_locations(ax)

    fig.savefig(OUT_DIR / "hero.png", dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def annotate_peak(ax, x: np.ndarray, values: xr.DataArray, color: str, *, offset: tuple[float, float]) -> None:
    arr = values.to_numpy()
    idx = int(np.argmax(arr))
    month = MONTHS[idx]
    value = float(arr[idx])
    ax.scatter(x[idx], value, s=64, color=color, edgecolor="white", linewidth=0.9, zorder=4)
    ann = ax.annotate(
        f"{month}\n{value:.1f} m/s",
        xy=(x[idx], value),
        xytext=(x[idx] + offset[0], value + offset[1]),
        color=color,
        fontsize=10.8,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5, "shrinkA": 0, "shrinkB": 0},
        zorder=5,
    )
    stroke_text(ann, lw=2.8)


def image_seasonal_indices(ds: xr.Dataset) -> None:
    p = p_wind(ds)
    nw = nw_wind_japan(ds)
    x = np.arange(1, 13)

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.6), sharex=True)
    fig.patch.set_facecolor(COLORS["bg"])
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.82, hspace=0.24)
    specs = [
        (axes[0], p, "Pacific zonal wind at 300 hPa", COLORS["blue"], "Mean east-west wind along 35N, 140E-240E"),
        (axes[1], nw, "Japan northwesterly component at 850 hPa", COLORS["green"], "Area mean over 30-45N, 130-145E"),
    ]

    for ax, values, title, color, subtitle in specs:
        ax.axvspan(1, 3, color="#dbe9ef", alpha=0.7, zorder=0)
        ax.axvspan(11, 12, color="#dbe9ef", alpha=0.7, zorder=0)
        ax.plot(x, values, color=color, linewidth=3.0)
        ax.scatter(x, values, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_title(title, loc="left", pad=15, fontsize=13.8, fontweight="bold")
        subtitle_text = ax.text(0.0, 0.99, subtitle, transform=ax.transAxes, color=COLORS["muted"], fontsize=9.8)
        ax.set_ylabel("m/s")
        ax.grid(axis="y", alpha=0.45)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymin = float(np.floor(values.min() - 1.0))
        ymax = float(np.ceil(values.max() + 1.4))
        ax.set_ylim(ymin, ymax)
        annotate_peak(ax, x, values, color, offset=(-0.3, -1.2))
        subtitle_text.set_path_effects([pe.withStroke(linewidth=2.4, foreground=COLORS["panel"])])

    axes[-1].set_xticks(x, MONTHS)
    axes[-1].set_xlabel("month")
    fig.suptitle("The wind indices rise in the cool season", x=0.08, y=0.965, ha="left", fontsize=18, fontweight="bold", color=COLORS["ink"])
    save(fig, "seasonal-wind-indices.png")


def image_monthly_animation(ds: xr.Dataset) -> None:
    clim = monthly_climatology(ds, 300)
    speed = np.hypot(clim["u"], clim["v"])
    levels = np.arange(6, 58, 4)
    q_lon = clim["lon"].to_numpy()[::5]
    q_lat = clim["lat"].to_numpy()[::4]
    q_x, q_y = np.meshgrid(q_lon, q_lat)

    fig = plt.figure(figsize=(11.0, 6.6))
    ax = fig.add_axes([0.015, 0.03, 0.97, 0.95], projection=ccrs.PlateCarree(central_longitude=180))

    def draw(month_index: int):
        ax.clear()
        add_pacific_context(ax, grid_labels=False)
        month = month_index + 1
        u = clim["u"].sel(month=month)
        v = clim["v"].sel(month=month)
        spd = speed.sel(month=month)
        ax.contourf(
            clim["lon"],
            clim["lat"],
            spd,
            levels=levels,
            cmap="YlGnBu",
            extend="max",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.quiver(
            q_x,
            q_y,
            u.to_numpy()[::4, ::5],
            v.to_numpy()[::4, ::5],
            transform=ccrs.PlateCarree(),
            color="#14384d",
            scale=820,
            width=0.0027,
            alpha=0.82,
            zorder=5,
        )
        ax.plot([140, 240], [35, 35], color=COLORS["red"], linewidth=2.1, transform=ccrs.PlateCarree(), zorder=6)
        add_locations(ax)
        month_text = ax.text(
            0.02,
            0.962,
            MONTHS[month_index],
            transform=ax.transAxes,
            fontsize=21,
            weight="bold",
            color=COLORS["ink"],
            zorder=8,
        )
        stroke_text(month_text, lw=3.5)
        subtitle = ax.text(
            0.02,
            0.915,
            "Monthly climatology, 1996-2006",
            transform=ax.transAxes,
            fontsize=10.4,
            color=COLORS["muted"],
            zorder=8,
        )
        stroke_text(subtitle, lw=3.0)

    ani = animation.FuncAnimation(fig, draw, frames=12, interval=520, repeat=True)
    ani.save(OUT_DIR / "monthly-pacific-winds.gif", writer=animation.PillowWriter(fps=2), dpi=120)
    plt.close(fig)


def image_analysis_boundary() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(0.05, 0.93, "Evidence ladder", fontsize=18, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.875,
        "Four layers, each with a narrower claim than the one above it.",
        fontsize=10.9,
        color=COLORS["muted"],
    )

    rows = [
        ("1", "Observed clinical timing", COLORS["blue"], "Kawasaki disease peaks on a seasonal clock in the paper's case records."),
        ("2", "Reproducible wind fields", COLORS["green"], "NOAA reanalysis shows a cool-season North Pacific corridor."),
        ("3", "Transport hypothesis", COLORS["gold"], "A wind-borne trigger can move along that pathway."),
        ("4", "Still unknown", COLORS["red"], "The causal agent is not identified, so causation stays open."),
    ]

    top = 0.70
    row_h = 0.135
    gap = 0.04
    for i, (num, title, color, body) in enumerate(rows):
        y = top - i * (row_h + gap)
        ax.add_patch(Rectangle((0.06, y), 0.88, row_h, facecolor=COLORS["panel"], edgecolor="#d7ddd9", linewidth=1.1))
        ax.add_patch(Rectangle((0.06, y), 0.018, row_h, facecolor=color, edgecolor="none"))
        num_text = ax.text(0.09, y + 0.092, num, fontsize=20, fontweight="bold", color=color, va="center")
        title_text = ax.text(0.14, y + 0.098, title, fontsize=14.0, fontweight="bold", color=COLORS["ink"], va="center")
        body_text = ax.text(0.14, y + 0.043, textwrap.fill(body, width=78), fontsize=11.1, color=COLORS["muted"], va="center")
        stroke_text(num_text, lw=2.5)
        stroke_text(title_text, lw=2.8)
        stroke_text(body_text, lw=2.2)
        if i < len(rows) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0.50, y - 0.004),
                    (0.50, y - gap + 0.006),
                    arrowstyle="->",
                    mutation_scale=14,
                    linewidth=1.4,
                    color="#8b8a82",
                )
            )

    save(fig, "evidence-boundary.png")


def print_summary(ds: xr.Dataset) -> None:
    p = p_wind(ds)
    nw = nw_wind_japan(ds)
    p_arr = p.to_numpy()
    nw_arr = nw.to_numpy()
    print("period: 1996-2006 monthly means")
    print(f"p-wind max month: {MONTHS[int(np.argmax(p_arr))]} ({float(p.max()):.2f} m/s)")
    print(f"p-wind min month: {MONTHS[int(np.argmin(p_arr))]} ({float(p.min()):.2f} m/s)")
    print(f"japan nw-wind max month: {MONTHS[int(np.argmax(nw_arr))]} ({float(nw.max()):.2f} m/s)")
    print(f"japan nw-wind min month: {MONTHS[int(np.argmin(nw_arr))]} ({float(nw.min()):.2f} m/s)")


def main() -> None:
    ds = load_winds()
    image_hero(ds)
    image_seasonal_indices(ds)
    image_monthly_animation(ds)
    image_analysis_boundary()
    print_summary(ds)


if __name__ == "__main__":
    main()
