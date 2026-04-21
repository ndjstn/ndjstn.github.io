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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

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

WIND_CMAP = LinearSegmentedColormap.from_list(
    "pacific_wind_speed",
    ["#f3f6c8", "#b9dfbf", "#69c7c4", "#2389bd", "#182f7c"],
)

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

SOURCE_REGIONS = {
    "Siberia": (128, 55.2),
    "Mongolia /\nnorth China": (116.5, 43.0),
}

FEATURE_LABELS = {
    "Kuril Islands": (151.5, 46.0),
    "Aleutian arc": (181, 52.7),
    "Bering Sea": (188, 58.0),
    "Gulf of Alaska": (219, 55.2),
    "North Pacific": (184, 24.0),
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
        dy = 1.6
        if label == "Hawaii":
            dx = 3.0
            dy = 1.5
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


def add_source_regions(ax) -> None:
    for label, (lon, lat) in SOURCE_REGIONS.items():
        text = ax.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=9.2,
            color="#5e5135",
            weight="bold",
            ha="center",
            va="center",
            zorder=8,
        )
        stroke_text(text, lw=3.0)


def add_feature_labels(ax, *, include_north_pacific: bool = True) -> None:
    for label, (lon, lat) in FEATURE_LABELS.items():
        if label == "North Pacific" and not include_north_pacific:
            continue
        text = ax.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=8.4,
            color="#586661",
            style="italic",
            weight="bold",
            alpha=0.90,
            zorder=8,
        )
        stroke_text(text, lw=2.6)


def add_corridor(ax, *, label: bool = True, wide: bool = False) -> None:
    if wide:
        ax.add_patch(
            Rectangle(
                (140, 34.0),
                100,
                2.0,
                transform=ccrs.PlateCarree(),
                facecolor=COLORS["red"],
                edgecolor="none",
                alpha=0.12,
                zorder=4,
            )
        )
    ax.plot(
        [140, 240],
        [35, 35],
        color=COLORS["red"],
        linewidth=2.1,
        linestyle=(0, (7, 5)),
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    if label:
        text = ax.text(
            169,
            37.4,
            "35N averaging corridor\nnot a literal route",
            color=COLORS["red"],
            fontsize=9.2,
            weight="bold",
            transform=ccrs.PlateCarree(),
            zorder=8,
        )
        stroke_text(text, lw=3.0)


def add_pathway_cues(ax, *, northern: bool = True) -> None:
    arrow_style = {
        "arrowstyle": "-|>",
        "color": COLORS["deep_blue"],
        "lw": 2.0,
        "mutation_scale": 16,
        "alpha": 0.82,
        "connectionstyle": "arc3,rad=0.05",
    }
    for start, end in [
        ((118, 43), (139.5, 36.5)),
        ((146, 37.5), (174, 39.0)),
        ((180, 39.0), (209, 36.5)),
        ((215, 35.5), (241, 33.2)),
    ]:
        ax.annotate("", xy=end, xytext=start, xycoords=ccrs.PlateCarree(), textcoords=ccrs.PlateCarree(), arrowprops=arrow_style, zorder=8)

    if not northern:
        return

    ax.plot(
        [150, 174, 202, 226],
        [47, 53.5, 56, 50],
        color=COLORS["gold"],
        linewidth=2.0,
        linestyle=(0, (4, 5)),
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(226, 50),
        xytext=(202, 56),
        xycoords=ccrs.PlateCarree(),
        textcoords=ccrs.PlateCarree(),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["gold"], "lw": 1.8, "mutation_scale": 15},
        zorder=8,
    )
    text = ax.text(
        187,
        57.0,
        "northern route to test",
        transform=ccrs.PlateCarree(),
        fontsize=8.8,
        color=COLORS["gold"],
        weight="bold",
        zorder=8,
    )
    stroke_text(text, lw=2.8)


def add_wind_speed_key(ax) -> None:
    box = Rectangle(
        (0.705, 0.032),
        0.270,
        0.122,
        transform=ax.transAxes,
        facecolor=COLORS["panel"],
        edgecolor="#d7ddd9",
        linewidth=0.8,
        alpha=0.90,
        zorder=8,
    )
    ax.add_patch(box)
    ax.text(0.724, 0.122, "wind speed (m/s)", transform=ax.transAxes, fontsize=8.4, color=COLORS["ink"], weight="bold", zorder=9)
    swatches = ["#f3f6c8", "#b9dfbf", "#69c7c4", "#2389bd", "#182f7c"]
    values = ["10", "20", "30", "40", "50+"]
    x0 = 0.724
    w = 0.043
    for i, (color, value) in enumerate(zip(swatches, values)):
        x = x0 + i * w
        ax.add_patch(Rectangle((x, 0.086), w, 0.018, transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=9))
        ax.text(x + w / 2, 0.061, value, transform=ax.transAxes, fontsize=7.1, color=COLORS["muted"], ha="center", zorder=9)
    ax.text(0.724, 0.041, "lines = direction; red = analysis corridor", transform=ax.transAxes, fontsize=7.1, color=COLORS["muted"], zorder=9)


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
    q_lon = wind["lon"].to_numpy()[::7]
    q_lat = wind["lat"].to_numpy()[::4]
    q_x, q_y = np.meshgrid(q_lon, q_lat)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.025, 0.060, 0.95, 0.895], projection=ccrs.PlateCarree(central_longitude=180))
    add_pacific_context(ax, grid_labels=False)
    ax.set_aspect("auto")

    ax.contourf(
        wind["lon"],
        wind["lat"],
        speed,
        levels=np.arange(10, 58, 4),
        cmap=WIND_CMAP,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    ax.quiver(
        q_x,
        q_y,
        wind["u"].to_numpy()[::4, ::7],
        wind["v"].to_numpy()[::4, ::7],
        transform=ccrs.PlateCarree(),
        color="#14384d",
        scale=720,
        width=0.0022,
        alpha=0.46,
        zorder=5,
    )
    add_corridor(ax, wide=True)
    add_pathway_cues(ax)
    add_source_regions(ax)
    add_feature_labels(ax)
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
    add_wind_speed_key(ax)

    fig.text(
        0.04,
        0.020,
        "January 300 hPa winds. NOAA NCEP/NCAR Reanalysis 1 monthly climatology, 1996-2006.",
        fontsize=10.2,
        color=COLORS["muted"],
    )
    fig.savefig(OUT_DIR / "hero.png", dpi=180)
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
        (axes[0], p, "Pacific zonal wind at 300 hPa", COLORS["blue"], "Upper-air version of the 35N, 140E-240E P-WIND corridor", (-0.55, -4.0)),
        (axes[1], nw, "Japan northwesterly component at 850 hPa", COLORS["green"], "45-degree northwest/southeast projection over 30-45N, 130-145E", (-0.30, -1.2)),
    ]

    for ax, values, title, color, subtitle, peak_offset in specs:
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
        annotate_peak(ax, x, values, color, offset=peak_offset)
        subtitle_text.set_path_effects([pe.withStroke(linewidth=2.4, foreground=COLORS["panel"])])

    axes[-1].set_xticks(x, MONTHS)
    axes[-1].set_xlabel("month")
    fig.suptitle("The wind indices peak in cool months", x=0.08, y=0.965, ha="left", fontsize=17, fontweight="bold", color=COLORS["ink"])
    save(fig, "seasonal-wind-indices.png")


def image_monthly_animation(ds: xr.Dataset) -> None:
    clim = monthly_climatology(ds, 300)
    speed = np.hypot(clim["u"], clim["v"])
    levels = np.arange(6, 58, 4)
    p = p_wind(ds)
    lon = clim["lon"].to_numpy()
    lat = clim["lat"].to_numpy()
    if lat[0] > lat[-1]:
        lat_plot = lat[::-1]
        lat_slice = slice(None, None, -1)
    else:
        lat_plot = lat
        lat_slice = slice(None)

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree(central_longitude=180))

    def draw(month_index: int):
        ax.clear()
        add_pacific_context(ax, grid_labels=False)
        ax.set_aspect("auto")
        month = month_index + 1
        u = clim["u"].sel(month=month)
        v = clim["v"].sel(month=month)
        spd = speed.sel(month=month)
        u_arr = u.to_numpy()[lat_slice, :]
        v_arr = v.to_numpy()[lat_slice, :]
        spd_arr = spd.to_numpy()[lat_slice, :]
        line_width = 0.45 + 1.40 * np.clip((spd_arr - 8) / 45, 0, 1)
        ax.contourf(
            clim["lon"],
            clim["lat"],
            spd,
            levels=levels,
            cmap=WIND_CMAP,
            extend="max",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.streamplot(
            lon,
            lat_plot,
            u_arr,
            v_arr,
            density=(1.25, 0.82),
            linewidth=line_width,
            arrowsize=1.35,
            arrowstyle="->",
            color="#14384d",
            transform=ccrs.PlateCarree(),
            broken_streamlines=False,
            zorder=5,
        )
        add_corridor(ax, wide=False)
        add_feature_labels(ax, include_north_pacific=False)
        add_locations(ax)

        title_box = Rectangle(
            (0.018, 0.802),
            0.305,
            0.172,
            transform=ax.transAxes,
            facecolor=COLORS["panel"],
            edgecolor="#d7ddd9",
            linewidth=0.8,
            alpha=0.88,
            zorder=5,
        )
        ax.add_patch(title_box)
        month_text = ax.text(
            0.035,
            0.935,
            MONTHS[month_index],
            transform=ax.transAxes,
            fontsize=21.5,
            weight="bold",
            color=COLORS["ink"],
            zorder=8,
        )
        ax.text(
            0.035,
            0.885,
            f"P-WIND {float(p.sel(month=month)):.1f} m/s",
            transform=ax.transAxes,
            fontsize=11.0,
            color=COLORS["deep_blue"],
            weight="bold",
            zorder=8,
        )
        ax.text(
            0.035,
            0.852,
            "same map + scale; month changes",
            transform=ax.transAxes,
            fontsize=8.5,
            color=COLORS["muted"],
            zorder=8,
        )
        ax.text(
            0.035,
            0.825,
            "300 hPa monthly climatology, 1996-2006",
            transform=ax.transAxes,
            fontsize=8.1,
            color=COLORS["muted"],
            zorder=8,
        )
        add_wind_speed_key(ax)

    ani = animation.FuncAnimation(fig, draw, frames=12, interval=520, repeat=True)
    ani.save(OUT_DIR / "monthly-pacific-winds.gif", writer=animation.PillowWriter(fps=2), dpi=100)
    plt.close(fig)


def image_analysis_boundary() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(0.05, 0.93, "Evidence boundary", fontsize=18, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.875,
        "What the data show, what I think is worth testing, and what is not shown yet.",
        fontsize=10.9,
        color=COLORS["muted"],
    )

    columns = [
        ("Shown", COLORS["green"], "Case timing in the paper and NOAA wind fields have a recurring seasonal structure."),
        ("Worth testing", COLORS["gold"], "A winter air-mass pathway links continental Asia, Japan, Hawaii, and southern California often enough to chase."),
        ("Not shown", COLORS["red"], "The causal agent is not identified. A wind map is not proof that wind causes Kawasaki disease."),
    ]

    x0 = 0.055
    y0 = 0.28
    width = 0.285
    gap = 0.027
    height = 0.48
    for i, (title, color, body) in enumerate(columns):
        x = x0 + i * (width + gap)
        ax.add_patch(Rectangle((x, y0), width, height, facecolor=COLORS["panel"], edgecolor="#d7ddd9", linewidth=1.1))
        ax.add_patch(Rectangle((x, y0 + height - 0.075), width, 0.075, facecolor=color, edgecolor="none", alpha=0.94))
        ax.text(x + 0.025, y0 + height - 0.047, title, fontsize=13.8, fontweight="bold", color="white", va="center")
        ax.text(x + 0.025, y0 + 0.32, textwrap.fill(body, width=28), fontsize=11.8, color=COLORS["ink"], va="top", linespacing=1.32)

    ax.plot([0.678, 0.678], [0.22, 0.80], color=COLORS["red"], linewidth=2.3, linestyle=(0, (4, 5)), alpha=0.8)
    ax.text(
        0.695,
        0.205,
        "this is the line I would not cross without a stronger study",
        fontsize=9.8,
        color=COLORS["red"],
        weight="bold",
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
