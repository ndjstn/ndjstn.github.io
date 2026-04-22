from __future__ import annotations

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
from matplotlib.patches import Ellipse, Polygon, Rectangle
from scipy.interpolate import RegularGridInterpolator

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
    "Siberia": (126, 55.4),
    "Mongolia /\nGobi Desert": (106.5, 43.0),
    "North China\nPlain": (116.0, 36.2),
}

FEATURE_LABELS = {
    "Sea of Japan": (136.5, 41.0),
    "Kuril Islands": (151.5, 46.0),
    "Aleutian arc": (181, 52.7),
    "Bering Sea": (188, 58.0),
    "Gulf of Alaska": (219, 55.2),
    "North Pacific": (184, 24.0),
}

INSPECTION_REGIONS = [
    ("Mongolia / Gobi", 106.5, 43.0, 13.5, 7.4),
    ("North China Plain", 116.0, 36.2, 11.0, 6.4),
    ("Siberia", 126.0, 55.4, 23.0, 6.6),
]


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
    ax.set_extent([100, 255, 15, 62], crs=ccrs.PlateCarree())
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


def add_source_context(ax) -> None:
    ax.set_extent([82, 143, 33, 51], crs=ccrs.PlateCarree())
    ax.set_facecolor(COLORS["water"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=COLORS["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, color=COLORS["coast"], zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, color="#9b9a90", alpha=0.75, zorder=4)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.55,
        color="#9fb0ac",
        alpha=0.32,
        linestyle="-",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8.0, "color": COLORS["muted"]}
    gl.ylabel_style = {"size": 8.0, "color": COLORS["muted"]}


def add_inspection_regions(ax, *, labels: bool = True, alpha: float = 0.18) -> None:
    for label, lon, lat, width, height in INSPECTION_REGIONS:
        ax.add_patch(
            Ellipse(
                (lon, lat),
                width,
                height,
                transform=ccrs.PlateCarree(),
                facecolor=COLORS["gold"],
                edgecolor=COLORS["gold"],
                linewidth=1.2,
                alpha=alpha,
                zorder=3,
            )
        )
        if not labels:
            continue
        text = ax.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=8.1,
            color="#5e5135",
            weight="bold",
            ha="center",
            va="center",
            zorder=8,
        )
        stroke_text(text, lw=2.7)


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


def add_wind_speed_key(ax, *, motion_label: str = "arrows = direction; red = analysis corridor") -> None:
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
    ax.text(0.724, 0.041, motion_label, transform=ax.transAxes, fontsize=7.1, color=COLORS["muted"], zorder=9)


def tracer_seeds() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    lon_parts = []
    lat_parts = []
    kind_parts = []

    source_specs = [
        (106.5, 43.0, 5.0, 2.2, 36),
        (116.0, 36.2, 4.8, 2.0, 36),
        (126.0, 55.4, 7.0, 2.0, 30),
    ]
    for lon0, lat0, lon_sd, lat_sd, count in source_specs:
        lon_parts.append(rng.normal(lon0, lon_sd, count))
        lat_parts.append(rng.normal(lat0, lat_sd, count))
        kind_parts.append(np.full(count, 0))

    count = 120
    lon_parts.append(rng.uniform(138, 236, count))
    lat_parts.append(rng.normal(35.0, 3.2, count))
    kind_parts.append(np.full(count, 1))

    count = 48
    lon_parts.append(rng.uniform(150, 226, count))
    lat_parts.append(rng.normal(51.5, 2.0, count))
    kind_parts.append(np.full(count, 2))

    lon = np.concatenate(lon_parts)
    lat = np.concatenate(lat_parts)
    kind = np.concatenate(kind_parts)
    offsets = rng.uniform(0, 1, lon.size)
    return lon, lat, kind, offsets


def wind_interpolators(clim: xr.Dataset, month: int) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    u = clim["u"].sel(month=month)
    v = clim["v"].sel(month=month)
    lat = u["lat"].to_numpy()
    lon = u["lon"].to_numpy()
    u_arr = u.to_numpy()
    v_arr = v.to_numpy()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u_arr = u_arr[::-1, :]
        v_arr = v_arr[::-1, :]
    kwargs = {"bounds_error": False, "fill_value": np.nan}
    return RegularGridInterpolator((lat, lon), u_arr, **kwargs), RegularGridInterpolator((lat, lon), v_arr, **kwargs)


def tracer_positions(
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
    seed_lon: np.ndarray,
    seed_lat: np.ndarray,
    seed_offsets: np.ndarray,
    frame_phase: int,
    subframes: int,
    *,
    lag: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = ((frame_phase + seed_offsets * subframes - lag) % subframes) / subframes
    points = np.column_stack([seed_lat, seed_lon])
    u = u_interp(points)
    v = v_interp(points)
    cos_lat = np.maximum(np.cos(np.deg2rad(seed_lat)), 0.42)
    lon = seed_lon + (u * 0.30 * phase / cos_lat)
    lat = seed_lat + (v * 0.19 * phase)
    valid = np.isfinite(lon) & np.isfinite(lat) & (lon >= 100) & (lon <= 255) & (lat >= 15) & (lat <= 62)
    return lon, lat, valid


def add_tracer_layer(
    ax,
    clim: xr.Dataset,
    month: int,
    frame_phase: int,
    subframes: int,
    seed_lon: np.ndarray,
    seed_lat: np.ndarray,
    seed_kind: np.ndarray,
    seed_offsets: np.ndarray,
) -> None:
    u_interp, v_interp = wind_interpolators(clim, month)
    styles = {
        0: {"color": "#ffd969", "edge": "#6b5421", "size": 20},
        1: {"color": "#f8fbff", "edge": "#174a68", "size": 15},
        2: {"color": "#d9a73a", "edge": "#5d4520", "size": 16},
    }
    for lag, alpha, shrink in [(2.3, 0.15, 0.58), (1.2, 0.30, 0.78), (0.0, 0.86, 1.0)]:
        lon, lat, valid = tracer_positions(u_interp, v_interp, seed_lon, seed_lat, seed_offsets, frame_phase, subframes, lag=lag)
        for kind, style in styles.items():
            mask = valid & (seed_kind == kind)
            if not np.any(mask):
                continue
            ax.scatter(
                lon[mask],
                lat[mask],
                s=style["size"] * shrink,
                color=style["color"],
                edgecolor=style["edge"],
                linewidth=0.25,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )


def image_source_screening_map() -> None:
    fig = plt.figure(figsize=(13.2, 10.8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_axes([0.040, 0.345, 0.920, 0.540], projection=ccrs.PlateCarree())
    list_ax = fig.add_axes([0.040, 0.060, 0.920, 0.245])
    add_source_context(ax)
    ax.set_aspect("auto")
    list_ax.axis("off")
    list_ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=list_ax.transAxes,
            facecolor=COLORS["panel"],
            edgecolor=COLORS["grid"],
            linewidth=1.0,
            zorder=0,
        )
    )

    swath = np.array(
        [
            [88, 43.6],
            [99, 46.2],
            [113, 46.4],
            [126, 44.8],
            [137, 40.5],
            [143, 37.2],
            [141, 34.6],
            [130, 37.0],
            [117, 38.5],
            [104, 39.5],
            [92, 39.6],
            [88, 41.0],
        ]
    )
    ax.add_patch(
        Polygon(
            swath,
            closed=True,
            transform=ccrs.PlateCarree(),
            facecolor="#d39b2d",
            edgecolor="#9c6d12",
            linewidth=1.6,
            alpha=0.24,
            zorder=2,
        )
    )

    core = np.array(
        [
            [94, 42.2],
            [106, 42.4],
            [119, 41.2],
            [132, 38.7],
            [141.5, 36.0],
            [140.5, 34.8],
            [130, 36.6],
            [117, 38.4],
            [104, 40.0],
            [94, 40.0],
        ]
    )
    ax.add_patch(
        Polygon(
            core,
            closed=True,
            transform=ccrs.PlateCarree(),
            facecolor="#c55e43",
            edgecolor="none",
            alpha=0.18,
            zorder=3,
        )
    )

    sites = [
        ("1", "Dunhuang", "Taklamakan edge", 94.66, 40.14, -7.0, 1.6, COLORS["gold"]),
        ("2", "Dalanzadgad", "Gobi Desert", 104.43, 43.57, -7.6, 2.4, COLORS["gold"]),
        ("3", "Lanzhou", "Loess Plateau", 103.84, 36.06, -6.8, -2.1, COLORS["gold"]),
        ("4", "Hohhot", "steppe edge", 111.75, 40.84, -5.5, -2.4, COLORS["gold"]),
        ("5", "Beijing", "North China Plain", 116.40, 39.90, 1.2, 2.7, COLORS["green"]),
        ("6", "Shenyang", "NE China", 123.43, 41.80, 1.2, 2.5, COLORS["green"]),
        ("7", "Changchun", "crop soils", 125.32, 43.82, 1.2, 2.3, COLORS["green"]),
        ("8", "Harbin", "northern branch", 126.64, 45.76, 1.1, 2.1, COLORS["green"]),
        ("9", "Seoul", "downwind filter", 126.98, 37.57, 1.4, -2.4, COLORS["blue"]),
        ("10", "Tokyo", "receptor check", 139.69, 35.69, -7.2, -2.0, COLORS["red"]),
    ]
    for num, city, note, lon, lat, dx, dy, color in sites:
        ax.scatter(lon, lat, s=132, marker="o", color=color, edgecolor="white", linewidth=1.25, transform=ccrs.PlateCarree(), zorder=9)
        num_text = ax.text(lon, lat, num, fontsize=9.4, weight="bold", color="white", ha="center", va="center", transform=ccrs.PlateCarree(), zorder=10)
        stroke_text(num_text, lw=1.1)
        label = ax.annotate(
            city,
            xy=(lon, lat),
            xytext=(lon + dx, lat + dy),
            xycoords=ccrs.PlateCarree(),
            textcoords=ccrs.PlateCarree(),
            fontsize=12.2,
            color=COLORS["ink"],
            weight="bold",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.1, "alpha": 0.85},
            zorder=11,
        )
        stroke_text(label, lw=2.6)

    ax.text(
        0.035,
        0.045,
        "Broad winter pathway to test; pins are concrete places, not claimed sources.",
        transform=ax.transAxes,
        fontsize=11.6,
        color=COLORS["ink"],
        weight="bold",
        zorder=12,
        bbox={"boxstyle": "square,pad=0.42", "facecolor": COLORS["panel"], "edgecolor": COLORS["grid"], "alpha": 0.92},
    )

    for start, end in [
        ((94.7, 40.2), (104.4, 43.6)),
        ((104.4, 43.6), (116.4, 39.9)),
        ((116.4, 39.9), (126.0, 41.5)),
        ((126.0, 41.5), (139.7, 35.7)),
    ]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ccrs.PlateCarree(),
            textcoords=ccrs.PlateCarree(),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["deep_blue"], "lw": 2.1, "mutation_scale": 15, "alpha": 0.86},
            zorder=8,
        )
    ax.annotate(
        "",
        xy=(139.7, 35.7),
        xytext=(126.64, 45.76),
        xycoords=ccrs.PlateCarree(),
        textcoords=ccrs.PlateCarree(),
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["gold"],
            "lw": 2.0,
            "linestyle": (0, (4, 4)),
            "mutation_scale": 15,
            "alpha": 0.90,
        },
        zorder=8,
    )

    list_ax.text(0.025, 0.835, "Look here first", transform=list_ax.transAxes, fontsize=19.5, weight="bold", color=COLORS["ink"], zorder=2)
    row_y = [0.825, 0.645, 0.465, 0.285, 0.105]
    for i, (num, city, _note, _lon, _lat, _dx, _dy, color) in enumerate(sites):
        col_x = 0.350 if i < 5 else 0.665
        y = row_y[i % 5]
        list_ax.scatter(col_x, y, s=142, color=color, edgecolor="white", linewidth=1.1, transform=list_ax.transAxes, zorder=3)
        list_ax.text(col_x, y, num, transform=list_ax.transAxes, fontsize=9.0, color="white", weight="bold", ha="center", va="center", zorder=4)
        list_ax.text(col_x + 0.035, y, city, transform=list_ax.transAxes, fontsize=15.2, color=COLORS["ink"], weight="bold", ha="left", va="center", zorder=2)
    list_ax.text(
        0.025,
        0.430,
        "Sample:\ndust, soils, crop residue,\nfungi/yeasts, toxins,\nfine aerosols.",
        transform=list_ax.transAxes,
        fontsize=10.6,
        color=COLORS["ink"],
        weight="bold",
        linespacing=1.25,
        wrap=True,
        zorder=2,
    )

    fig.text(0.040, 0.965, "Where I would look first", fontsize=24, weight="bold", color=COLORS["ink"])
    fig.text(
        0.040,
        0.928,
        "A source-screening map, not an origin claim: use the wind pathway to choose real places for sampling.",
        fontsize=12.3,
        color=COLORS["muted"],
    )
    fig.savefig(OUT_DIR / "source-screening-swath.png", dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


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
    add_inspection_regions(ax, labels=False, alpha=0.16)
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
    seed_lon, seed_lat, seed_kind, seed_offsets = tracer_seeds()
    subframes = 4

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree(central_longitude=180))

    def draw(frame_index: int):
        ax.clear()
        add_pacific_context(ax, grid_labels=False)
        ax.set_aspect("auto")
        month_index = frame_index // subframes
        frame_phase = frame_index % subframes
        month = month_index + 1
        spd = speed.sel(month=month)
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
        add_inspection_regions(ax, labels=True, alpha=0.20)
        add_corridor(ax, wide=False)
        add_pathway_cues(ax)
        add_feature_labels(ax, include_north_pacific=False)
        add_locations(ax)
        add_tracer_layer(ax, clim, month, frame_phase, subframes, seed_lon, seed_lat, seed_kind, seed_offsets)

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
            "moving dots follow monthly wind",
            transform=ax.transAxes,
            fontsize=8.5,
            color=COLORS["muted"],
            zorder=8,
        )
        ax.text(
            0.035,
            0.825,
            "gold = upstream areas to inspect",
            transform=ax.transAxes,
            fontsize=8.1,
            color=COLORS["muted"],
            zorder=8,
        )
        add_wind_speed_key(ax, motion_label="dots = wind motion; red = analysis corridor")

    ani = animation.FuncAnimation(fig, draw, frames=12 * subframes, interval=130, repeat=True)
    ani.save(OUT_DIR / "wind-pathway-particles.gif", writer=animation.PillowWriter(fps=8), dpi=92)
    plt.close(fig)


def image_analysis_boundary() -> None:
    fig = plt.figure(figsize=(13.4, 7.3))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.86], projection=ccrs.PlateCarree(central_longitude=180))
    add_pacific_context(ax, grid_labels=False)
    ax.set_aspect("auto")
    ax.set_title("What this map actually narrows", loc="left", pad=12, fontsize=18, fontweight="bold", color=COLORS["ink"])

    add_inspection_regions(ax, labels=True, alpha=0.24)
    add_corridor(ax, label=True, wide=True)
    add_pathway_cues(ax)
    add_feature_labels(ax)
    add_locations(ax)

    notes = [
        ("shown", "seasonal timing + wind alignment", 0.035, 0.80, COLORS["green"]),
        ("look here", "upstream land, dust, biology,\nand the winter route", 0.035, 0.69, COLORS["gold"]),
        ("not shown", "the causal agent", 0.735, 0.79, COLORS["red"]),
    ]
    for header, body, x, y, color in notes:
        ax.add_patch(
            Rectangle(
                (x - 0.012, y - 0.018),
                0.235,
                0.095,
                transform=ax.transAxes,
                facecolor=COLORS["panel"],
                edgecolor=color,
                linewidth=1.2,
                alpha=0.92,
                zorder=11,
            )
        )
        ax.text(x, y + 0.040, header, transform=ax.transAxes, fontsize=9.5, color=color, weight="bold", zorder=12)
        ax.text(x, y + 0.004, body, transform=ax.transAxes, fontsize=8.2, color=COLORS["ink"], linespacing=1.22, zorder=12)

    boundary = ax.text(
        0.735,
        0.705,
        "A pathway is a search area.\nIt is not proof.",
        transform=ax.transAxes,
        fontsize=12.0,
        color=COLORS["red"],
        weight="bold",
        zorder=12,
    )
    stroke_text(boundary, lw=2.8)
    ax.text(
        0.03,
        0.035,
        "The useful output is geographic: where to sample, which route to test, and what negative controls should fail.",
        transform=ax.transAxes,
        fontsize=9.0,
        color=COLORS["muted"],
        zorder=12,
    )

    fig.savefig(OUT_DIR / "research-boundary-map.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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
    image_source_screening_map()
    image_monthly_animation(ds)
    image_analysis_boundary()
    print_summary(ds)


if __name__ == "__main__":
    main()
