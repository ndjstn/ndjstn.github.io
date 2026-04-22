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

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
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
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor=COLORS["land"],
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.7,
        color=COLORS["coast"],
        zorder=4,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.35,
        color="#9b9a90",
        alpha=0.7,
        zorder=4,
    )
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
    ax.set_extent([74, 143, 33, 55], crs=ccrs.PlateCarree())
    ax.set_facecolor(COLORS["water"])
    relief = ax.stock_img()
    relief.set_alpha(0.46)
    relief.set_zorder(0)
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor=COLORS["land"],
        edgecolor="none",
        alpha=0.56,
        zorder=1,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.75,
        color=COLORS["coast"],
        zorder=6,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.42,
        color="#8f9188",
        alpha=0.75,
        zorder=6,
    )
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
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ccrs.PlateCarree(),
            textcoords=ccrs.PlateCarree(),
            arrowprops=arrow_style,
            zorder=8,
        )

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
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["gold"],
            "lw": 1.8,
            "mutation_scale": 15,
        },
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


def add_wind_speed_key(
    ax, *, motion_label: str = "arrows = direction; red = analysis corridor"
) -> None:
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
    ax.text(
        0.724,
        0.122,
        "wind speed (m/s)",
        transform=ax.transAxes,
        fontsize=8.4,
        color=COLORS["ink"],
        weight="bold",
        zorder=9,
    )
    swatches = ["#f3f6c8", "#b9dfbf", "#69c7c4", "#2389bd", "#182f7c"]
    values = ["10", "20", "30", "40", "50+"]
    x0 = 0.724
    w = 0.043
    for i, (color, value) in enumerate(zip(swatches, values, strict=False)):
        x = x0 + i * w
        ax.add_patch(
            Rectangle(
                (x, 0.086),
                w,
                0.018,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=9,
            )
        )
        ax.text(
            x + w / 2,
            0.061,
            value,
            transform=ax.transAxes,
            fontsize=7.1,
            color=COLORS["muted"],
            ha="center",
            zorder=9,
        )
    ax.text(
        0.724,
        0.041,
        motion_label,
        transform=ax.transAxes,
        fontsize=7.1,
        color=COLORS["muted"],
        zorder=9,
    )


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


def wind_interpolators(
    clim: xr.Dataset, month: int
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
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
    return RegularGridInterpolator(
        (lat, lon), u_arr, **kwargs
    ), RegularGridInterpolator((lat, lon), v_arr, **kwargs)


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
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= 100)
        & (lon <= 255)
        & (lat >= 15)
        & (lat <= 62)
    )
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
        lon, lat, valid = tracer_positions(
            u_interp,
            v_interp,
            seed_lon,
            seed_lat,
            seed_offsets,
            frame_phase,
            subframes,
            lag=lag,
        )
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
            [76.5, 43.1],
            [84, 45.8],
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
            [81, 40.3],
            [76.5, 41.8],
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

    river_color = "#4f8fa8"
    lake_fill = "#c8e4ed"
    lake_edge = "#6b9caf"
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        linewidth=0.90,
        color=river_color,
        alpha=0.80,
        zorder=6,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor=lake_fill,
        edgecolor=lake_edge,
        linewidth=0.60,
        alpha=0.88,
        zorder=6,
    )

    waterway_labels = [
        ("Lake Balkhash", 76.9, 46.9),
        ("Lake Baikal", 108.4, 53.6),
        ("Yellow River", 109.0, 38.1),
        ("Songhua River", 126.3, 45.1),
        ("Amur River", 128.4, 50.8),
    ]
    for label, lon, lat in waterway_labels:
        text = ax.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=9.5,
            color="#346f89",
            style="italic",
            ha="center",
            va="center",
            zorder=7,
        )
        stroke_text(text, lw=2.3)

    terrain_labels = [
        ("Tian Shan", 78.2, 41.8),
        ("Altai", 88.5, 48.4),
        ("Gobi", 103.0, 43.1),
        ("Loess Plateau", 106.5, 36.4),
    ]
    for label, lon, lat in terrain_labels:
        text = ax.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=9.8,
            color="#7b6842",
            weight="bold",
            alpha=0.82,
            ha="center",
            va="center",
            zorder=7,
        )
        stroke_text(text, lw=2.4)

    context_cities = [
        ("Almaty", 76.89, 43.24, 1.1, -1.1),
        ("Oskemen", 82.61, 49.95, 1.0, 0.6),
        ("Irkutsk", 104.30, 52.29, 1.0, 0.7),
        ("Ulaanbaatar", 106.91, 47.92, 1.0, 0.7),
        ("Sainshand", 110.14, 44.89, 1.0, -1.0),
        ("Chita", 113.50, 52.03, 1.0, 0.6),
        ("Choibalsan", 114.54, 48.08, 1.1, 0.5),
        ("Blagoveshchensk", 127.53, 50.27, -9.6, 0.7),
        ("Khabarovsk", 135.08, 48.48, -8.0, -0.9),
        ("Vladivostok", 131.89, 43.12, -8.1, -1.1),
    ]
    for city, lon, lat, dx, dy in context_cities:
        ax.scatter(
            lon,
            lat,
            s=30,
            marker="s",
            color="#59645f",
            edgecolor="white",
            linewidth=0.7,
            transform=ccrs.PlateCarree(),
            zorder=8,
        )
        text = ax.text(
            lon + dx,
            lat + dy,
            city,
            transform=ccrs.PlateCarree(),
            fontsize=8.8,
            color="#424b47",
            weight="bold",
            zorder=8,
        )
        stroke_text(text, lw=2.2)

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
    for num, city, _note, lon, lat, dx, dy, color in sites:
        ax.scatter(
            lon,
            lat,
            s=132,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=1.25,
            transform=ccrs.PlateCarree(),
            zorder=9,
        )
        num_text = ax.text(
            lon,
            lat,
            num,
            fontsize=9.4,
            weight="bold",
            color="white",
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
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
        bbox={
            "boxstyle": "square,pad=0.42",
            "facecolor": COLORS["panel"],
            "edgecolor": COLORS["grid"],
            "alpha": 0.92,
        },
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
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["deep_blue"],
                "lw": 2.1,
                "mutation_scale": 15,
                "alpha": 0.86,
            },
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

    list_ax.text(
        0.025,
        0.890,
        "Look here first",
        transform=list_ax.transAxes,
        fontsize=19.5,
        weight="bold",
        color=COLORS["ink"],
        va="top",
        zorder=2,
    )
    row_y = [0.825, 0.645, 0.465, 0.285, 0.105]
    for i, (num, city, _note, _lon, _lat, _dx, _dy, color) in enumerate(sites):
        col_x = 0.350 if i < 5 else 0.665
        y = row_y[i % 5]
        list_ax.scatter(
            col_x,
            y,
            s=142,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            transform=list_ax.transAxes,
            zorder=3,
        )
        list_ax.text(
            col_x,
            y,
            num,
            transform=list_ax.transAxes,
            fontsize=9.0,
            color="white",
            weight="bold",
            ha="center",
            va="center",
            zorder=4,
        )
        list_ax.text(
            col_x + 0.035,
            y,
            city,
            transform=list_ax.transAxes,
            fontsize=15.2,
            color=COLORS["ink"],
            weight="bold",
            ha="left",
            va="center",
            zorder=2,
        )
    list_ax.text(
        0.025,
        0.650,
        "Layers:\n1-10 = source pins\ngray = cities\nblue = water\nrelief = terrain",
        transform=list_ax.transAxes,
        fontsize=10.6,
        color=COLORS["ink"],
        weight="bold",
        linespacing=1.18,
        va="top",
        wrap=True,
        zorder=2,
    )

    fig.text(
        0.040,
        0.965,
        "Where I would look first",
        fontsize=24,
        weight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.040,
        0.928,
        "A source-screening map, not an origin claim: terrain, rivers, lakes, and cities help decide where to sample.",
        fontsize=12.3,
        color=COLORS["muted"],
    )
    fig.savefig(
        OUT_DIR / "source-screening-swath.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
    )
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
    return (
        component.weighted(weights)
        .mean(("lat", "lon"))
        .groupby("time.month")
        .mean("time")
    )


def image_hero(ds: xr.Dataset) -> None:
    wind = jan_300_wind(ds)
    speed = np.hypot(wind["u"], wind["v"])
    q_lon = wind["lon"].to_numpy()[::7]
    q_lat = wind["lat"].to_numpy()[::4]
    q_x, q_y = np.meshgrid(q_lon, q_lat)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes(
        [0.025, 0.060, 0.95, 0.895], projection=ccrs.PlateCarree(central_longitude=180)
    )
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


def annotate_peak(
    ax, x: np.ndarray, values: xr.DataArray, color: str, *, offset: tuple[float, float]
) -> None:
    arr = values.to_numpy()
    idx = int(np.argmax(arr))
    month = MONTHS[idx]
    value = float(arr[idx])
    ax.scatter(
        x[idx], value, s=64, color=color, edgecolor="white", linewidth=0.9, zorder=4
    )
    ann = ax.annotate(
        f"{month}\n{value:.1f} m/s",
        xy=(x[idx], value),
        xytext=(x[idx] + offset[0], value + offset[1]),
        color=color,
        fontsize=10.8,
        weight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 1.5,
            "shrinkA": 0,
            "shrinkB": 0,
        },
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
        (
            axes[0],
            p,
            "Pacific zonal wind at 300 hPa",
            COLORS["blue"],
            "Upper-air version of the 35N, 140E-240E P-WIND corridor",
            (-0.55, -4.0),
        ),
        (
            axes[1],
            nw,
            "Japan northwesterly component at 850 hPa",
            COLORS["green"],
            "45-degree northwest/southeast projection over 30-45N, 130-145E",
            (-0.30, -1.2),
        ),
    ]

    for ax, values, title, color, subtitle, peak_offset in specs:
        ax.axvspan(1, 3, color="#dbe9ef", alpha=0.7, zorder=0)
        ax.axvspan(11, 12, color="#dbe9ef", alpha=0.7, zorder=0)
        ax.plot(x, values, color=color, linewidth=3.0)
        ax.scatter(
            x, values, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3
        )
        ax.set_title(title, loc="left", pad=15, fontsize=13.8, fontweight="bold")
        subtitle_text = ax.text(
            0.0,
            0.99,
            subtitle,
            transform=ax.transAxes,
            color=COLORS["muted"],
            fontsize=9.8,
        )
        ax.set_ylabel("m/s")
        ax.grid(axis="y", alpha=0.45)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymin = float(np.floor(values.min() - 1.0))
        ymax = float(np.ceil(values.max() + 1.4))
        ax.set_ylim(ymin, ymax)
        annotate_peak(ax, x, values, color, offset=peak_offset)
        subtitle_text.set_path_effects(
            [pe.withStroke(linewidth=2.4, foreground=COLORS["panel"])]
        )

    axes[-1].set_xticks(x, MONTHS)
    axes[-1].set_xlabel("month")
    fig.suptitle(
        "The wind indices peak in cool months",
        x=0.08,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )
    save(fig, "seasonal-wind-indices.png")


def image_monthly_animation(ds: xr.Dataset) -> None:
    clim = monthly_climatology(ds, 300)
    speed = np.hypot(clim["u"], clim["v"])
    levels = np.arange(6, 58, 4)
    p = p_wind(ds)
    seed_lon, seed_lat, seed_kind, seed_offsets = tracer_seeds()
    subframes = 4

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes(
        [0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree(central_longitude=180)
    )

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
        add_tracer_layer(
            ax,
            clim,
            month,
            frame_phase,
            subframes,
            seed_lon,
            seed_lat,
            seed_kind,
            seed_offsets,
        )

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
        ax.text(
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
        add_wind_speed_key(
            ax, motion_label="dots = wind motion; red = analysis corridor"
        )

    ani = animation.FuncAnimation(
        fig, draw, frames=12 * subframes, interval=130, repeat=True
    )
    ani.save(
        OUT_DIR / "wind-pathway-particles.gif",
        writer=animation.PillowWriter(fps=8),
        dpi=92,
    )
    plt.close(fig)


def image_analysis_boundary() -> None:
    fig = plt.figure(figsize=(13.4, 7.3))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_axes(
        [0.02, 0.06, 0.96, 0.86], projection=ccrs.PlateCarree(central_longitude=180)
    )
    add_pacific_context(ax, grid_labels=False)
    ax.set_aspect("auto")
    ax.set_title(
        "What this map actually narrows",
        loc="left",
        pad=12,
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )

    add_inspection_regions(ax, labels=True, alpha=0.24)
    add_corridor(ax, label=True, wide=True)
    add_pathway_cues(ax)
    add_feature_labels(ax)
    add_locations(ax)

    notes = [
        ("shown", "seasonal timing + wind alignment", 0.035, 0.80, COLORS["green"]),
        (
            "look here",
            "upstream land, dust, biology,\nand the winter route",
            0.035,
            0.69,
            COLORS["gold"],
        ),
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
        ax.text(
            x,
            y + 0.040,
            header,
            transform=ax.transAxes,
            fontsize=9.5,
            color=color,
            weight="bold",
            zorder=12,
        )
        ax.text(
            x,
            y + 0.004,
            body,
            transform=ax.transAxes,
            fontsize=8.2,
            color=COLORS["ink"],
            linespacing=1.22,
            zorder=12,
        )

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

    fig.savefig(
        OUT_DIR / "research-boundary-map.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def image_emergence_context() -> None:
    """Historical land-use change in the source zone (1950-1965) and KD emergence.

    Three schematic overlays — NE China industrialisation, Virgin Lands Campaign,
    Great Leap Forward deforestation — against the wind source zone, with a
    timeline strip linking those events to Japan KD case counts.
    KD numbers are approximate schematic values from published Japan biennial surveys.
    """
    C_IND = "#7a2a2a"  # industrial dark red
    C_VL = "#c8a050"  # sandy — Virgin Lands
    C_GLF = "#4a7040"  # muted forest green — GLF deforestation
    RIVER = "#4f8fa8"
    LAKE_F = "#c8e4ed"
    LAKE_E = "#6b9caf"

    fig = plt.figure(figsize=(14.5, 10.2))
    fig.patch.set_facecolor(COLORS["bg"])

    # ── main map ──────────────────────────────────────────────────────────────
    ax = fig.add_axes([0.030, 0.285, 0.940, 0.670], projection=ccrs.PlateCarree())
    ax.set_extent([57, 145, 29, 62], crs=ccrs.PlateCarree())
    ax.set_facecolor(COLORS["water"])
    ax.set_aspect("auto")

    relief = ax.stock_img()
    relief.set_alpha(0.28)
    relief.set_zorder(0)
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor=COLORS["land"],
        edgecolor="none",
        alpha=0.55,
        zorder=1,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.65,
        color=COLORS["coast"],
        zorder=7,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.28,
        color="#9b9a90",
        alpha=0.60,
        zorder=7,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        linewidth=0.75,
        color=RIVER,
        alpha=0.72,
        zorder=5,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor=LAKE_F,
        edgecolor=LAKE_E,
        linewidth=0.50,
        alpha=0.85,
        zorder=5,
    )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="#9fb0ac",
        alpha=0.28,
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 8.0, "color": COLORS["muted"]}
    gl.ylabel_style = {"size": 8.0, "color": COLORS["muted"]}

    # ── overlay 1: Virgin Lands Campaign (1953-1965) ──────────────────────────
    vl_poly = np.array(
        [
            [58, 50.0],
            [63, 51.5],
            [69, 52.5],
            [76, 53.2],
            [82, 53.8],
            [86, 54.0],
            [86, 57.5],
            [80, 57.2],
            [72, 56.5],
            [63, 55.2],
            [58, 53.8],
            [58, 50.0],
        ]
    )
    ax.add_patch(
        Polygon(
            vl_poly,
            closed=True,
            transform=ccrs.PlateCarree(),
            facecolor=C_VL,
            edgecolor="#8b6914",
            linewidth=0.9,
            alpha=0.28,
            zorder=2,
            hatch="///",
        )
    )
    vt = ax.text(
        71.5,
        55.5,
        "Virgin Lands Campaign\n40M ha plowed 1953-65",
        transform=ccrs.PlateCarree(),
        fontsize=9.0,
        color="#6b4a08",
        weight="bold",
        ha="center",
        zorder=9,
    )
    stroke_text(vt, 2.2)

    # ── overlay 2: GLF deforestation zone in Manchuria (1957-1964) ───────────
    glf_poly = np.array(
        [
            [118, 46.0],
            [126, 46.5],
            [131, 46.2],
            [133, 48.0],
            [131, 51.2],
            [127, 52.5],
            [122, 54.0],
            [118, 54.0],
            [118, 46.0],
        ]
    )
    ax.add_patch(
        Polygon(
            glf_poly,
            closed=True,
            transform=ccrs.PlateCarree(),
            facecolor=C_GLF,
            edgecolor="#2d5028",
            linewidth=0.9,
            alpha=0.25,
            zorder=3,
            hatch="\\\\\\",
        )
    )
    gt = ax.text(
        129.0,
        53.5,
        "GLF forest clearance\n1957-64",
        transform=ccrs.PlateCarree(),
        fontsize=9.0,
        color="#1d3d17",
        weight="bold",
        ha="center",
        zorder=9,
    )
    stroke_text(gt, 2.2)

    # ── overlay 3: NE China industrial belt — 156 Projects (1953-1957) ───────
    ax.add_patch(
        Ellipse(
            (123.0, 43.2),
            15.0,
            9.5,
            transform=ccrs.PlateCarree(),
            facecolor=C_IND,
            edgecolor="none",
            alpha=0.10,
            zorder=3,
        )
    )
    industrial = [
        ("Shenyang", 123.43, 41.80, 1.2, -1.5),
        ("Anshan", 122.99, 41.11, -7.2, -1.0),
        ("Fushun", 123.95, 41.86, 1.2, 1.2),
        ("Changchun", 125.32, 43.82, 1.2, 1.2),
        ("Harbin", 126.64, 45.76, 1.2, 1.2),
        ("Baotou", 110.00, 40.65, -7.0, -1.0),
    ]
    for name, lon, lat, dx, dy in industrial:
        ax.scatter(
            lon,
            lat,
            s=88,
            marker="s",
            color=C_IND,
            edgecolor="white",
            linewidth=0.9,
            transform=ccrs.PlateCarree(),
            zorder=9,
        )
        t = ax.text(
            lon + dx,
            lat + dy,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=8.0,
            color="#5a1a1a",
            weight="bold",
            zorder=9,
        )
        stroke_text(t, 1.9)

    it = ax.text(
        116.5,
        45.8,
        "156 Soviet industrial projects\n(58 in NE China, 1953-57)",
        transform=ccrs.PlateCarree(),
        fontsize=9.0,
        color="#5a1a1a",
        weight="bold",
        ha="center",
        zorder=9,
    )
    stroke_text(it, 2.3)

    # ── source swath outline ──────────────────────────────────────────────────
    swath = np.array(
        [
            [76.5, 43.1],
            [84, 45.8],
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
            [81, 40.3],
            [76.5, 41.8],
        ]
    )
    ax.add_patch(
        Polygon(
            swath,
            closed=True,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor=COLORS["gold"],
            linewidth=1.6,
            alpha=0.55,
            zorder=6,
            linestyle=(0, (6, 4)),
        )
    )
    st = ax.text(
        106.0,
        36.8,
        "wind source swath",
        transform=ccrs.PlateCarree(),
        fontsize=8.5,
        color=COLORS["gold"],
        weight="bold",
        ha="center",
        zorder=9,
    )
    stroke_text(st, 2.2)

    # ── Japan + wind hint ─────────────────────────────────────────────────────
    ax.scatter(
        139.7,
        35.7,
        s=180,
        marker="*",
        color=COLORS["red"],
        edgecolor="white",
        linewidth=1.1,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    jt = ax.text(
        141.8,
        34.5,
        "Japan\n(1st case 1961)",
        transform=ccrs.PlateCarree(),
        fontsize=9.5,
        color=COLORS["red"],
        weight="bold",
        ha="left",
        zorder=10,
    )
    stroke_text(jt, 2.6)
    ax.annotate(
        "",
        xy=(139.0, 37.5),
        xytext=(129.5, 41.0),
        xycoords=ccrs.PlateCarree(),
        textcoords=ccrs.PlateCarree(),
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["blue"],
            "lw": 1.6,
            "mutation_scale": 13,
            "alpha": 0.55,
            "linestyle": (0, (5, 4)),
        },
        zorder=6,
    )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C_IND, "", "■  156 Industrial Projects (coal, steel, no emission controls)"),
        (
            C_VL,
            "///",
            "▨  Virgin Lands Campaign (massive soil disturbance, livestock ↑)",
        ),
        (
            C_GLF,
            "\\\\\\",
            "▧  Great Leap Forward (Manchurian forest clearance, new dust sources)",
        ),
    ]
    for i, (color, _hatch, label) in enumerate(legend_items):
        bx = 0.018
        by = 0.128 - i * 0.042
        ax.add_patch(
            Rectangle(
                (bx, by),
                0.022,
                0.026,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="#666",
                linewidth=0.6,
                alpha=0.55,
                zorder=12,
            )
        )
        ax.text(
            bx + 0.030,
            by + 0.013,
            label,
            transform=ax.transAxes,
            fontsize=8.8,
            color=COLORS["ink"],
            va="center",
            zorder=12,
        )

    ax.set_title(
        "The upstream zone transforms  -  1950-1965",
        loc="left",
        pad=10,
        fontsize=14.0,
        fontweight="bold",
        color=COLORS["ink"],
    )

    # ── timeline strip ────────────────────────────────────────────────────────
    tl = fig.add_axes([0.08, 0.038, 0.88, 0.210])
    tl.set_facecolor(COLORS["panel"])
    for sp in ("top", "right"):
        tl.spines[sp].set_visible(False)
    tl.spines["left"].set_color(COLORS["grid"])
    tl.spines["bottom"].set_color(COLORS["grid"])

    x0, x1 = 1946, 1992
    tl.set_xlim(x0, x1)
    tl.set_ylim(0, 1)
    tl.set_yticks([])
    tl.grid(axis="x", alpha=0.18, color=COLORS["grid"])

    # Event bars
    for start, end, y, color, label in [
        (1953, 1957, 0.75, C_IND, "156 industrial projects"),
        (1953, 1965, 0.52, C_VL, "Virgin Lands Campaign"),
        (1957, 1964, 0.29, C_GLF, "Great Leap Forward"),
    ]:
        tl.barh(
            y,
            end - start,
            left=start,
            height=0.16,
            color=color,
            alpha=0.50,
            edgecolor=color,
            linewidth=0.7,
        )
        tl.text(
            (start + end) / 2,
            y,
            label,
            fontsize=8.5,
            color="white",
            weight="bold",
            ha="center",
            va="center",
        )

    # Schematic KD case curve on right axis
    kd_ax = tl.twinx()
    kd_ax.set_ylim(0, 19)
    kd_ax.set_yticks([0, 5, 10, 15])
    kd_ax.tick_params(colors=COLORS["red"], labelsize=8)
    kd_ax.spines["right"].set_color(COLORS["grid"])
    kd_ax.spines["top"].set_visible(False)
    kd_ax.spines["left"].set_visible(False)

    kd_yr = [
        1961,
        1963,
        1965,
        1967,
        1969,
        1970,
        1972,
        1974,
        1976,
        1978,
        1979,
        1980,
        1982,
        1983,
        1985,
        1986,
        1988,
        1990,
    ]
    kd_cs = [
        0.3,
        0.5,
        0.8,
        1.2,
        1.8,
        2.5,
        3.1,
        3.8,
        4.4,
        5.1,
        11.7,
        5.9,
        15.5,
        7.0,
        7.8,
        12.9,
        8.3,
        8.8,
    ]

    pre_yr = [y for y in kd_yr if y < 1970]
    pre_cs = [c for y, c in zip(kd_yr, kd_cs, strict=False) if y < 1970]
    post_yr = [y for y in kd_yr if y >= 1970]
    post_cs = [c for y, c in zip(kd_yr, kd_cs, strict=False) if y >= 1970]

    kd_ax.plot(
        pre_yr,
        pre_cs,
        color=COLORS["red"],
        linewidth=1.6,
        linestyle="--",
        alpha=0.55,
        zorder=4,
    )
    kd_ax.plot(
        post_yr,
        post_cs,
        color=COLORS["red"],
        linewidth=1.8,
        linestyle="-",
        alpha=0.88,
        zorder=4,
    )
    kd_ax.fill_between(kd_yr, 0, kd_cs, alpha=0.07, color=COLORS["red"])

    for yr in (1979, 1982, 1986):
        kd_ax.axvline(yr, color=COLORS["red"], linewidth=0.9, alpha=0.35, linestyle=":")

    kd_ax.text(
        x1 - 0.4,
        17.0,
        "Japan KD cases (x1000)\nschematic - see biennial surveys",
        fontsize=7.8,
        color=COLORS["red"],
        ha="right",
        va="top",
    )

    # Vertical event markers
    for yr, color, label, dy in [
        (1953, C_IND, "Five Year Plan\n+ Virgin Lands", 0.93),
        (1961, COLORS["red"], "First KD case", 0.93),
        (1967, COLORS["muted"], "First publication", 0.93),
        (1970, COLORS["muted"], "Surveys begin", 0.75),
    ]:
        tl.axvline(yr, color=color, linewidth=1.1, alpha=0.60, linestyle="--")
        tl.text(
            yr,
            dy,
            label,
            fontsize=7.2,
            color=color,
            ha="center",
            va="bottom",
            transform=tl.get_xaxis_transform(),
        )

    tl.set_xticks(range(1948, 1993, 4))
    tl.tick_params(colors=COLORS["muted"], labelsize=8.5)

    fig.text(
        0.030,
        0.268,
        "Polygon extents are schematic. KD curve approximate from Japan biennial surveys "
        "(Nakamura et al.; Pediatrics International). Dashed = pre-systematic surveillance.",
        fontsize=8.2,
        color=COLORS["muted"],
    )

    fig.savefig(
        OUT_DIR / "emergence-context.png", dpi=180, bbox_inches="tight", pad_inches=0.10
    )
    plt.close(fig)
    print("Saved: emergence-context.png")


def figure_chemistry_transit() -> None:
    """Schematic: source-region material transforms chemically during five-day transit."""
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06)

    ax.text(
        0.02, 0.955, "The atmosphere is not a passive conveyor belt",
        fontsize=16.5, fontweight="bold", color=COLORS["ink"], va="top",
    )
    ax.text(
        0.02, 0.905,
        "Five days at altitude with UV, ozone, and reactive trace gases — the material at the receptor is not the material at the source.",
        fontsize=10.2, color=COLORS["muted"], va="top",
    )

    box_h = 0.46
    box_y = 0.24
    box_w = 0.26
    panels = [
        {
            "x": 0.04,
            "title": "AT THE SOURCE",
            "subtitle": "NE China / Mongolia, January",
            "color": COLORS["gold"],
            "items": [
                "Mineral dust from Gobi edge",
                "Crop-residue fragments",
                "Fungal spores + bacterial cells",
                "Soil organochlorine residues",
                "Industrial SO₂, NO₂",
                "Ammonia from livestock soils",
            ],
        },
        {
            "x": 0.38,
            "title": "DURING TRANSIT",
            "subtitle": "850 hPa, 5 days, UV + O₃",
            "color": COLORS["blue"],
            "items": [
                "SO₂ + NH₃ → ammonium sulfate",
                "Organic VOCs → secondary organics",
                "Organochlorines photolyze + oxidize",
                "Spores adsorb onto dust surfaces",
                "Protein coatings fragment / recombine",
                "pH, water content shift continuously",
            ],
        },
        {
            "x": 0.72,
            "title": "AT THE RECEPTOR",
            "subtitle": "Japan / Hawaii / California airway",
            "color": COLORS["red"],
            "items": [
                "Aged sulfate + organic aerosol",
                "Modified spore surface chemistry",
                "New compounds absent at source",
                "Dust-bound protein complexes",
                "Reaction products, not ingredients",
                "Size-sorted by deposition en route",
            ],
        },
    ]

    for p in panels:
        ax.add_patch(Rectangle(
            (p["x"], box_y), box_w, box_h,
            facecolor="white", edgecolor=p["color"], linewidth=1.6, zorder=2,
        ))
        ax.add_patch(Rectangle(
            (p["x"], box_y + box_h - 0.085), box_w, 0.085,
            facecolor=p["color"], edgecolor="none", alpha=0.15, zorder=3,
        ))
        ax.text(
            p["x"] + 0.014, box_y + box_h - 0.028, p["title"],
            fontsize=10.5, fontweight="bold", color=p["color"], va="center", zorder=4,
        )
        ax.text(
            p["x"] + 0.014, box_y + box_h - 0.058, p["subtitle"],
            fontsize=8.5, color=COLORS["muted"], va="center", zorder=4,
        )
        for j, item in enumerate(p["items"]):
            y = box_y + box_h - 0.115 - j * 0.050
            ax.scatter(
                p["x"] + 0.022, y, s=18, color=p["color"],
                edgecolor="none", zorder=4,
            )
            ax.text(
                p["x"] + 0.038, y, item,
                fontsize=9.0, color=COLORS["ink"], va="center", zorder=4,
            )

    for cx in [0.305, 0.645]:
        ax.annotate(
            "",
            xy=(cx + 0.065, box_y + box_h / 2),
            xytext=(cx, box_y + box_h / 2),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["deep_blue"],
                "lw": 2.4,
                "mutation_scale": 22,
            },
        )
    ax.text(
        0.337, box_y + box_h / 2 + 0.055, "5 days",
        fontsize=8.6, color=COLORS["deep_blue"], fontweight="bold",
        ha="center", style="italic",
    )
    ax.text(
        0.677, box_y + box_h / 2 + 0.055, "deposition",
        fontsize=8.6, color=COLORS["deep_blue"], fontweight="bold",
        ha="center", style="italic",
    )

    ax.text(
        0.5, 0.15,
        "Sampling design consequence",
        fontsize=10.5, fontweight="bold", color=COLORS["ink"],
        ha="center", va="top",
    )
    ax.text(
        0.5, 0.115,
        "Source-region grab samples do not contain the transit reaction products. Flight sampling over the receptor (Rodo 2014) captures what is actually present.",
        fontsize=9.3, color=COLORS["muted"], ha="center", va="top",
    )

    save(fig, "chemistry-transit.png")


def figure_emergence_animation() -> None:
    """Animated timeline: source-zone disruptions 1950-1967 + Japan's first KD case."""
    C_IND = "#7a2a2a"
    C_VL = "#c8a050"
    C_GLF = "#4a7040"
    RIVER = "#4f8fa8"
    LAKE_F = "#c8e4ed"
    LAKE_E = "#6b9caf"

    vl_poly = np.array(
        [
            [58, 50.0], [63, 51.5], [69, 52.5], [76, 53.2], [82, 53.8],
            [86, 54.0], [86, 57.5], [80, 57.2], [72, 56.5], [63, 55.2],
            [58, 53.8], [58, 50.0],
        ]
    )
    glf_poly = np.array(
        [
            [118, 46.0], [126, 46.5], [131, 46.2], [133, 48.0],
            [131, 51.2], [127, 52.5], [122, 54.0], [118, 54.0], [118, 46.0],
        ]
    )
    swath = np.array(
        [
            [76.5, 43.1], [84, 45.8], [99, 46.2], [113, 46.4], [126, 44.8],
            [137, 40.5], [143, 37.2], [141, 34.6], [130, 37.0], [117, 38.5],
            [104, 39.5], [92, 39.6], [81, 40.3], [76.5, 41.8],
        ]
    )
    industrial_cities = [
        ("Shenyang", 123.43, 41.80, 1.2, -1.5),
        ("Anshan", 122.99, 41.11, -7.2, -1.0),
        ("Fushun", 123.95, 41.86, 1.2, 1.2),
        ("Changchun", 125.32, 43.82, 1.2, 1.2),
        ("Harbin", 126.64, 45.76, 1.2, 1.2),
        ("Baotou", 110.00, 40.65, -7.0, -1.0),
    ]

    def ramp(year: float, start: float, end: float, max_val: float = 1.0) -> float:
        if year < start:
            return 0.0
        if year >= end:
            return max_val
        return max_val * (year - start) / (end - start)

    base_years = np.arange(1950.0, 1967.1, 0.5)
    year_seq = [1950.0] * 3 + list(base_years)
    idx_1961 = next(i for i, y in enumerate(year_seq) if abs(y - 1961.0) < 0.01)
    year_seq = year_seq[: idx_1961 + 1] + [1961.0] * 3 + year_seq[idx_1961 + 1 :]
    year_seq = year_seq + [1967.0] * 6

    fig = plt.figure(figsize=(11.5, 7.6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_axes([0.030, 0.155, 0.940, 0.80], projection=ccrs.PlateCarree())
    bar_ax = fig.add_axes([0.08, 0.055, 0.84, 0.060])

    def draw(frame_i: int) -> None:
        ax.clear()
        bar_ax.clear()
        year = year_seq[frame_i]

        ax.set_extent([57, 145, 29, 62], crs=ccrs.PlateCarree())
        ax.set_facecolor(COLORS["water"])
        ax.set_aspect("auto")
        relief = ax.stock_img()
        relief.set_alpha(0.28)
        relief.set_zorder(0)
        ax.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor=COLORS["land"], edgecolor="none", alpha=0.55, zorder=1,
        )
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            linewidth=0.65, color=COLORS["coast"], zorder=7,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.28, color="#9b9a90", alpha=0.60, zorder=7,
        )
        ax.add_feature(
            cfeature.RIVERS.with_scale("50m"),
            linewidth=0.75, color=RIVER, alpha=0.72, zorder=5,
        )
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            facecolor=LAKE_F, edgecolor=LAKE_E,
            linewidth=0.50, alpha=0.85, zorder=5,
        )
        ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.35, color="#9fb0ac", alpha=0.28,
        )

        ax.add_patch(Polygon(
            swath, closed=True, transform=ccrs.PlateCarree(),
            facecolor="none", edgecolor=COLORS["gold"],
            linewidth=1.6, alpha=0.55, zorder=6, linestyle=(0, (6, 4)),
        ))
        st = ax.text(
            106.0, 36.8, "wind source swath",
            transform=ccrs.PlateCarree(), fontsize=8.3,
            color=COLORS["gold"], weight="bold", ha="center", zorder=9,
        )
        stroke_text(st, 2.2)

        vl_a = ramp(year, 1953.0, 1956.0, 0.28)
        if vl_a > 0.001:
            ax.add_patch(Polygon(
                vl_poly, closed=True, transform=ccrs.PlateCarree(),
                facecolor=C_VL, edgecolor="#8b6914",
                linewidth=0.9, alpha=vl_a, zorder=2, hatch="///",
            ))
            label_a = ramp(year, 1953.0, 1954.5, 1.0)
            if label_a > 0.001:
                vt = ax.text(
                    71.5, 55.5, "Virgin Lands\n1953-65",
                    transform=ccrs.PlateCarree(), fontsize=8.5,
                    color="#6b4a08", weight="bold", ha="center",
                    alpha=label_a, zorder=9,
                )
                stroke_text(vt, 2.2)

        glf_a = ramp(year, 1957.0, 1959.0, 0.25)
        if glf_a > 0.001:
            ax.add_patch(Polygon(
                glf_poly, closed=True, transform=ccrs.PlateCarree(),
                facecolor=C_GLF, edgecolor="#2d5028",
                linewidth=0.9, alpha=glf_a, zorder=3, hatch="\\\\\\",
            ))
            label_a = ramp(year, 1957.0, 1958.5, 1.0)
            if label_a > 0.001:
                gt = ax.text(
                    129.0, 53.5, "GLF clearance\n1957-64",
                    transform=ccrs.PlateCarree(), fontsize=8.5,
                    color="#1d3d17", weight="bold", ha="center",
                    alpha=label_a, zorder=9,
                )
                stroke_text(gt, 2.2)

        ind_a = ramp(year, 1953.0, 1957.0, 1.0)
        if ind_a > 0.001:
            ax.add_patch(Ellipse(
                (123.0, 43.2), 15.0, 9.5,
                transform=ccrs.PlateCarree(),
                facecolor=C_IND, edgecolor="none",
                alpha=0.10 * ind_a, zorder=3,
            ))
            for name, lon, lat, dx, dy in industrial_cities:
                ax.scatter(
                    lon, lat, s=70, marker="s", color=C_IND,
                    edgecolor="white", linewidth=0.7, alpha=ind_a,
                    transform=ccrs.PlateCarree(), zorder=9,
                )
                t = ax.text(
                    lon + dx, lat + dy, name,
                    transform=ccrs.PlateCarree(),
                    fontsize=7.5, color="#5a1a1a", weight="bold",
                    alpha=ind_a, zorder=9,
                )
                stroke_text(t, 1.9)
            label_a = ramp(year, 1953.0, 1954.5, 1.0)
            if label_a > 0.001:
                it = ax.text(
                    116.5, 45.8, "156 industrial projects\n(NE China, 1953-57)",
                    transform=ccrs.PlateCarree(), fontsize=8.5,
                    color="#5a1a1a", weight="bold", ha="center",
                    alpha=label_a, zorder=9,
                )
                stroke_text(it, 2.3)

        if year >= 1961.0:
            kd_a = ramp(year, 1961.0, 1961.25, 1.0)
            ax.scatter(
                139.7, 35.7, s=200, marker="*", color=COLORS["red"],
                edgecolor="white", linewidth=1.1, alpha=kd_a,
                transform=ccrs.PlateCarree(), zorder=10,
            )
            jt = ax.text(
                141.8, 34.5, "Japan\n(1st KD case)",
                transform=ccrs.PlateCarree(), fontsize=9.0,
                color=COLORS["red"], weight="bold", ha="left",
                alpha=kd_a, zorder=10,
            )
            stroke_text(jt, 2.6)
            ax.annotate(
                "",
                xy=(139.0, 37.5), xytext=(129.5, 41.0),
                xycoords=ccrs.PlateCarree(), textcoords=ccrs.PlateCarree(),
                arrowprops={
                    "arrowstyle": "-|>", "color": COLORS["blue"],
                    "lw": 1.5, "mutation_scale": 12,
                    "alpha": 0.55 * kd_a, "linestyle": (0, (5, 4)),
                },
                zorder=6,
            )
        else:
            ax.scatter(
                139.7, 35.7, s=50, marker="o", color="white",
                edgecolor=COLORS["coast"], linewidth=0.9, alpha=0.65,
                transform=ccrs.PlateCarree(), zorder=10,
            )

        ax.text(
            0.965, 0.955, f"{int(year)}",
            transform=ax.transAxes, fontsize=24, fontweight="bold",
            color=COLORS["ink"], ha="right", va="top", zorder=15,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": COLORS["panel"], "edgecolor": COLORS["grid"],
                "linewidth": 1.0, "alpha": 0.92,
            },
        )
        ax.set_title(
            "The upstream zone transforms, 1950-1967",
            loc="left", pad=8, fontsize=13, fontweight="bold", color=COLORS["ink"],
        )

        bar_ax.set_xlim(1950, 1967)
        bar_ax.set_ylim(0, 1)
        bar_ax.set_yticks([])
        for sp in ("top", "right", "left"):
            bar_ax.spines[sp].set_visible(False)
        bar_ax.set_facecolor(COLORS["bg"])

        timeline_events = [
            (1953, 1957, 0.75, C_IND, "156 projects"),
            (1953, 1965, 0.45, C_VL, "Virgin Lands"),
            (1957, 1964, 0.15, C_GLF, "GLF clearance"),
        ]
        for start, end, y_off, color, label in timeline_events:
            bar_ax.barh(
                y_off, end - start, left=start, height=0.16,
                color=color, alpha=0.28, edgecolor=color, linewidth=0.5,
            )
            if year >= start:
                active_w = min(year, end) - start
                if active_w > 0:
                    bar_ax.barh(
                        y_off, active_w, left=start, height=0.16,
                        color=color, alpha=0.85, edgecolor="none",
                    )
            bar_ax.text(
                end + 0.25, y_off, label,
                fontsize=7.2, color=color, weight="bold", va="center",
            )

        bar_ax.axvline(
            1961, color=COLORS["red"],
            linewidth=1.0 if year < 1961 else 1.6,
            alpha=0.45 if year < 1961 else 1.0, linestyle="--",
        )
        if year >= 1961:
            bar_ax.text(
                1961.1, 1.08, "first KD case",
                fontsize=7.2, color=COLORS["red"], weight="bold",
                va="top", transform=bar_ax.get_xaxis_transform(),
            )
        bar_ax.axvline(year, color=COLORS["ink"], linewidth=1.6, alpha=0.70)
        bar_ax.set_xticks(range(1950, 1968, 3))
        bar_ax.tick_params(colors=COLORS["muted"], labelsize=7.5)

    ani = animation.FuncAnimation(
        fig, draw, frames=len(year_seq), interval=180, repeat=True,
    )
    ani.save(
        OUT_DIR / "emergence-timeline.gif",
        writer=animation.PillowWriter(fps=6), dpi=88,
    )
    plt.close(fig)
    print("Saved: emergence-timeline.gif")


def figure_incidence_specificity() -> None:
    """Horizontal bar chart: annual KD incidence by country."""
    entries = [
        ("Japan", 265, COLORS["red"], True),
        ("South Korea", 190, COLORS["red"], True),
        ("Taiwan", 90, COLORS["gold"], True),
        ("Hawaii (USA)", 45, COLORS["gold"], True),
        ("United States", 22, COLORS["blue"], False),
        ("Australia", 9, COLORS["muted"], False),
        ("United Kingdom", 8, COLORS["muted"], False),
        ("Germany", 5, COLORS["muted"], False),
    ]
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    east_asia = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])
    fig.subplots_adjust(left=0.22, right=0.96, top=0.82, bottom=0.12)

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, values, color=colors, edgecolor="none", height=0.60, alpha=0.88
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis()

    for bar, val, _ea in zip(bars, values, east_asia, strict=False):
        label_str = f"{val} / 100k"
        if bar.get_width() > 220:
            ax.text(
                bar.get_width() - 5,
                bar.get_y() + bar.get_height() / 2,
                label_str,
                va="center",
                ha="right",
                fontsize=10.0,
                weight="bold",
                color="white",
            )
        else:
            ax.text(
                bar.get_width() + 4,
                bar.get_y() + bar.get_height() / 2,
                label_str,
                va="center",
                ha="left",
                fontsize=10.0,
                weight="bold",
                color=COLORS["ink"],
            )

    ax.axhline(3.55, color=COLORS["grid"], linewidth=1.4, linestyle="--", alpha=0.7)
    ax.text(
        270,
        3.0,
        "Asia-Pacific / downwind",
        fontsize=8.5,
        color=COLORS["muted"],
        ha="right",
        style="italic",
    )
    ax.text(
        270,
        4.15,
        "Western / upwind",
        fontsize=8.5,
        color=COLORS["muted"],
        ha="right",
        style="italic",
    )

    ax.set_xlabel(
        "Annual incidence per 100,000 children under 5 (approximate)",
        fontsize=10.0,
        color=COLORS["muted"],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.28)
    ax.set_xlim(0, 295)

    ax.set_title(
        "Kawasaki disease incidence is highest downwind of the source zone",
        loc="left",
        pad=14,
        fontsize=14.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.22,
        0.88,
        "Japan's rate exceeds Germany's by more than 50-fold. Hawaii, geographically between Japan and the US mainland, sits in between.",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    fig.text(
        0.22,
        0.01,
        "Approximate values from: Nakamura et al. (2012, Pediatrics Int.); Korean CDC biennial surveys; "
        "Chang et al. (2015, PLOS ONE); Burns (2018, Int. J. Rheumatic Dis.)",
        fontsize=7.5,
        color=COLORS["muted"],
    )

    save(fig, "incidence-specificity.png")


def figure_biology_constraints() -> None:
    """Four biological constraints that any surviving KD agent must satisfy."""
    GREEN = COLORS["green"]
    BLUE = COLORS["blue"]

    rows = [
        {
            "constraint": "Respiratory entry route",
            "evidence": "Oligoclonal IgA plasma cells in medium-airway\ntissue -- Rowley et al. 1997 (J. Infect. Dis.)",
            "filter": "INHALATION\nONLY",
            "vc": BLUE,
            "note": "Not ingestion, not dermal contact.\nRespiratory mucosal sampling matters;\nfecal and dermal screens probably miss it.",
        },
        {
            "constraint": "Trans-Pacific transport survival\n(5-10 days at altitude, UV, desiccation)",
            "evidence": "850 hPa pathway at 5-15 m/s; continuous UV,\nfreezing temps, and humidity cycling en route",
            "filter": "SPORE / STABLE\nPROTEIN / TOXIN",
            "vc": GREEN,
            "note": "Fragile enveloped viruses mostly excluded.\nFungal spores, bacterial endospores,\ndust-adsorbed heat-stable proteins survive.",
        },
        {
            "constraint": "Conventional antigen-specific\nimmune response",
            "evidence": "CD4+ and CD8+ memory T cells emerge\npost-illness -- Burns 2024 (JCI)",
            "filter": "NOT A\nSUPERANTIGEN",
            "vc": BLUE,
            "note": "Normal antigen-presentation pathway.\nHighly immunogenic surface antigen needed\nto drive systemic vasculitis in young children.",
        },
        {
            "constraint": "Single predominant agent\nacross 50+ years and two countries",
            "evidence": "Same cytoplasmic viral inclusions in 20 samples\n(US + Japan, 1966-2017) -- Rowley 2025 (Lab Invest)",
            "filter": "ONE AGENT\nNOT MANY",
            "vc": GREEN,
            "note": "Inclusions in medium airways confirm respiratory\nentry. Multiple-pathogen and toxin-only\nhypotheses inconsistent with this pattern.",
        },
    ]

    fig, ax = plt.subplots(figsize=(14.0, 7.0), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)

    ax.text(
        0.02,
        0.97,
        "What the biology says the agent must be",
        fontsize=17.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        0.02,
        0.912,
        "Four constraints imposed by the disease itself. Any surviving candidate must satisfy all four before sampling starts to mean anything.",
        fontsize=10.8,
        color=COLORS["muted"],
        va="top",
    )

    hdr_y = 0.852
    for x, label in [
        (0.04, "BIOLOGICAL CONSTRAINT"),
        (0.38, "EVIDENCE BASIS"),
        (0.625, "FILTER"),
        (0.775, "WHAT IT RULES IN OR OUT"),
    ]:
        ax.text(
            x,
            hdr_y,
            label,
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["muted"],
            va="top",
        )
    ax.plot(
        [0.02, 0.98],
        [hdr_y - 0.022, hdr_y - 0.022],
        color=COLORS["grid"],
        linewidth=1.5,
    )

    row_top = 0.810
    row_h = 0.185

    for i, row in enumerate(rows):
        y_top = row_top - i * row_h
        y_bot = y_top - row_h
        y_mid = (y_top + y_bot) / 2

        if i % 2 == 1:
            ax.add_patch(
                Rectangle(
                    (0.02, y_bot + 0.006),
                    0.96,
                    row_h - 0.006,
                    facecolor="#f4f8f4",
                    edgecolor="none",
                )
            )

        ax.add_patch(
            Rectangle(
                (0.02, y_bot + 0.009),
                0.007,
                row_h - 0.018,
                facecolor=row["vc"],
                edgecolor="none",
            )
        )

        ax.text(
            0.04,
            y_mid,
            row["constraint"],
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
            linespacing=1.35,
        )

        ax.text(
            0.38,
            y_mid,
            row["evidence"],
            fontsize=9.4,
            color=COLORS["muted"],
            va="center",
            linespacing=1.35,
        )

        badge_bg = "#eef4ff" if row["vc"] == BLUE else "#f0f8f0"
        ax.add_patch(
            Rectangle(
                (0.625, y_mid - 0.055),
                0.130,
                0.110,
                facecolor=badge_bg,
                edgecolor=row["vc"],
                linewidth=1.5,
            )
        )
        ax.text(
            0.690,
            y_mid,
            row["filter"],
            fontsize=8.6,
            fontweight="bold",
            color=row["vc"],
            ha="center",
            va="center",
            linespacing=1.25,
        )

        ax.text(
            0.775,
            y_mid,
            row["note"],
            fontsize=9.4,
            color=COLORS["ink"],
            va="center",
            linespacing=1.35,
        )

        ax.plot(
            [0.02, 0.98],
            [y_bot + 0.006, y_bot + 0.006],
            color=COLORS["grid"],
            linewidth=0.7,
        )

    save(fig, "biology-constraints.png")


def figure_workflow_steps() -> None:
    """Five-step open-data analysis workflow as a vertical flow diagram."""
    steps = [
        {
            "n": "1",
            "title": "Back-trajectory ensemble from the downstream sites",
            "tool": "NOAA HYSPLIT via READY web interface (no account required)",
            "what": "Jan-Mar origins from Yokohama, Honolulu, San Diego: do epidemic-year\ntrajectories cluster over NE China / Gobi? Random scatter -> stop here.",
            "kill": True,
            "color": COLORS["red"],
        },
        {
            "n": "2",
            "title": "Aerosol optical depth in the upstream source zone",
            "tool": "NASA MODIS AOD (Giovanni portal) · CALIPSO vertical profiles · MERRA-2 dust fields",
            "what": "Is AOD elevated in NE China / Gobi during high-P-WIND months?\nCan CALIPSO separate dust from smoke at aerosol-transport altitude?",
            "kill": False,
            "color": COLORS["gold"],
        },
        {
            "n": "3",
            "title": "Characterize the upstream land surface without visiting it",
            "tool": "FAO Global Livestock Density · ESA CCI Land Cover · NASA FIRMS fire detections",
            "what": "Grazing intensity (Coxiella prior), dominant crop type (Fusarium pressure),\nharvest burn timing. Does burn aerosol lag the Jan KD peak by a consistent gap?",
            "kill": False,
            "color": COLORS["gold"],
        },
        {
            "n": "4",
            "title": "Industrial emission and pollution overlays",
            "tool": "GBD air pollution layers · EDGAR emission inventory · Copernicus TROPOMI (NO₂, SO₂)",
            "what": "Does the signal follow industrial PM2.5 from manufacturing corridors, or Gobi dust?\nTROPOMI separates combustion zones from desert-edge upwind regions.",
            "kill": False,
            "color": COLORS["blue"],
        },
        {
            "n": "5",
            "title": "Negative controls: other winter respiratory diseases in Japan",
            "tool": "Japan NIID seasonal surveillance (influenza, RSV, rotavirus case counts by month)",
            "what": "Does P-WIND predict all winter diseases equally? If yes -> winter proxy, not transport.\nIf KD tracks better than influenza and RSV -> specificity is real.",
            "kill": True,
            "color": COLORS["muted"],
        },
    ]

    fig, ax = plt.subplots(figsize=(13.5, 9.0))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    ax.text(
        0.02,
        0.975,
        "Five steps from public data to a decision",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        0.02,
        0.934,
        "In order of what is most likely to kill the hypothesis. Steps marked KILL TEST are the decision points that end the analysis if negative.",
        fontsize=10.0,
        color=COLORS["muted"],
        va="top",
    )

    n_steps = len(steps)
    step_h = 0.142
    step_gap = 0.016
    y_start = 0.888
    lbox_x = 0.020
    lbox_w = 0.058
    panel_x = lbox_x + lbox_w + 0.012
    panel_w = 0.960 - panel_x

    for i, step in enumerate(steps):
        y_top = y_start - i * (step_h + step_gap)
        y_bot = y_top - step_h
        y_mid = (y_top + y_bot) / 2

        # Main background panel
        ax.add_patch(
            Rectangle(
                (panel_x, y_bot),
                panel_w,
                step_h,
                facecolor=COLORS["panel"],
                edgecolor=step["color"],
                linewidth=1.8 if step["kill"] else 0.7,
                alpha=0.95,
                zorder=2,
            )
        )

        # Left step-number colored box
        ax.add_patch(
            Rectangle(
                (lbox_x, y_bot),
                lbox_w,
                step_h,
                facecolor=step["color"],
                edgecolor="none",
                alpha=0.90,
                zorder=2,
            )
        )
        ax.text(
            lbox_x + lbox_w / 2,
            y_mid,
            step["n"],
            fontsize=20,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=3,
        )

        # KILL TEST badge
        if step["kill"]:
            bx = panel_x + panel_w - 0.118
            by = y_mid + 0.025
            ax.add_patch(
                Rectangle(
                    (bx, by - 0.018),
                    0.108,
                    0.036,
                    facecolor=step["color"],
                    edgecolor="none",
                    alpha=0.22,
                    zorder=3,
                )
            )
            ax.text(
                bx + 0.054,
                by,
                "KILL TEST",
                fontsize=7.8,
                fontweight="bold",
                color=step["color"],
                ha="center",
                va="center",
                zorder=4,
            )

        tx = panel_x + 0.016
        # Title
        ax.text(
            tx,
            y_top - 0.018,
            step["title"],
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            va="top",
            zorder=3,
        )
        # Data/tool label
        ax.text(
            tx,
            y_top - 0.052,
            f"Data:  {step['tool']}",
            fontsize=8.4,
            color=step["color"],
            va="top",
            zorder=3,
        )
        # Description
        ax.text(
            tx,
            y_top - 0.084,
            step["what"],
            fontsize=9.0,
            color=COLORS["muted"],
            va="top",
            linespacing=1.28,
            zorder=3,
        )

        # Downward connector arrow
        if i < n_steps - 1:
            ax.annotate(
                "",
                xy=(lbox_x + lbox_w / 2, y_bot - step_gap * 0.80),
                xytext=(lbox_x + lbox_w / 2, y_bot),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": COLORS["grid"],
                    "lw": 1.6,
                    "mutation_scale": 13,
                },
                zorder=2,
            )

    save(fig, "workflow-steps.png")


def figure_ruled_out() -> None:
    """Hypothesis filter table: tested candidates and their outcomes."""
    RED = COLORS["red"]  # "#b6533f"
    AMBER = COLORS["gold"]  # "#c8912d"

    rows = [
        {
            "hyp": "TSST-1 superantigen\nmechanism",
            "evidence": "Microbiologic + immunologic follow-up\nLeung 1993 (Lancet) -> Burns 2024 (JCI)",
            "verdict": "RULED OUT",
            "vc": RED,
            "note": "Response is conventional antigen-driven;\nCD4+/CD8+ memory T cells emerge post-illness",
        },
        {
            "hyp": "Known human coronaviruses\n(incl. all COVID variants)",
            "evidence": "KD sera antibody profiling vs.\nevery known + novel coronavirus epitope",
            "verdict": "RULED OUT",
            "vc": RED,
            "note": "No antibody response consistent with\nany known coronavirus -- Burns 2024 (JCI)",
        },
        {
            "hyp": "Fine particulate air\npollution (PM2.5)",
            "evidence": "Multicenter US study: KD case counts\nvs. ambient PM2.5 at patient locations",
            "verdict": "RULED OUT",
            "vc": RED,
            "note": "No relationship between PM2.5 and KD.\nSignal is specific, not generic smog",
        },
        {
            "hyp": "Multiple independent\ntriggers / pathogens",
            "evidence": "Antibody recognition of inclusion bodies:\n20 tissue samples, US + Japan, 50 years",
            "verdict": "RULED OUT",
            "vc": RED,
            "note": "Same target in every sample -> one\npredominant agent  (Rowley 2025, Lab Invest)",
        },
        {
            "hyp": "Coxiella burnetii as\nprimary disease driver",
            "evidence": "Q fever clinical picture in children\nvs. KD manifestations (CDC MMWR 2013)",
            "verdict": "MECHANISM\nMISMATCH",
            "vc": AMBER,
            "note": "Q fever -> hepatitis, pneumonia in children;\nKD -> coronary artery vasculitis",
        },
        {
            "hyp": "Bulk spring dust aerosol\nas the trigger",
            "evidence": "AERONET Level 2.0 AOD at Beijing, Seoul,\nOsaka vs. seasonal KD peak (P-WIND)",
            "verdict": "PHASE\nMISMATCH",
            "vc": AMBER,
            "note": "AOD peaks April-May; KD peaks Jan-Feb.\nHigh dust season != high disease season",
        },
    ]

    fig, ax = plt.subplots(figsize=(14.0, 7.6), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)

    ax.text(
        0.02,
        0.97,
        "What testing has ruled out -- and what it implies",
        fontsize=17.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        0.02,
        0.912,
        "Six candidates tested against mechanism and survival constraints imposed by the disease itself. "
        "Each failure narrows the remaining search.",
        fontsize=10.8,
        color=COLORS["muted"],
        va="top",
    )

    hdr_y = 0.855
    for x, label in [
        (0.04, "HYPOTHESIS"),
        (0.35, "EVIDENCE APPLIED"),
        (0.585, "VERDICT"),
        (0.745, "IMPLICATION"),
    ]:
        ax.text(
            x,
            hdr_y,
            label,
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["muted"],
            va="top",
        )
    ax.plot(
        [0.02, 0.98],
        [hdr_y - 0.022, hdr_y - 0.022],
        color=COLORS["grid"],
        linewidth=1.5,
    )

    row_top = 0.815
    row_h = 0.123

    for i, row in enumerate(rows):
        y_top = row_top - i * row_h
        y_bot = y_top - row_h
        y_mid = (y_top + y_bot) / 2

        if i % 2 == 1:
            ax.add_patch(
                Rectangle(
                    (0.02, y_bot + 0.006),
                    0.96,
                    row_h - 0.006,
                    facecolor="#f5f7f5",
                    edgecolor="none",
                )
            )

        ax.add_patch(
            Rectangle(
                (0.02, y_bot + 0.009),
                0.007,
                row_h - 0.018,
                facecolor=row["vc"],
                edgecolor="none",
            )
        )

        ax.text(
            0.04,
            y_mid,
            row["hyp"],
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
            linespacing=1.35,
        )

        ax.text(
            0.35,
            y_mid,
            row["evidence"],
            fontsize=9.4,
            color=COLORS["muted"],
            va="center",
            linespacing=1.35,
        )

        badge_bg = "#fff2ef" if row["vc"] == RED else "#fdf5e4"
        ax.add_patch(
            Rectangle(
                (0.585, y_mid - 0.038),
                0.145,
                0.076,
                facecolor=badge_bg,
                edgecolor=row["vc"],
                linewidth=1.5,
            )
        )
        ax.text(
            0.6575,
            y_mid,
            row["verdict"],
            fontsize=8.6,
            fontweight="bold",
            color=row["vc"],
            ha="center",
            va="center",
            linespacing=1.25,
        )

        ax.text(
            0.745,
            y_mid,
            row["note"],
            fontsize=9.4,
            color=COLORS["ink"],
            va="center",
            linespacing=1.35,
        )

        ax.plot(
            [0.02, 0.98],
            [y_bot + 0.006, y_bot + 0.006],
            color=COLORS["grid"],
            linewidth=0.7,
        )

    save(fig, "ruled-out-candidates.png")


def print_summary(ds: xr.Dataset) -> None:
    p = p_wind(ds)
    nw = nw_wind_japan(ds)
    p_arr = p.to_numpy()
    nw_arr = nw.to_numpy()
    print("period: 1996-2006 monthly means")
    print(
        f"p-wind max month: {MONTHS[int(np.argmax(p_arr))]} ({float(p.max()):.2f} m/s)"
    )
    print(
        f"p-wind min month: {MONTHS[int(np.argmin(p_arr))]} ({float(p.min()):.2f} m/s)"
    )
    print(
        f"japan nw-wind max month: {MONTHS[int(np.argmax(nw_arr))]} ({float(nw.max()):.2f} m/s)"
    )
    print(
        f"japan nw-wind min month: {MONTHS[int(np.argmin(nw_arr))]} ({float(nw.min()):.2f} m/s)"
    )


def main() -> None:
    ds = load_winds()
    image_hero(ds)
    image_seasonal_indices(ds)
    image_source_screening_map()
    image_monthly_animation(ds)
    image_analysis_boundary()
    image_emergence_context()
    figure_ruled_out()
    figure_incidence_specificity()
    figure_biology_constraints()
    figure_workflow_steps()
    figure_chemistry_transit()
    figure_emergence_animation()
    print_summary(ds)


if __name__ == "__main__":
    main()
