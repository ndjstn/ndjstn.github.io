"""Kawasaki open-data analysis: back-trajectories and AERONET AOD.

Two figures saved to assets/img/posts/kawasaki-wind-patterns/:
  back-trajectories.png     5-day kinematic back-trajectories from Japan at
                            850 hPa (realistic aerosol-transport level), all
                            months, colored by P-WIND intensity.
  aeronet-seasonal-aod.png  Monthly AERONET AOD at three swath stations with
                            the P-WIND seasonal cycle overlaid.

Data sources (no credentials required):
  NCEP/NCAR Reanalysis 1 via NOAA PSL OPeNDAP
  AERONET Version 3 Level 2.0 daily AOD via public HTTP API
"""

from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("assets/img/posts/kawasaki-wind-patterns")
CACHE_DIR = Path(".cache/kawasaki-wind")
AOD_DIR = CACHE_DIR / "aeronet"
WIND_CACHE_NARROW = (
    CACHE_DIR / "ncep_pacific_1996_2006.nc"
)  # existing: 300+850 hPa, 100-260°E
WIND_CACHE_WIDE = CACHE_DIR / "ncep_850hpa_wide_1996_2006.nc"  # new: 850 hPa, 65-145°E

# ── NOAA PSL OPeNDAP ──────────────────────────────────────────────────────────
U_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/uwnd.mon.mean.nc"
V_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/vwnd.mon.mean.nc"
PERIOD = slice("1996-01-01", "2006-12-31")

EARTH_R = 6_371_000  # metres
TRAJ_DAYS = 5  # 5 days at 850 hPa puts January origins in NE-China / Mongolia
TRAJ_DT_H = 6  # 6-hour integration step

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
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
    }
)

# ── AERONET stations ──────────────────────────────────────────────────────────
# Three stations with reliable coverage; Dunhuang (21 obs) excluded as too sparse.
# (api_name, display_name, lon, lat, role)
STATIONS = [
    ("Beijing", "Beijing\n(N China Plain)", 116.38, 39.98, "source"),
    ("Seoul_SNU", "Seoul\n(transit)", 126.95, 37.46, "transit"),
    ("Osaka", "Osaka\n(Japan check)", 135.59, 34.65, "validation"),
]

ROLE_COLOR = {
    "source": COLORS["gold"],
    "transit": COLORS["green"],
    "validation": COLORS["blue"],
}


# ── utilities ─────────────────────────────────────────────────────────────────


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    AOD_DIR.mkdir(parents=True, exist_ok=True)


def stroke(t, lw: float = 3.0) -> None:
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground=COLORS["panel"])])


# ── wind loading ──────────────────────────────────────────────────────────────


def load_narrow_winds() -> xr.Dataset:
    """300 + 850 hPa, lon 100-260 E. Reuses the existing wind-article cache."""
    if WIND_CACHE_NARROW.exists():
        return xr.open_dataset(WIND_CACHE_NARROW).load()

    print("  Downloading narrow-domain winds…")
    u = xr.open_dataset(U_URL)["uwnd"].sel(
        time=PERIOD, level=[300.0, 850.0], lat=slice(65, 5), lon=slice(100, 260)
    )
    v = xr.open_dataset(V_URL)["vwnd"].sel(
        time=PERIOD, level=[300.0, 850.0], lat=slice(65, 5), lon=slice(100, 260)
    )
    ds = xr.Dataset({"u": u, "v": v}).load()
    ds.to_netcdf(WIND_CACHE_NARROW)
    return ds


def load_wide_winds() -> xr.Dataset:
    """850 hPa only, lon 65-145 E.

    Using 850 hPa (low-tropospheric level) rather than 300 hPa for trajectories
    because aerosols travel at altitudes where they can be deposited, not in the
    jet core. Extending west to 65°E ensures 5-day winter trajectories don't
    stall at the domain boundary — at typical 850 hPa speeds (5-15 m/s) they
    land in NE China / Mongolia, not Central Asia.
    """
    if WIND_CACHE_WIDE.exists():
        return xr.open_dataset(WIND_CACHE_WIDE).load()

    print("  Downloading wide-domain 850 hPa winds (65-145 E)...")
    u = xr.open_dataset(U_URL)["uwnd"].sel(
        time=PERIOD, level=850.0, lat=slice(70, 10), lon=slice(65, 145)
    )
    v = xr.open_dataset(V_URL)["vwnd"].sel(
        time=PERIOD, level=850.0, lat=slice(70, 10), lon=slice(65, 145)
    )
    ds = xr.Dataset({"u": u, "v": v}).load()
    ds.to_netcdf(WIND_CACHE_WIDE)
    return ds


def p_wind_climatology(ds: xr.Dataset) -> np.ndarray:
    """12-value monthly P-WIND climatology (300 hPa zonal wind, 35N, 140-240E)."""
    u = ds.sel(level=300)["u"].sel(lat=35, method="nearest").sel(lon=slice(140, 240))
    return u.mean("lon").groupby("time.month").mean("time").to_numpy()


# ── trajectory ────────────────────────────────────────────────────────────────


def _make_interp_pair(
    ds: xr.Dataset, year: int, month: int
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """Build u, v interpolators for one year-month from a dataset without a
    level dimension (i.e. after level has been selected on load)."""
    mask = (ds.time.dt.year == year) & (ds.time.dt.month == month)
    u_f = ds["u"].isel(time=mask).squeeze()
    v_f = ds["v"].isel(time=mask).squeeze()

    lat = u_f["lat"].to_numpy()
    lon = u_f["lon"].to_numpy()
    u_a = u_f.to_numpy()
    v_a = v_f.to_numpy()

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u_a = u_a[::-1]
        v_a = v_a[::-1]

    kw = {"bounds_error": False, "fill_value": 0.0}
    return (
        RegularGridInterpolator((lat, lon), u_a, **kw),
        RegularGridInterpolator((lat, lon), v_a, **kw),
    )


def back_trajectory(
    u_i: RegularGridInterpolator,
    v_i: RegularGridInterpolator,
    start_lon: float,
    start_lat: float,
    n_days: int = TRAJ_DAYS,
    dt_hours: float = TRAJ_DT_H,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler kinematic back-trajectory at 850 hPa.

    At typical 850 hPa winter speeds over East Asia (5-15 m/s), 5 days traces
    back 25-75 degrees of longitude — landing in NE China or Mongolia, not the
    Middle East. fill_value=0 outside grid causes natural stalling at the
    domain edge rather than crashing.
    """
    dt = dt_hours * 3600
    n_steps = int(n_days * 24 / dt_hours)
    lon, lat = float(start_lon), float(start_lat)
    lons, lats = [lon], [lat]

    for _ in range(n_steps):
        u = float(u_i([[lat, lon]])[0])
        v = float(v_i([[lat, lon]])[0])
        cos_lat = max(abs(np.cos(np.deg2rad(lat))), 0.08)
        lat = float(np.clip(lat - v * dt / EARTH_R * (180 / np.pi), 10, 70))
        lon = lon - u * dt / (EARTH_R * cos_lat) * (180 / np.pi)
        lons.append(lon)
        lats.append(lat)

    return np.array(lons), np.array(lats)


# ── AERONET ───────────────────────────────────────────────────────────────────


def fetch_aeronet(
    api_name: str, year1: int = 2001, year2: int = 2020
) -> pd.DataFrame | None:
    """Download AERONET Level 2.0 daily AOD and return a 12-row monthly climatology.

    AOD20=1 → Level 2.0 quality-assured data. AVG=20 → one row per day.
    We group by calendar month to build the climatology.
    """
    cache = AOD_DIR / f"{api_name}_{year1}_{year2}_aod20_daily.csv"
    if cache.exists():
        try:
            return pd.read_csv(cache)
        except Exception:  # noqa: S110
            pass

    url = (
        "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
        f"?site={api_name}"
        f"&year={year1}&month=1&day=1"
        f"&year2={year2}&month2=12&day2=31"
        "&AOD20=1&AVG=20&if_no_html=1"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"    {api_name}: request failed — {exc}")
        return None

    lines = resp.text.splitlines()
    hdr_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("AERONET_Site")), None
    )
    if hdr_idx is None:
        print(f"    {api_name}: no data header in response")
        return None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines[hdr_idx:])),
            na_values=["N/A", "-999", "-999.", "-999.0"],
        )
    except Exception as exc:
        print(f"    {api_name}: CSV parse error — {exc}")
        return None

    aod_col = next(
        (
            c
            for c in ["AOD_500nm", "AOD_440nm", "AOD_675nm", "AOD_380nm", "AOD_870nm"]
            if c in df.columns
        ),
        next((c for c in df.columns if "AOD_" in c and "nm" in c), None),
    )
    if aod_col is None:
        print(f"    {api_name}: no AOD column found")
        return None

    date_col = next((c for c in df.columns if "Date" in c), None)
    if date_col is None:
        return None

    try:
        df["_month"] = pd.to_datetime(df[date_col], format="%d:%m:%Y").dt.month
    except Exception:
        return None

    df[aod_col] = pd.to_numeric(df[aod_col], errors="coerce")
    df = df[df[aod_col] > 0].dropna(subset=[aod_col])
    if df.empty:
        print(f"    {api_name}: no valid readings after cleaning")
        return None

    clim = (
        df.groupby("_month")[aod_col]
        .agg(aod_mean="mean", aod_std="std", n_obs="count")
        .reindex(range(1, 13))
        .reset_index()
        .rename(columns={"_month": "month"})
    )
    clim["wavelength"] = aod_col
    clim.to_csv(cache, index=False)
    print(f"    {api_name}: {len(df)} daily obs → climatology ({aod_col})")
    return clim


# ── figure 1: back-trajectories ───────────────────────────────────────────────


def figure_trajectories(narrow_ds: xr.Dataset, wide_ds: xr.Dataset) -> None:
    """
    5-day 850 hPa back-trajectory ensemble from Japan, one mean path per month.
    Lines colored warm→cool by P-WIND (300 hPa circulation index).

    Using 850 hPa keeps origins physically plausible: at typical low-tropospheric
    speeds over East Asia (5-15 m/s), 5 days traces back into NE China, the
    North China Plain, or eastern Mongolia — not into Central Asia or beyond.
    """
    print("  Computing 850 hPa trajectory ensemble…")
    p_clim = p_wind_climatology(narrow_ds)
    years = np.unique(narrow_ds.time.dt.year.to_numpy())

    cmap = plt.cm.RdYlBu_r
    p_norm = Normalize(vmin=p_clim.min(), vmax=p_clim.max())

    JAPAN = (139.7, 35.7)

    fig = plt.figure(figsize=(13.5, 8.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_axes([0.03, 0.08, 0.87, 0.84], projection=ccrs.PlateCarree())
    ax.set_extent([65, 145, 18, 62], crs=ccrs.PlateCarree())
    ax.set_facecolor(COLORS["water"])
    ax.set_aspect("auto")

    # Terrain relief gives geographic context without cluttering
    relief = ax.stock_img()
    relief.set_alpha(0.38)
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
        linewidth=0.70,
        color=COLORS["coast"],
        zorder=5,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.30,
        color="#9b9a90",
        alpha=0.65,
        zorder=5,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        linewidth=0.65,
        color="#4f8fa8",
        alpha=0.72,
        zorder=4,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="#c8e4ed",
        edgecolor="#6b9caf",
        linewidth=0.45,
        alpha=0.82,
        zorder=4,
    )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.42,
        color="#9fb0ac",
        alpha=0.30,
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 8.5, "color": COLORS["muted"]}
    gl.ylabel_style = {"size": 8.5, "color": COLORS["muted"]}

    # Source swath (same polygon as wind article)
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
            linewidth=1.4,
            alpha=0.26,
            zorder=2,
        )
    )
    st = ax.text(
        100,
        45.5,
        "source swath",
        transform=ccrs.PlateCarree(),
        fontsize=9.5,
        color="#7a5a1a",
        weight="bold",
        ha="center",
        zorder=8,
    )
    stroke(st, 2.5)

    # Terrain labels for geographic context
    for label, lon, lat in [
        ("Gobi", 103.5, 42.0),
        ("Manchuria", 125.0, 46.5),
        ("N China Plain", 116.0, 35.5),
        ("Mongolia", 103.0, 47.5),
    ]:
        t = ax.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=8.0,
            color="#6b5a3a",
            style="italic",
            ha="center",
            alpha=0.80,
            zorder=6,
        )
        stroke(t, 2.0)

    for month_idx in range(12):
        month = month_idx + 1
        color = cmap(p_norm(p_clim[month_idx]))
        is_winter = p_clim[month_idx] > float(np.median(p_clim))

        year_lons, year_lats = [], []
        for yr in years:
            mask = (wide_ds.time.dt.year == int(yr)) & (wide_ds.time.dt.month == month)
            if not bool(mask.any()):
                continue
            try:
                ui, vi = _make_interp_pair(wide_ds, int(yr), month)
                lns, lts = back_trajectory(ui, vi, *JAPAN)
                year_lons.append(lns)
                year_lats.append(lts)
                ax.plot(
                    lns,
                    lts,
                    color=color,
                    linewidth=0.5,
                    alpha=0.16,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            except Exception:  # noqa: S112
                continue

        if not year_lons:
            continue

        mean_lons = np.nanmean(year_lons, axis=0)
        mean_lats = np.nanmean(year_lats, axis=0)

        ax.plot(
            mean_lons,
            mean_lats,
            color=color,
            linewidth=2.6 if is_winter else 1.2,
            alpha=0.92 if is_winter else 0.55,
            transform=ccrs.PlateCarree(),
            zorder=5,
            solid_capstyle="round",
        )

        # Origin marker at 5-day upstream end
        ax.scatter(
            mean_lons[-1],
            mean_lats[-1],
            s=45,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )

        # Month label — offset varies to reduce crowding
        dx = -2.5 if mean_lons[-1] < 110 else 0
        dy = 2.0 if month_idx % 2 == 0 else -2.8
        mt = ax.text(
            mean_lons[-1] + dx,
            mean_lats[-1] + dy,
            MONTHS[month_idx],
            transform=ccrs.PlateCarree(),
            fontsize=8.5,
            color=color,
            weight="bold",
            ha="center",
            zorder=8,
        )
        stroke(mt, 2.2)

    # Japan endpoint marker
    ax.scatter(
        *JAPAN,
        s=140,
        marker="*",
        color=COLORS["red"],
        edgecolor="white",
        linewidth=1.1,
        transform=ccrs.PlateCarree(),
        zorder=9,
    )
    jt = ax.text(
        JAPAN[0] + 1.4,
        JAPAN[1] + 2.2,
        "Japan",
        transform=ccrs.PlateCarree(),
        fontsize=12.5,
        color=COLORS["ink"],
        weight="bold",
        zorder=9,
    )
    stroke(jt, 3.2)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=p_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.915, 0.20, 0.018, 0.55])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P-WIND (m/s)", fontsize=9, color=COLORS["muted"])
    cb.ax.tick_params(labelsize=8, colors=COLORS["muted"])

    # Annotation box explaining the level choice
    ax.text(
        0.015,
        0.055,
        "850 hPa level (low troposphere) — more relevant to aerosol transport\n"
        "than the 300 hPa jet. At 5-15 m/s, 5 days reaches NE China / Mongolia.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS["muted"],
        zorder=10,
        bbox={
            "boxstyle": "square,pad=0.35",
            "facecolor": COLORS["panel"],
            "edgecolor": COLORS["grid"],
            "alpha": 0.88,
        },
    )

    ax.set_title(
        "5-day 850 hPa back-trajectories from Japan  -  1996-2006 ensemble mean",
        loc="left",
        pad=10,
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["ink"],
    )

    fig.savefig(
        str(OUT_DIR / "back-trajectories.png"),
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)
    print("  Saved: back-trajectories.png")


# ── figure 2: AERONET AOD ─────────────────────────────────────────────────────


def figure_aod(aod_results: dict[str, pd.DataFrame], p_clim: np.ndarray) -> None:
    """Monthly AERONET AOD at three swath stations with normalized P-WIND overlaid."""
    available = [s for s in STATIONS if s[0] in aod_results]
    if not available:
        print("  No AERONET data — skipping AOD figure.")
        return

    n = len(available)
    x = np.arange(1, 13)
    p_norm_vals = (p_clim - p_clim.min()) / (p_clim.max() - p_clim.min())

    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.8), sharey=False)
    fig.patch.set_facecolor(COLORS["bg"])
    if n == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.80, wspace=0.38)

    for ax, (api, disp, lon, lat, role) in zip(axes, available, strict=False):
        clim = aod_results[api]
        color = ROLE_COLOR[role]

        aod_mean = np.full(12, np.nan)
        aod_std = np.full(12, np.nan)
        for _, row in clim.iterrows():
            m = int(row["month"]) - 1
            if 0 <= m < 12:
                aod_mean[m] = row["aod_mean"]
                if pd.notna(row.get("aod_std")):
                    aod_std[m] = row["aod_std"]

        total_obs = int(clim["n_obs"].sum()) if "n_obs" in clim.columns else 0

        ax.set_facecolor(COLORS["panel"])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.tick_params(colors=COLORS["muted"], labelsize=10)

        # High-P-WIND months shaded
        ax.axvspan(0.5, 3.5, color="#dbe9ef", alpha=0.55, zorder=0, label="_")
        ax.axvspan(10.5, 12.5, color="#dbe9ef", alpha=0.55, zorder=0, label="_")

        # Shaded error band behind bars
        valid = ~np.isnan(aod_mean) & ~np.isnan(aod_std)
        if valid.any():
            ax.fill_between(
                x[valid],
                aod_mean[valid] - aod_std[valid],
                aod_mean[valid] + aod_std[valid],
                alpha=0.18,
                color=color,
                zorder=1,
                linewidth=0,
            )

        ax.bar(
            x, aod_mean, color=color, alpha=0.78, width=0.70, edgecolor="none", zorder=2
        )

        # P-WIND on right axis
        ax2 = ax.twinx()
        ax2.plot(
            x,
            p_norm_vals,
            color=COLORS["deep_blue"],
            linewidth=2.2,
            linestyle="--",
            alpha=0.70,
            zorder=3,
        )
        ax2.set_ylim(-0.12, 1.30)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(colors=COLORS["deep_blue"], labelsize=9)

        if ax is axes[-1]:
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(
                ["P-WIND\nmin", "P-WIND\nmax"], fontsize=8.5, color=COLORS["deep_blue"]
            )
            ax2.spines["right"].set_color(COLORS["grid"])
        else:
            ax2.set_yticks([])
            ax2.spines["right"].set_visible(False)

        wavelength = (
            clim["wavelength"].iloc[0] if "wavelength" in clim.columns else "AOD"
        )

        # Title with role color band
        role_label = {
            "source": "upstream source",
            "transit": "transit / filter",
            "validation": "Japan validation",
        }.get(role, "")
        ax.set_title(
            disp,
            fontsize=12.5,
            weight="bold",
            color=COLORS["ink"],
            pad=6,
            linespacing=1.35,
        )
        ax.text(
            0.5,
            1.04,
            role_label,
            transform=ax.transAxes,
            fontsize=9.5,
            color=color,
            ha="center",
            weight="bold",
        )
        ax.text(
            0.5,
            0.965,
            f"{lat:.1f}°N · {lon:.1f}°E · n={total_obs} days",
            transform=ax.transAxes,
            fontsize=8.5,
            color=COLORS["muted"],
            ha="center",
            va="top",
        )

        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTHS, fontsize=9, rotation=40, ha="right")
        if ax is axes[0]:
            ax.set_ylabel(f"AOD ({wavelength})", fontsize=10, color=COLORS["muted"])
        ax.grid(axis="y", alpha=0.28, color=COLORS["grid"])
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Seasonal aerosol optical depth at swath stations",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.03,
        0.925,
        "AERONET Level 2.0 daily climatology.  "
        "Dashed = normalized P-WIND (300 hPa).  "
        "Shading = ±1 SD.  Blue = high-P-WIND months.",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    fig.savefig(
        str(OUT_DIR / "aeronet-seasonal-aod.png"),
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)
    print("  Saved: aeronet-seasonal-aod.png")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ensure_dirs()

    print("── Loading NCEP winds (narrow, existing cache) ─────────────────")
    narrow_ds = load_narrow_winds()
    p_clim = p_wind_climatology(narrow_ds)

    print("── Loading NCEP 850 hPa winds (wide domain for trajectories) ───")
    wide_ds = load_wide_winds()

    print("── Back-trajectory figure ───────────────────────────────────────")
    figure_trajectories(narrow_ds, wide_ds)

    print("── AERONET AOD ──────────────────────────────────────────────────")
    aod_results: dict[str, pd.DataFrame] = {}
    for api_name, *_ in STATIONS:
        print(f"  {api_name}…")
        result = fetch_aeronet(api_name)
        if result is not None:
            aod_results[api_name] = result
        time.sleep(0.5)

    print(f"\n  {len(aod_results)}/{len(STATIONS)} stations returned data")
    figure_aod(aod_results, p_clim)

    print("\nDone.")


if __name__ == "__main__":
    main()
