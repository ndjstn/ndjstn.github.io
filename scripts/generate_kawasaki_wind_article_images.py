from __future__ import annotations

import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
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
    "bg": "#f7f3e9",
    "panel": "#fffdf8",
    "ink": "#19221f",
    "muted": "#66736d",
    "grid": "#d9d3c6",
    "water": "#dce9ec",
    "land": "#eee3cf",
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


def save(fig: plt.Figure, name: str, *, pad: float = 0.15) -> None:
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def add_pacific_context(ax) -> None:
    ax.set_extent([110, 255, 15, 60], crs=ccrs.PlateCarree())
    ax.set_facecolor(COLORS["water"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=COLORS["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, color=COLORS["coast"], zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, color="#9b9a90", alpha=0.7, zorder=4)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.55,
        color="#9fb0ac",
        alpha=0.45,
        linestyle="-",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8.5, "color": COLORS["muted"]}
    gl.ylabel_style = {"size": 8.5, "color": COLORS["muted"]}


def add_locations(ax) -> None:
    for label, (lon, lat) in LOCATIONS.items():
        ax.scatter(lon, lat, s=42, color=COLORS["red"], edgecolor="white", linewidth=0.8, transform=ccrs.PlateCarree(), zorder=7)
        dx = 3.2 if label != "San Diego" else -18
        dy = 1.4 if label != "Hawaii" else -3.0
        ax.text(
            lon + dx,
            lat + dy,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=9.5,
            color=COLORS["ink"],
            weight="bold",
            zorder=8,
        )


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


def zscore(values: xr.DataArray) -> np.ndarray:
    arr = values.to_numpy()
    return (arr - arr.mean()) / arr.std()


def image_hero(ds: xr.Dataset) -> None:
    wind = jan_300_wind(ds)
    speed = np.hypot(wind["u"], wind["v"])
    q_lon = wind["lon"].to_numpy()[::4]
    q_lat = wind["lat"].to_numpy()[::3]
    q_x, q_y = np.meshgrid(q_lon, q_lat)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.045, 0.14, 0.61, 0.64], projection=ccrs.PlateCarree(central_longitude=180))
    add_pacific_context(ax)
    levels = np.arange(10, 58, 4)
    filled = ax.contourf(
        wind["lon"],
        wind["lat"],
        speed,
        levels=levels,
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
        scale=850,
        width=0.0022,
        alpha=0.82,
        zorder=5,
    )
    ax.plot([140, 240], [35, 35], color=COLORS["red"], linewidth=2.4, transform=ccrs.PlateCarree(), zorder=6)
    ax.text(177, 36.8, "P-WIND line", color=COLORS["red"], fontsize=10.5, weight="bold", transform=ccrs.PlateCarree(), zorder=7)
    add_locations(ax)
    ax.set_title("January 300 hPa winds across the North Pacific", loc="left", fontsize=16, fontweight="bold", pad=10)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.036, pad=0.016)
    cbar.outline.set_visible(False)
    cbar.set_label("wind speed (m/s)", color=COLORS["muted"], labelpad=8)

    side = fig.add_axes([0.765, 0.22, 0.19, 0.46])
    p = p_wind(ds)
    nw = nw_wind_japan(ds)
    x = np.arange(1, 13)
    side.plot(x, zscore(p), color=COLORS["blue"], linewidth=2.6, label="Pacific westerlies")
    side.plot(x, zscore(nw), color=COLORS["green"], linewidth=2.6, label="Japan NW component")
    side.axhline(0, color="#bdb5a7", linewidth=1)
    side.axvspan(1, 3, color="#dbe9ef", alpha=0.65, zorder=0)
    side.axvspan(11, 12, color="#dbe9ef", alpha=0.65, zorder=0)
    side.set_xticks(x, MONTHS, rotation=45, ha="right")
    side.set_ylabel("standardized")
    side.set_title("Same seasonal turn", loc="left", fontsize=14, fontweight="bold")
    side.grid(axis="y", alpha=0.6)
    side.grid(axis="x", visible=False)
    side.legend(frameon=False, fontsize=9.2, loc="lower left")

    fig.text(0.045, 0.92, "When a disease time series starts looking like a weather map", fontsize=22, weight="bold", color=COLORS["ink"])
    fig.text(
        0.045,
        0.875,
        "The paper's hypothesis depends on a seasonal North Pacific circulation pattern. The wind fields are public; the clinical case records are not a Kaggle-style dataset.",
        fontsize=11,
        color=COLORS["muted"],
    )
    fig.text(0.05, 0.025, "Source: NOAA PSL NCEP/NCAR Reanalysis 1, monthly means, 1996-2006", fontsize=9.5, color=COLORS["muted"])
    fig.savefig(OUT_DIR / "hero.png", dpi=180)
    plt.close(fig)


def image_seasonal_indices(ds: xr.Dataset) -> None:
    p = p_wind(ds)
    nw = nw_wind_japan(ds)
    x = np.arange(1, 13)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.1), sharex=True)
    specs = [
        (axes[0], p, "Pacific Zonal Wind Index at 300 hPa", COLORS["blue"], "Mean east-west wind along 35N, 140E-240E"),
        (axes[1], nw, "Japan northwesterly component at 850 hPa", COLORS["green"], "Area mean over 30-45N, 130-145E"),
    ]

    for ax, values, title, color, subtitle in specs:
        ax.axvspan(1, 3, color="#dbe9ef", alpha=0.7, zorder=0)
        ax.axvspan(11, 12, color="#dbe9ef", alpha=0.7, zorder=0)
        ax.plot(x, values, color=color, linewidth=2.8)
        ax.scatter(x, values, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_title(title, loc="left", pad=12, fontsize=15, fontweight="bold")
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=COLORS["muted"], fontsize=10.2)
        ax.set_ylabel("m/s")
        ax.grid(axis="y", alpha=0.62)
        ax.grid(axis="x", visible=False)

    axes[-1].set_xticks(x, MONTHS)
    axes[-1].set_xlabel("month")
    fig.suptitle("The wind indices turn upward in the cool season", x=0.08, y=0.985, ha="left", fontsize=18, fontweight="bold", color=COLORS["ink"])
    fig.text(
        0.08,
        0.936,
        "These are atmospheric indices only. They do not include patient records or prove a cause.",
        color=COLORS["muted"],
        fontsize=10.8,
    )
    save(fig, "seasonal-wind-indices.png")


def image_monthly_animation(ds: xr.Dataset) -> None:
    clim = monthly_climatology(ds, 300)
    speed = np.hypot(clim["u"], clim["v"])
    levels = np.arange(6, 58, 4)
    q_lon = clim["lon"].to_numpy()[::5]
    q_lat = clim["lat"].to_numpy()[::4]
    q_x, q_y = np.meshgrid(q_lon, q_lat)

    fig = plt.figure(figsize=(11.5, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))

    def draw(month_index: int):
        ax.clear()
        add_pacific_context(ax)
        month = month_index + 1
        u = clim["u"].sel(month=month)
        v = clim["v"].sel(month=month)
        spd = speed.sel(month=month)
        ax.contourf(clim["lon"], clim["lat"], spd, levels=levels, cmap="YlGnBu", extend="max", transform=ccrs.PlateCarree(), zorder=1)
        ax.quiver(
            q_x,
            q_y,
            u.to_numpy()[::4, ::5],
            v.to_numpy()[::4, ::5],
            transform=ccrs.PlateCarree(),
            color="#14384d",
            scale=850,
            width=0.0025,
            alpha=0.82,
            zorder=5,
        )
        ax.plot([140, 240], [35, 35], color=COLORS["red"], linewidth=2.1, transform=ccrs.PlateCarree(), zorder=6)
        add_locations(ax)
        ax.set_title(f"300 hPa North Pacific winds: {MONTHS[month_index]}", loc="left", fontsize=16, fontweight="bold", pad=10)
        ax.text(
            0.02,
            0.04,
            "Monthly climatology, 1996-2006",
            transform=ax.transAxes,
            fontsize=9.8,
            color=COLORS["muted"],
            bbox={"facecolor": COLORS["panel"], "edgecolor": COLORS["grid"], "boxstyle": "round,pad=0.25"},
            zorder=8,
        )

    ani = animation.FuncAnimation(fig, draw, frames=12, interval=520, repeat=True)
    ani.save(OUT_DIR / "monthly-pacific-winds.gif", writer=animation.PillowWriter(fps=2), dpi=135)
    plt.close(fig)


def image_analysis_boundary() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.92, "A careful reading separates evidence from claim", fontsize=18, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.865,
        "The paper connects disease timing with atmospheric circulation. The reproducible part here is the circulation, not the private clinical records.",
        fontsize=11,
        color=COLORS["muted"],
    )

    columns = [
        ("Observed in paper", 0.08, COLORS["blue"], ["KD admission records", "Seasonal and epidemic peaks", "Japan, Hawaii, San Diego"]),
        ("Reproduced here", 0.39, COLORS["green"], ["NOAA wind fields", "P-WIND and NW-WIND indices", "Monthly circulation maps"]),
        ("Hypothesis", 0.70, COLORS["gold"], ["Wind-borne trigger is plausible", "Aerosols are worth testing", "Causation remains open"]),
    ]

    for title, x0, color, bullets in columns:
        ax.add_patch(Rectangle((x0, 0.28), 0.22, 0.44, facecolor=COLORS["panel"], edgecolor="#ded6c9", linewidth=1.2))
        ax.add_patch(Rectangle((x0, 0.69), 0.22, 0.03, facecolor=color, edgecolor="none"))
        ax.text(x0 + 0.02, 0.64, title, fontsize=13.2, fontweight="bold", color=COLORS["ink"])
        for i, bullet in enumerate(bullets):
            ax.text(x0 + 0.025, 0.565 - i * 0.105, f"- {bullet}", fontsize=11.1, color=COLORS["muted"])

    arrows = [
        ((0.30, 0.50), (0.39, 0.50), "compare timing"),
        ((0.61, 0.50), (0.70, 0.50), "form testable mechanism"),
    ]
    for start, end, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, linewidth=1.8, color="#8b8a82"))
        ax.text((start[0] + end[0]) / 2, 0.455, label, ha="center", fontsize=9.8, color=COLORS["muted"])

    ax.add_patch(Rectangle((0.08, 0.12), 0.84, 0.08, facecolor="#f5e2dc", edgecolor="#d6b4aa", linewidth=1.0))
    ax.text(
        0.10,
        0.153,
        "Guardrail: this workflow can strengthen a mechanistic hypothesis, but it cannot identify the causal agent by itself.",
        fontsize=11.2,
        color=COLORS["ink"],
        va="center",
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
