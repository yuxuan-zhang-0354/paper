"""Generate the main-text D5 mechanism-attribution Figure 8."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/manuscript_data"
OUT = ROOT / "figures/results"
STEM = OUT / "fig8_allocator_stability_zh"

PRESSURE = "#315A8A"
SCALE = "#C4543D"
RAW_BG = "#FAF2EE"
WARP_BG = "#EFF5F8"
GRID = "#DDE3E7"
TEXT = "#2F3941"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def load() -> tuple[dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    methods = {}
    for row in rows("table7_d5_method_summary.csv"):
        methods[row["suite"], row["method"]] = {
            "cycle_rate": 100 * float(row["mean_nonconverged_episode"]),
            "utility": float(row["mean_normalized_utility"]),
        }
    effects = {}
    for row in rows("table7_d5_factorial_effects.csv"):
        if row["stratum"] == "ALL":
            effects[row["suite"], row["effect"]] = {
                "mean": float(row["mean"]),
                "low": float(row["ci_low"]),
                "high": float(row["ci_high"]),
            }
    return methods, effects


def draw(methods, effects) -> plt.Figure:
    fig = plt.figure(figsize=(7.15, 3.15))
    gs = fig.add_gridspec(
        1, 2, width_ratios=(1.12, 1.18), wspace=0.36,
        left=0.14, right=0.985, top=0.86, bottom=0.18,
    )
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    variants = ("V00", "V01", "V10", "V11")
    labels = (
        "V00  raw + 保留/释放",
        "V01  raw + 完整重构",
        "V10  压缩 + 保留/释放",
        "V11  压缩 + 完整重构（P）",
    )
    yv = np.arange(len(variants))[::-1]
    width = 0.30
    pressure = [methods["pressure", method]["cycle_rate"] for method in variants]
    scale = [methods["scale", method]["cycle_rate"] for method in variants]
    ax.axhspan(1.5, 3.5, color=RAW_BG, zorder=-3)
    ax.axhspan(-0.5, 1.5, color=WARP_BG, zorder=-3)
    bars_p = ax.barh(yv + width / 2, pressure, width, color=PRESSURE, edgecolor="white", linewidth=0.7)
    bars_s = ax.barh(yv - width / 2, scale, width, color=SCALE, edgecolor="white", linewidth=0.7)
    for bars, values in ((bars_p, pressure), (bars_s, scale)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                max(value, 0) + 0.35, bar.get_y() + bar.get_height() / 2,
                "0" if value < 0.05 else f"{value:.1f}%",
                ha="left", va="center", fontsize=6.5,
                color=bar.get_facecolor(), weight="bold",
            )
    ax.set_yticks(yv)
    ax.set_yticklabels(labels)
    ax.set_xlabel("出现cycle的场景比例（%）")
    ax.set_xlim(0, 16.5)
    ax.set_xticks((0, 4, 8, 12, 16))
    ax.set_ylim(-0.5, 3.5)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_axisbelow(True)
    ax.legend((bars_p[0], bars_s[0]), ("分配压力域", "规模域"), frameon=False, ncol=2, loc="lower right")
    ax.text(-0.10, 1.06, "（a）四变体的分配稳定性", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.2, weight="bold", color=TEXT)

    effect_names = ("warping", "full_reconstruction", "interaction")
    effect_labels = ("前缀压缩投标", "完整重构", "二者交互")
    y = np.array((2.0, 1.0, 0.0))
    offset = 0.12
    bx.axhspan(1.55, 2.45, color=WARP_BG, zorder=-3)
    bx.axvline(0, color="#737D84", lw=0.9, ls="--", zorder=0)
    for suite, color, dy, label, marker in (
        ("pressure", PRESSURE, offset, "分配压力域", "D"),
        ("scale", SCALE, -offset, "规模域", "o"),
    ):
        for yy, effect in zip(y, effect_names, strict=True):
            value = effects[suite, effect]
            bx.errorbar(
                value["mean"], yy + dy,
                xerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]],
                fmt=marker, ms=5.4, mfc=color, mec="white", mew=0.6,
                color=color, ecolor=color, elinewidth=1.1, capsize=2.4,
                label=label if yy == y[0] else None, zorder=3,
            )
    bx.set_yticks(y)
    bx.set_yticklabels(effect_labels)
    bx.set_xlim(-0.006, 0.022)
    bx.set_xticks((-0.005, 0.000, 0.005, 0.010, 0.015, 0.020))
    bx.set_xlabel("归一化实现效用的场景级因子效应（95% CI）")
    bx.grid(axis="x", color=GRID, lw=0.55)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, ncol=2, loc="lower right")
    bx.text(-0.16, 1.06, "（b）最终实现效用的机制归因", transform=bx.transAxes,
            ha="left", va="bottom", fontsize=8.2, weight="bold", color=TEXT)

    for axis in (ax, bx):
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_linewidth(0.7)
            axis.spines[side].set_color("#7D878E")
    return fig


def main() -> None:
    style()
    methods, effects = load()
    fig = draw(methods, effects)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            STEM.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)

    source_rows = rows("table7_d5_method_summary.csv") + rows("table7_d5_factorial_effects.csv")
    fields = sorted({key for row in source_rows for key in row})
    with (OUT / "fig8_allocator_stability_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()
