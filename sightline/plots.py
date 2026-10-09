"""All plots, regenerated from saved results only. No plot reads a video or a model.

Palette: blue / orange / aqua from the validated categorical order. Aqua is below 3:1
contrast on the light surface, so every series is also direct-labelled and the
underlying numbers are in the results JSON (the table view).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
TRACKER_COLOR = {"iou": ORANGE, "bytetrack": BLUE}
TRACKER_LABEL = {"iou": "IoU baseline", "bytetrack": "ByteTrack-style"}


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _fig(w=7.0, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def plot_detector_sweep(tuning_dir: Path, out: Path) -> Path | None:
    rows = _load(tuning_dir / "detector_threshold_sweep_val.json")
    if not rows:
        return None
    fig, ax = _fig()
    x = [r["threshold"] for r in rows]
    for key, color, label, marker in (("precision", BLUE, "precision", "o"), ("recall", ORANGE, "recall", "s"), ("f1", AQUA, "F1", "^")):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.axhline(0.80, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate("bar 0.80", (x[0], 0.80), xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_ylim(0, 1.02)
    _style(ax, "Person detection vs confidence threshold (validation clips)", "confidence threshold", "score")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _grouped(metrics: dict, section: str, key: str, title: str, ylabel: str, out: Path, trackers=("iou", "bytetrack")) -> Path | None:
    groups = list(metrics[section][trackers[0]])
    if not groups:
        return None
    fig, ax = _fig(7.6, 3.8)
    n = len(trackers)
    width = 0.8 / n
    for i, t in enumerate(trackers):
        vals = [metrics[section][t][g].get(key) or 0.0 for g in groups]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(groups))]
        bars = ax.bar(xs, vals, width=width - 0.05, color=TRACKER_COLOR[t], label=TRACKER_LABEL[t])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={metrics[section][trackers[0]][g]['n_clips']})" for g in groups], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    _style(ax, title, "", ylabel)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_per_condition(final: dict, out_dir: Path) -> list[Path]:
    made = []
    for section, key, title, ylabel, name in (
        ("tracking", "idf1", "IDF1 by condition (held-out)", "IDF1", "idf1_by_condition.png"),
        ("events", "event_recall", "Event recall by condition (held-out)", "event recall", "event_recall_by_condition.png"),
        ("events", "event_precision", "Event precision by condition (held-out)", "event precision", "event_precision_by_condition.png"),
    ):
        p = _grouped(final, section, key, title, ylabel, out_dir / name)
        if p:
            made.append(p)
    return made


def plot_latency(final_per_clip: Path, out: Path) -> Path | None:
    lat = []
    for f in sorted(final_per_clip.glob("bytetrack/*.json")):
        lat += json.loads(f.read_text()).get("event_latency_frames", [])
    if not lat:
        return None
    fig, ax = _fig(6.4, 3.4)
    lo, hi = min(lat), max(lat)
    ax.hist(lat, bins=range(lo, hi + 2), color=BLUE, edgecolor=SURFACE, linewidth=1.5, align="left")
    ax.axvline(5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate("bar: median <= 5", (5, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points", color=MUTED, fontsize=8)
    _style(ax, "Event latency after first frame inside (held-out, matched events)", "frames", "events")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_stress(rows: list[dict], out: Path) -> Path | None:
    if not rows:
        return None
    kinds = list(dict.fromkeys(r["degradation"] for r in rows))
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = [axes] if len(kinds) == 1 else list(axes)
    for ax, kind in zip(axes, kinds):
        sub = [r for r in rows if r["degradation"] == kind]
        x = list(range(len(sub)))
        for key, color, label, marker in (("event_recall", BLUE, "event recall", "o"), ("idf1", ORANGE, "IDF1", "s")):
            y = [r[key] if r[key] is not None else float("nan") for r in sub]
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(label, (x[-1], y[-1]), xytext=(-4, 7 if key == "event_recall" else -12), textcoords="offset points",
                        color=INK, fontsize=7, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["level"]) for r in sub], fontsize=7)
        ax.set_ylim(0, 1.05)
        _style(ax, kind, "level (left = clean)", "score" if ax is axes[0] else "")
    fig.suptitle("Frozen system under controlled degradation (not retuned)", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_runtime(runtime: dict, out: Path, budget_ms: float = 100.0) -> Path | None:
    keys = [("latency_ms_p50", "p50 latency"), ("latency_ms_p95", "p95 latency"),
            ("inference_ms_p50", "p50 inference"), ("inference_ms_p95", "p95 inference")]
    vals = [(lab, runtime.get(k)) for k, lab in keys if runtime.get(k) is not None]
    if not vals:
        return None
    fig, ax = _fig(6.4, 3.2)
    bars = ax.barh([v[0] for v in vals], [v[1] for v in vals], color=BLUE, height=0.55)
    for b, (_l, v) in zip(bars, vals):
        ax.annotate(f"{v:.1f} ms", (v, b.get_y() + b.get_height() / 2), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center", color=INK)
    ax.axvline(budget_ms, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate("10 fps budget", (budget_ms, -0.45), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Per-frame latency on the demo machine", "milliseconds", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def make_all(results_dir: str | Path = "results") -> list[Path]:
    rd = Path(results_dir)
    out = rd / "plots"
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path | None] = [plot_detector_sweep(rd / "tuning", out / "detector_threshold_sweep_val.png")]
    final = _load(rd / "final_metrics.json")
    if final:
        made += plot_per_condition(final, out)
        made.append(plot_latency(rd / "per_clip", out / "event_latency_hist.png"))
    stress = _load(rd / "stress" / "stress_results.json")
    if stress:
        made.append(plot_stress(stress, out / "stress_curves.png"))
    runtime = _load(rd / "runtime_report.json")
    if runtime:
        made.append(plot_runtime(runtime.get("summary", runtime), out / "runtime_latency.png"))
    return [p for p in made if p]
