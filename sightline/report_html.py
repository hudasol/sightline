"""A single static HTML report built only from saved result files (`sightline report-html`).

It opens by double-clicking: one self-contained file, CSS inline, the stress plot embedded as a data URI,
no JavaScript, no external library, no network request. It shows:
- the held-out bar table, with pass or fail per bar;
- the stress plot and table;
- the tracker ablation;
- one corridor event from approach to close.

Every value is read from `results/`; nothing is computed that the saved files do not already hold, apart from
formatting.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from sightline.report import bar_rows
from sightline.timeline import timeline

# the walkthrough event, fixed in advance: the first matched event of the ordinary held-out clip in docs/DEMO.md
DEFAULT_CLIP = "IC-acs_eg-clip_00001_00061"
DEFAULT_EVENT = 1

CSS = """
:root { --bg:#fbfbfa; --fg:#151515; --muted:#5d5c58; --line:#dedcd6; --pass:#16794a; --fail:#b3261e; --card:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141413; --fg:#ecebe7; --muted:#a7a59e; --line:#3a3936; --pass:#5fcf95; --fail:#ff8a80; --card:#1d1d1b; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 36px 0 8px; }
p.lead, .muted { color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 13px; }
.pass { color: var(--pass); font-weight: 700; } .fail { color: var(--fail); font-weight: 700; }
img { max-width: 100%; height: auto; border-radius: 6px; background: #fff; }
pre { white-space: pre-wrap; font: 13px/1.45 ui-monospace, Consolas, monospace; margin: 0; }
code { font: 13px ui-monospace, Consolas, monospace; }
footer { margin-top: 40px; color: var(--muted); font-size: 13px; }
"""


def _e(v) -> str:
    return html.escape(str(v))


def _f(v, nd=3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _table(header: list[str], rows: list[list], cls_col: int | None = None) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{_e(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            cls = ""
            if i == cls_col:
                cls = ' class="pass"' if c == "PASS" else (' class="fail"' if c == "FAIL" else "")
            cells.append(f"<td{cls}>{_e(c)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) + "</tbody></table>"


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def build_html(results_dir: Path, clip: str = DEFAULT_CLIP, event: int = DEFAULT_EVENT) -> str:
    rd = Path(results_dir)
    final = _load(rd / "final_metrics.json")
    if final is None:
        raise FileNotFoundError(f"{rd / 'final_metrics.json'} is missing: run the held-out evaluation first")
    runs = _load(rd / "runtime" / "heldout_offline" / "summary.json")
    note = "recorded check used results/runtime_report.json"
    if runs:
        t = next(iter(runs.values()))["throughput_fps"]
        note += f"; separate timing checks: median of 3 runs {t['median']:.2f} fps (range {t['min']:.2f}-{t['max']:.2f})"
    parts = [f"<h2>The held-out bar</h2><p class='muted'>From <code>results/final_metrics.json</code> "
             f"(its <code>bar</code> block, also <code>results/bar_check.json</code>). Detection judged on: "
             f"{_e(final['bar']['detection_judged_on'])}. Detector <code>{_e(final['detector']['name'])}</code> at "
             f"threshold {_e(final['detector']['score_threshold'])}, detection interval k = {_e(final['detect_every'])}. "
             "PASS and FAIL are the mechanical check as recorded by the single evaluation.</p>",
             "<div class='card'>" + _table(["Bar check", "Measured", "Result"], bar_rows(final, note), cls_col=2) + "</div>"]

    stress = _load(rd / "stress" / "stress_results.json")
    png = rd / "plots" / "stress_curves.png"
    parts.append("<h2>Stress suite (frozen system, nothing retuned)</h2>")
    if png.is_file():
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        parts.append(f"<div class='card'><img alt='IDF1 and event recall against degradation level, per degradation' "
                     f"src='data:image/png;base64,{b64}'></div>")
    if stress:
        rows = [[r["degradation"], r["level"], _f(r["detection_recall"]), _f(r["idf1"]), _f(r["event_recall"]),
                 _f(r["false_clear_rate"])] for r in stress]
        parts.append("<p class='muted'>From <code>results/stress/stress_results.json</code>. Only the detector sees "
                     "degraded frames; the replay computing tracking, events and health reads the clean frames, so "
                     "this does not show whether health flags degraded video.</p>")
        parts.append("<div class='card'>" + _table(["Degradation", "Level", "Detection recall", "IDF1", "Event recall",
                                                    "False-CLEAR rate"], rows) + "</div>")

    ab = _load(rd / "ablation" / "ablation_heldout.json")
    if ab:
        rows = [[n, _f(v["idf1"]), _f(v["id_switches"]), _f(v["event_precision"]), _f(v["event_recall"]),
                 _f(v["events_unresolved"])] for n, v in ab["variants"].items()]
        parts.append("<h2>Tracker ablation (held-out, cached detections, analysis only)</h2>"
                     "<p class='muted'>From <code>results/ablation/ablation_heldout.json</code>.</p>"
                     "<div class='card'>" + _table(["Variant", "IDF1", "ID switches", "Event precision", "Event recall",
                                                    "Unresolved"], rows) + "</div>")

    try:
        lines = timeline(rd, clip, event, _clip_fps(rd, clip))
        parts.append(f"<h2>One corridor event, from approach to close</h2><p class='muted'>Clip <code>{_e(clip)}</code>, "
                     f"event {event}, chosen in advance (the first matched event of the ordinary clip in "
                     "<code>docs/DEMO.md</code>). Built from <code>results/per_clip/bytetrack/</code> and the saved "
                     "tracks by the same code as <code>python -m sightline timeline</code>. No frame is shown. "
                     "The approach and last-reported lines need <code>results/tracks/</code>, which is git-ignored; "
                     "without it they are omitted.</p>"
                     "<div class='card'><pre>" + _e("\n".join(lines)) + "</pre></div>")
    except (KeyError, FileNotFoundError) as exc:
        parts.append(f"<h2>One corridor event</h2><p class='muted'>Not available: {_e(exc)}</p>")

    code = final.get("code") or {}
    head = (f"<h1>Sightline: held-out results</h1><p class='lead'>Advisory corridor perception. The held-out set "
            f"(lock <code>{_e(final.get('locked_manifest_sha256', '')[:12])}</code>) was evaluated once with the "
            f"config tuned on val only (<code>{_e(final.get('frozen_config_sha256', '')[:12])}</code>), at commit "
            f"<code>{_e((code.get('git_commit') or 'n/a')[:10])}</code>. Every value below is read from a saved "
            "file under <code>results/</code>. No frame of MOT17 or IndoorCrowd is included.</p>")
    foot = ("<footer>Generated by <code>python -m sightline report-html</code> from <code>results/</code>. "
            "Do not edit by hand. Interpretation: <code>docs/EVALUATION.md</code>; all tables: "
            "<code>docs/RESULTS.md</code>.</footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Sightline held-out report</title><style>{CSS}</style></head><body><main>"
            + head + "".join(parts) + foot + "</main></body></html>\n")


def _clip_fps(rd: Path, clip: str) -> float:
    """The clip's fps from the saved ingest records under results/ingest/ (never from data/)."""
    rec = _load(rd / "ingest" / f"{clip}.ingest.json")
    if rec and rec.get("fps"):
        return float(rec["fps"])
    imp = _load(rd / "ingest" / "indoorcrowd_import.json") or {}
    for c in imp.get("clips", []):
        if c.get("id") == clip:
            return float(c["fps"])
    raise KeyError(f"no saved frame rate for {clip} under {rd / 'ingest'}")
