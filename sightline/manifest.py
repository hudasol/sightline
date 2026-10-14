"""Clip manifests, the leakage check, and locking the held-out set.

The split unit is the whole scene / camera / recording session, never frames.
MOT17 ships each scene three times (DPM / FRCNN / SDP): those are ONE scene and
are normalised to a single base scene id here.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from sightline.ingest import IMAGE_EXTS, sha256_file

CONDITIONS = ("clear", "occlusion_crowding", "camera_motion_blur", "low_light_compression")
# what a clip is scored on. A moving-camera clip drops "events": its image-space corridor polygon
# does not stay on the floor, so corridor events (and their derived labels) mean nothing there.
METRICS = ("detection", "tracking", "events")
DATASETS = ("own", "mot17", "synthetic")
REQUIRED_FIELDS = ("id", "source", "scene", "camera", "session", "conditions", "corridor", "gt", "license")


class ManifestError(ValueError):
    pass


@dataclass
class Clip:
    id: str
    source: str
    scene: str
    camera: str
    session: str
    conditions: list[str]
    corridor: str
    gt: str
    license: str
    negative: bool = False
    gt_events: str | None = None
    fps: float | None = None
    metrics: tuple[str, ...] = METRICS
    raw: dict = field(default_factory=dict)

    def scored_on(self, metric: str) -> bool:
        return metric in self.metrics

    @property
    def dataset(self) -> str:
        """"own" or "mot17". The detection bar is judged on "own" clips (PLAN.md amendment A1).
        An explicit "dataset" field wins; otherwise a MOT scene id means mot17."""
        explicit = self.raw.get("dataset")
        if explicit:
            return str(explicit)
        return "mot17" if self.base_scene.startswith("MOT-") else "own"

    @property
    def base_scene(self) -> str:
        m = re.search(r"MOT(?:16|17|20)-(\d+)", self.scene) or re.search(r"MOT(?:16|17|20)-(\d+)", self.id)
        return f"MOT-{m.group(1)}" if m else self.scene


@dataclass
class Manifest:
    split: str
    clips: list[Clip]
    path: str = ""


def load_manifest(path: str | Path) -> Manifest:
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{p} is not valid JSON: {exc}") from exc
    if "clips" not in data or not isinstance(data["clips"], list):
        raise ManifestError(f"{p}: needs a 'clips' list")
    clips, seen = [], set()
    for i, c in enumerate(data["clips"]):
        missing = [k for k in REQUIRED_FIELDS if k not in c]
        if missing:
            raise ManifestError(f"{p}: clip #{i} is missing {missing}")
        if c["id"] in seen:
            raise ManifestError(f"{p}: duplicate clip id {c['id']!r}")
        seen.add(c["id"])
        unknown = [t for t in c["conditions"] if t not in CONDITIONS]
        if unknown:
            raise ManifestError(f"{p}: clip {c['id']!r} has unknown conditions {unknown}; use {list(CONDITIONS)}")
        metrics = c.get("metrics", list(METRICS))
        if (not isinstance(metrics, list) or not metrics or len(set(metrics)) != len(metrics)
                or any(m not in METRICS for m in metrics)):
            raise ManifestError(f"{p}: clip {c['id']!r} metrics must be a non-empty subset of {list(METRICS)}, got {metrics!r}")
        if c.get("dataset") is not None and c["dataset"] not in DATASETS:
            raise ManifestError(f"{p}: clip {c['id']!r} dataset must be one of {list(DATASETS)}, got {c['dataset']!r}")
        if c.get("negative") and "events" not in metrics:
            raise ManifestError(f"{p}: clip {c['id']!r} is negative but not scored on events; a negative clip only means something for events")
        clips.append(Clip(
            id=c["id"], source=c["source"], scene=c["scene"], camera=c["camera"], session=c["session"],
            conditions=list(c["conditions"]), corridor=c["corridor"], gt=c["gt"], license=c["license"],
            negative=bool(c.get("negative", False)), gt_events=c.get("gt_events"), fps=c.get("fps"),
            metrics=tuple(m for m in METRICS if m in metrics), raw=c,
        ))
    return Manifest(split=str(data.get("split", "")), clips=clips, path=str(p))


def split_check(val: Manifest, test: Manifest) -> list[str]:
    """Return the list of leakage problems. Empty means the split is clean."""
    problems = []
    for label, key in (("clip id", lambda c: c.id), ("source path", lambda c: str(Path(c.source))),
                       ("scene", lambda c: c.scene), ("base scene", lambda c: c.base_scene),
                       ("camera", lambda c: c.camera), ("recording session", lambda c: c.session)):
        shared = {key(c) for c in val.clips} & {key(c) for c in test.clips}
        for s in sorted(shared):
            problems.append(f"val and test share {label}: {s!r}")
    return problems


def coverage_check(test: Manifest, min_clips: int = 12, difficult: tuple[str, ...] = (
        "occlusion_crowding", "camera_motion_blur", "low_light_compression")) -> dict:
    present = sorted({t for c in test.clips for t in c.conditions})
    diff_present = [t for t in difficult if t in present]
    ev = [c for c in test.clips if c.scored_on("events")]
    ev_present = sorted({t for c in ev for t in c.conditions})
    return {
        "n_clips": len(test.clips),
        "meets_min_clips": len(test.clips) >= min_clips,
        "conditions_present": present,
        "difficult_conditions_present": diff_present,
        "meets_three_difficult": len(diff_present) >= 3,
        "negative_clips": sum(1 for c in test.clips if c.negative),
        "has_negative_set": any(c.negative for c in test.clips),
        # what the EVENT metrics actually rest on, once detection/tracking-only clips are set aside
        "clips_scored_on": {m: sum(1 for c in test.clips if c.scored_on(m)) for m in METRICS},
        "event_conditions_present": ev_present,
        "event_difficult_conditions_present": [t for t in difficult if t in ev_present],
    }


def hash_path(path: Path) -> str:
    """Content hash of a video file or an image-sequence folder."""
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        h = hashlib.sha256()
        for f in sorted(p for p in path.rglob("*") if p.is_file() and (p.suffix.lower() in IMAGE_EXTS or p.name == "seqinfo.ini")):
            h.update(str(f.relative_to(path)).encode())
            h.update(sha256_file(f).encode())
        return h.hexdigest()
    raise ManifestError(f"cannot hash missing path: {path}")


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


_LOCK_META = ("manifest_sha256", "locked_at_utc", "git_commit")


def _lock_body(locked: dict) -> dict:
    return {k: v for k, v in locked.items() if k not in _LOCK_META}


def lock_manifest(manifest_path: str | Path, out_path: str | Path, root: str | Path = ".",
                  eval_config: str | Path | None = None) -> dict:
    """Hash every clip, its labels and its corridor, and write the locked held-out manifest.

    The corridor is hashed because MOT17 event labels are DERIVED from the polygon:
    editing a test polygon after locking would silently change the labels. The
    eval config (label rules and the bar) is hashed for the same reason.

    Tuning records the hash of this file, which is how an evaluation can prove the
    held-out set was fixed BEFORE the final tuning pass.
    """
    root = Path(root)
    man = load_manifest(manifest_path)
    entries = []
    for c in man.clips:
        entry = dict(c.raw)
        entry["source_sha256"] = hash_path(root / c.source)
        entry["gt_sha256"] = sha256_file(root / c.gt)
        entry["corridor_sha256"] = sha256_file(root / c.corridor)
        if c.gt_events:
            entry["gt_events_sha256"] = sha256_file(root / c.gt_events)
        entries.append(entry)
    coverage = coverage_check(man)
    complete = coverage["meets_min_clips"] and coverage["meets_three_difficult"] and coverage["has_negative_set"]
    # a DRAFT lock is allowed (it exercises the protocol early) but `tune` refuses it
    body = {"version": 1, "split": man.split, "clips": entries, "coverage": coverage,
            "status": "final" if complete else "draft"}
    if eval_config is not None:
        body["eval_config"] = str(eval_config)
        body["eval_config_sha256"] = sha256_file(root / eval_config)
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    locked = {**body, "manifest_sha256": digest, "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "git_commit": _git_commit(root)}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(locked, indent=1))
    return locked


def require_final_lock(lock_path: str | Path, root: str | Path = ".") -> dict:
    """The locked held-out set tuning may rely on: complete (status final) and unchanged since locking."""
    lp = Path(lock_path)
    if not lp.is_file():
        raise ManifestError(f"locked held-out manifest {lp} does not exist: run `lock-manifest` BEFORE tuning")
    locked = json.loads(lp.read_text())
    if locked.get("status") != "final":
        cov = locked.get("coverage") or {}
        raise ManifestError(
            f"{lp} is a DRAFT lock (status={locked.get('status', 'missing')!r}, coverage={json.dumps(cov)}). "
            "Complete the held-out set (>=12 clips, 3 difficult conditions, a negative set), re-run "
            "`lock-manifest`, and only then tune.")
    problems = verify_lock(lp, root)
    if problems:
        raise ManifestError("held-out set changed since locking, refusing:\n  " + "\n  ".join(problems))
    return locked


def verify_lock(lock_path: str | Path, root: str | Path = ".") -> list[str]:
    """Recompute every hash. Any difference means the held-out set changed after locking."""
    root = Path(root)
    lp = Path(lock_path)
    if not lp.is_file():
        raise ManifestError(f"locked manifest not found: {lp}")
    locked = json.loads(lp.read_text())
    problems = []
    body = _lock_body(locked)
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != locked.get("manifest_sha256"):
        problems.append("locked manifest content does not match its recorded hash")
    if locked.get("eval_config"):
        try:
            if sha256_file(root / locked["eval_config"]) != locked.get("eval_config_sha256"):
                problems.append(f"{locked['eval_config']}: evaluation rules changed since locking")
        except FileNotFoundError as exc:
            problems.append(f"eval config: {exc}")
    for c in locked["clips"]:
        try:
            if hash_path(root / c["source"]) != c["source_sha256"]:
                problems.append(f"{c['id']}: clip data changed since locking")
            if sha256_file(root / c["gt"]) != c["gt_sha256"]:
                problems.append(f"{c['id']}: labels changed since locking")
            if "corridor_sha256" in c and sha256_file(root / c["corridor"]) != c["corridor_sha256"]:
                problems.append(f"{c['id']}: corridor polygon changed since locking")
            if c.get("gt_events") and sha256_file(root / c["gt_events"]) != c.get("gt_events_sha256"):
                problems.append(f"{c['id']}: event labels changed since locking")
        except (ManifestError, FileNotFoundError) as exc:
            problems.append(f"{c['id']}: {exc}")
    return problems
