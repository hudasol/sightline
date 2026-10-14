# Manifests

`val.json` (tuning) and `test.json` (held-out) are the source of truth for which
clip is in which split. Paths are relative to the repo root.

## Current split (decided 2026-10-12, before any metric existed)

| Split | MOT17 base scenes (FRCNN copy only) |
|---|---|
| val | 02, 04, 05, 11, 13 |
| test | 09, 10 |

`split-check` on these two files passes but reports the held-out coverage gaps.

Before a **final** lock, your own corridor clips must add:
- at least **10 more test clips** (the bar is 12, and MOT17 gives 2)
- at least one test clip tagged **`occlusion_crowding`** (09 and 10 already give
  `clear`, `camera_motion_blur` and `low_light_compression`)
- at least one **negative** test clip (`"negative": true`): people visible, nobody
  enters the corridor

For the **event** metrics to see any hard condition, own clips must also cover
crowding, low light and blur. MOT17-10 is moving-camera, so it counts for detection
and tracking only. Blur for events needs a static camera on fast motion
(`docs/RECORDING_GUIDE.md`).

## What a clip is scored on: `metrics`

`"metrics"` lists the metric families a clip counts for: any non-empty subset of
`["detection", "tracking", "events"]`. Leave it out for the default, all three.

The corridor is an image-space polygon, so it only stays on the floor while the
camera stays still. On a moving camera the system's corridor events, and the
event labels derived from the polygon, mean nothing. Those clips therefore carry
`"metrics": ["detection", "tracking"]` plus a `metrics_note` saying why. Today that
is MOT17-05, 11 and 13 (val) and MOT17-10 (test).

- `aggregate`, `tune` and `interval-eval` pool each family only over its own clips.
  The false-CLEAR safety figure follows events.
- `split-check` reports clips per family, and which difficult conditions the
  **event** metrics actually rest on (`results/split/split_check.json`).
- A negative clip must include `events`.
- The field is part of the clip entry, so it is hashed into the lock.

A handheld own clip has a moving camera too, so it cannot count for events either.
Event coverage of `camera_motion_blur` would need a static camera on a blurred
scene, or a ground-plane corridor.

## Adding an own corridor clip

1. Put the video at `data/own/<id>.mp4` (git-ignored). Use an id like
   `own-<location>-<yyyymmdd>-<nn>`, e.g. `own-corridorA-20261014-03`.
2. Run `python -m sightline ingest data/own/<id>.mp4` to record fps, frames and hash.
3. Draw the corridor polygon on a frame and save it as `configs/corridors/<camera>.yaml`
   (`name`, `frame_size: [w, h]`, `polygon: [[x, y], ...]`).
4. Annotate person boxes in `annotations/<id>/gt.txt` in MOTChallenge format
   (`frame,id,x,y,w,h,1,1,visibility`, frames 1-based). Give every box a real
   visibility value; the in-scope rule needs it (PLAN.md A1, `docs/DATA_CARD.md`).
   From CVAT, export "MOT 1.1" and convert it, stating which frames were annotated:

   ```bash
   python -m sightline import-cvat export.zip --out annotations/<id> --frames 1-600:10
   ```

   This writes `gt.txt`, `annotated_frames.json` and `import_provenance.json`. It
   refuses boxes on frames you did not declare, and warns if every box says
   visibility 1.0. If only some frames were annotated, add
   `"annotated_frames": "annotations/<id>/annotated_frames.json"` to the entry.
   Only those frames are scored. An annotated frame with no boxes is a real
   negative; an unannotated frame is never scored. A sparsely annotated clip must
   have hand-labelled `gt_events` (step 5) if it counts for events.
5. Hand-label events in `annotations/<id>/events.json`:
   `[{"first_inside_frame": 41, "last_inside_frame": 97}, ...]` (feet inside the polygon).
   A negative clip gets `[]`.
6. Add an entry to `val.json` **or** `test.json`:

```json
{
 "id": "own-corridorA-20261014-03",
 "source": "data/own/own-corridorA-20261014-03.mp4",
 "scene": "own-corridorA",
 "camera": "own-corridorA-tripod1",
 "session": "own-corridorA-20261014",
 "conditions": ["occlusion_crowding"],
 "corridor": "configs/corridors/own-corridorA-tripod1.yaml",
 "gt": "annotations/own-corridorA-20261014-03/gt.txt",
 "gt_events": "annotations/own-corridorA-20261014-03/events.json",
 "license": "own footage, written consent on file",
 "negative": false
}
```

**Leakage rule:** `split-check` fails if val and test share a `scene`, `camera` or
`session`. Every clip from one recording session, and every clip from one camera
position, goes into the **same** split. Choose the split per session before you
annotate it, never after seeing results.

## Order (do not change)

1. `split-check` passes.
2. Val-only measurements may run before the lock: `detect-eval`, `interval-eval`,
   `threads`, `benchmark`, `runtime-summary` and `errors --manifest` on val clips.
   They read no test clip and freeze nothing.
3. `lock-manifest configs/manifests/test.json` once the held-out set is **complete**.
   The lock hashes clips, labels, corridor polygons, `metrics` and `configs/eval.yaml`.
   It records `status: final` only when coverage meets the bar (>=12 clips, three
   difficult conditions, a negative set); otherwise `status: draft`.
4. Only then `tune`. `tune` refuses a missing, draft or changed lock, and
   `evaluate` refuses a draft lock.

Re-locking is harmless before tuning. After `tune` has run, any change to the
held-out set means `evaluate` refuses the frozen config, which is the point.
