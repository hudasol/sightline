# Output interface

What a downstream system, such as a robot controller or an operator display, gets from Sightline, and what it must
not assume. Sightline is advisory. Every field below is written by `python -m sightline replay` to the `--out` folder
and was read from the code in `sightline/types.py`, `sightline/events.py` and `sightline/health.py`.

## Files

| File | One line per | Content |
|---|---|---|
| `health.jsonl` | frame | `frame`, `ts`, `state`, `reasons`, `corridor` |
| `events.jsonl` | event change | `kind` (open or close), `frame`, `ts`, and an `event` record |
| `tracks.jsonl` | frame | the reported tracks: `id`, `box`, `predicted`, `score`, `state`, `age`, `hits`, `time_since_update`, `last_seen_ts` |
| `summary.json` | run | final health, frame counts, all events, the unavailable reason if any |
| `runtime.json` | run | throughput and latency percentiles, dropped and late frames, the machine |

## Meaning of the fields

- `frame` is 1 based and refers to the original source. `ts` is seconds of media time from the first frame of the
  source. It is not a wall clock time. A live integration has to add its own receive and publish times.
- `corridor` is `CLEAR`, `OCCUPIED` or `UNKNOWN`. `CLEAR` is written only when health is `OK` and no track, candidate
  or not yet confirmed track may be inside. `UNKNOWN` is written whenever the system cannot tell.
- `state` is `OK`, `DEGRADED` or `UNAVAILABLE`. The `reasons` list names the cause, for example `stream_stalled`,
  `timestamp_gap`, `corrupt_frame`, `low_light`, `blurry_or_low_contrast`, `low_detection_confidence` or
  `stale_detections_<age>s`. The command exits with code 2 when the run ends `UNAVAILABLE`. A usage error exits with 1.
- An event has `status` `ACTIVE`, `CLOSED` or `UNRESOLVED`. `UNRESOLVED` means the track was lost or the stream ended
  while the event was open, so the system does not know whether the person left. `close_reason` is `left_corridor`,
  `left_view`, `track_lost` or `stream_end`. `type` is `crossing` or `entry` and is set only for a closed event that
  left the corridor.
- `evidence` says whether a detector frame saw the person inside (`inside_detected`) or only the motion model
  predicted it (`inside_predicted`, `rests_on_prediction`). The detector runs on every third frame in the shipped
  configuration, so predicted boxes are normal.
- Track ids are anonymous integers, valid within one run. They are not stable across runs and do no re identification.

## What a consumer should do (a recommendation, not a safety case)

| Input | Suggested treatment |
|---|---|
| `corridor = OCCUPIED` | Treat as occupied. |
| `corridor = UNKNOWN` or `state != OK` | Treat as occupied. An unknown corridor is not a clear one. |
| `corridor = CLEAR` with `state = OK` | Treat as "no confirmed person". It is not a guarantee. On the held out set 78 of 890 occupied frames were reported `CLEAR` with health `OK`. |
| No new line for longer than the expected frame interval | Treat as `UNAVAILABLE`. Sightline writes no heartbeat of its own. |

Sightline must not be the only protection for people near a moving robot.

## Not provided, and the next step for each

- **No live source.** `replay` reads video files and image folders. A camera or network stream needs a source adapter.
- **No message definition for a robot middleware.** Mapping `health`, `corridor` and `events` to a ROS 2 message and
  publishing them from a node is the natural integration. Not done.
- **No schema version field and no sequence number or heartbeat** in the records, so a consumer cannot detect a
  stalled writer from the records alone.
- **Image space only.** The corridor is a polygon in pixels for one fixed camera view. There are no camera
  intrinsics, no ground plane and no distances in metres (PLAN.md, no metric distances are claimed).
- **Timing is measured offline.** The throughput and latency figures in `docs/RESULTS.md` come from replay on one
  laptop, with a 549 s stall recorded as a failed throughput check (PLAN.md A7.2, A9). A real time budget on robot
  hardware has not been measured.
