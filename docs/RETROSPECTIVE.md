# Retrospective

> **Draft, 2026-10-17.** Drafted with AI assistance from the project record (PLAN amendments, PROCESS_LOG, commit
> history and `results/`) after the held out evaluation, for Huda to rewrite in her own words. Every
> number is from a file under `results/`.
>
> TODO(author): confirm. Replace this line with a statement that the author reads and owns this retrospective as of the submission date.

## What went wrong
- **Own footage never happened.** The plan put filming on days 1-2 and depended on it for most held out
  clips. By the deadline no own clip existed, so the held out set had to come from a public indoor
  dataset (IndoorCrowd, A6) that was found and imported on the last day. The system was never evaluated
  on robot height footage of a real corridor.
  *Later note (2026-10-18, PLAN.md A8):* the mentor confirmed that the task only required sourcing data,
  never filming it. So own footage was never a requirement. What remains true is the planning lesson:
  the plan assumed filming, and the sourced held out set was found late.
- **The bar is missed on three numbers.**
  - IDF1 is 0.512 over all boxes; the bar is 0.65.
  - Event precision is 0.840 and recall 0.787; the bar is 0.85 for both.
  - The throughput check failed on a run with a 549 s stall (A7.2).
- **A corridor was drawn for the wrong view.** A6 assumed one camera per IndoorCrowd site. ACS-EC has two
  views, so on two of its recordings the corridor covers the counter area (A7.1). Those two recordings
  hold 10 of the 17 missed events and 56 of the 78 false CLEAR frames.
- **The detection interval was chosen at one frame rate and used at another.** A4 picked k = 3 because
  it reached 10 fps on 30 fps val video. At 5 fps that means one detection every 0.6 s. The likely result
  (inferred from the event pattern, not measured) is that people close to the camera move too far between
  detections and their tracks fragment. One ACS-EG clip produced 15 predicted events for 6 GT events, and
  ACS-EG accounts for 10 of the 12 false events.
- **No mid point check-in was sent.** The brief asked for one on day 4, and none was sent. Two questions
  went to the mentor during the task, but they were not the check-in.

## Bad assumptions
- "One fixed camera per site." It was not checked against every recording's first frame until after the
  evaluation. A contact sheet of first frames takes a minute and would have caught it.
- "A detection interval is a property of the machine." It is a property of the machine and the frame
  rate together. The interval should have been chosen in seconds (or as a detection rate) and checked on
  the frame-rate-matched val copies, not in frames on 30 fps video.
- "Timing on this laptop is stable." The same code ran 13.08 fps on 14 Oct and 7.30 fps on 17 Oct at
  k = 2. One held out run had a 549 s stall, during which the laptop was in Modern Standby (A9). A single
  timing run is not evidence.
- "MOT17 all-box detection numbers measure the detector." All four MOT17-09 detection failures in the
  error analysis are people cut by the image edge whose ground truth extends off frame. The in scope view
  (A1) excludes them, which supports judging the detection bar in scope.

## Redesigns along the way
- Frame skipping became a detection interval: every frame is processed and tracks are predicted in
  between (A2).
- Settings counted in frames became settings in seconds (A3), and the event label rule followed (A6.5).
- Health gained a detection age check after the review found that async mode could rest on stale
  detections.
- Coverage counting changed from manifest entries to distinct footage, so negative corridors on the same
  footage do not inflate the clip count. Stress, error selection, evaluation and ablation reuse
  detections per source for the same reason.

## Technical lessons
- **The motion model carries the gain.** The ablation shows the Kalman prediction carries nearly all of
  ByteTrack's gain (IDF1 0.438 without it against 0.512, and 423 ID switches against 235). The low score
  stage adds little here. A likely reason: with the detector on one frame in three, prediction carries
  two frames out of three.
- **CLEAR means "no track inside", not "nobody inside".** 8.8% of occupied frames were reported CLEAR
  with health OK. The health checks look at the input and the pipeline, not at missed people. An advisory
  "clear" needs its own evidence, for example a minimum detector recall on the zone, or an "unknown"
  state whenever the zone has had no detection for a while.
- **Confidence did not predict failure on held out data.** The Spearman correlation of frame confidence
  with recall was 0.26 on val and -0.01 on held out. The low confidence health trigger rests on a relation
  that did not transfer.
- **Generated tables beat typed ones.** `sightline report` writes `docs/RESULTS.md` from `results/`, and
  two of the per clip recomputations (sensitivity, false CLEAR) were checked to reproduce the evaluation
  totals exactly.

## What my own checks missed
- **The ACS-EC second view.** The import check measured that two recordings differed (62 grey levels)
  and read it only as "not the same video". It should have raised "maybe not the same camera".
- **Frame rate effects on the interval.** The frame-rate-matched val copies existed, but A4 had already
  fixed k from throughput alone, so tuning could not change it.
- **The runtime run that the bar check read was not looked at before the evaluation used it.** The stall
  was visible in its maximum latency (548,939 ms).
- **A timing test that passes on an idle machine can fail under load.** The async latency test failed
  once in three while the stress suite ran. It is a test of the machine as much as of the code.

## What I would refuse to claim from these results
- That the system works on a robot. No robot height footage was evaluated.
- That the negative set shows a false event rate. 3.6 minutes with zero events cannot show a rate below 1
  per 5 minutes.
- That IDF1 meets the bar "in scope" (0.713). The bar is defined on all boxes, and the in scope view is
  reported, not substituted.
- That the throughput bar is met. The median of three runs is 14.79 fps, but the recorded check failed,
  and run to run variation on this laptop is about a factor of 2.

## What I would do next
1. Optional (A8): record own corridor clips (`docs/SOLO_RECORDING_GUIDE.md`), add them with a fresh lock,
   and evaluate once more. The task did not require it.
2. Make the detection interval time based, and choose it on the frame-rate-matched val copies. At 5 fps,
   k = 1 does not fit on this laptop: YOLOX-s at k = 1 ran at 3.98 fps on val, below the source rate.
   The options are a smaller or faster detector at k = 1, or k = 2 with re-measured event results.
   (The async worker is a third option once its accuracy is measured.)
3. Add a "zone not observed" state, so CLEAR needs recent detector evidence on the zone itself.
4. Redraw each IndoorCrowd corridor per view, on a new lock. Never on this one.
