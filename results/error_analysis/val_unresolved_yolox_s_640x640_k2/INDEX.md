# Unresolved events (auto-selected, worst first)

Selection: UNRESOLVED events whose track was lost while the event was open (close_reason track_lost; stream_end is not a failure). Worst first = longest open before the loss (close_frame - open_frame), ties broken by clip id then frame. Fixed before any strip was looked at.

Run: configs/manifests/val.json | detector yolox_s @[640, 640] | detect_every 2 | score threshold 0.5 | config configs/default.yaml (untuned)

Requested 10; 6 event(s) met the selection rule.

Legend: green polygon = corridor, green boxes = ground truth, grey = detections >= threshold, light blue = reported tracks, red = the event's track. These frames are not to be published (they show faces).
The **Why it happened** entries are drafts written with AI assistance from the saved strips and data. The author has not yet reviewed them. No setting was changed from them.

| # | Clip | Event | Open frames before loss | Opened / last inside / unresolved | Why it happened |
|---|---|---|---|---|---|
| 1 | MOT17-02-FRCNN | 1 (track #3) | 274 | 5 / 248 / 279 | Group occlusion. The track is lost inside a dense cluster of overlapping people in the middle of the zone and expires 31 frames after its last inside frame. Same cluster and frames as #5. The event stayed open for 274 frames, so track #3 may also have jumped between people (not checked). |
| 2 | MOT17-02-FRCNN | 10 (track #37) | 175 | 414 / 558 / 589 | Small distant person at the far (top) edge of the zone, then a large close person walks through the foreground. Likely a detection and tracking loss for a small person on the polygon border (lower confidence, strip is small). |
| 3 | MOT17-04-FRCNN | 6 (track #3) | 74 | 549 / 592 / 623 | Fixed object occlusion. The person stands at the left edge of the zone beside the lamp post and the track is lost once they are hidden. A new track id appears on what looks like the same person afterwards, so the track fragmented. The polygon edge sits on the post. |
| 4 | MOT17-02-FRCNN | 5 (track #16) | 46 | 163 / 178 / 209 | Group occlusion. A group walking together overlaps on the left of the zone and the track is lost among the overlapping boxes. |
| 5 | MOT17-02-FRCNN | 7 (track #26) | 38 | 241 / 248 / 279 | Group occlusion at the right edge of the zone, behind the central cluster that also causes #1 (same frames, both lost at 248). Likely merged group detections (lower confidence, strip is small). |
| 6 | MOT17-02-FRCNN | 12 (track #2) | 34 | 437 / 440 / 471 | Large foreground occluder. A very close person with a green bag walks across the view and hides the tracked person after only 3 detected frames inside the zone. |
