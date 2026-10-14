# Unresolved events (auto-selected, worst first)

Selection: UNRESOLVED events whose track was lost while the event was open (close_reason track_lost; stream_end is not a failure). Worst first = longest open before the loss (close_frame - open_frame), ties broken by clip id then frame. Fixed before any strip was looked at.

Run: configs/manifests/val.json | detector yolox_s @[640, 640] | detect_every 2 | score threshold 0.5 | config configs/default.yaml (untuned)

Requested 10; 6 event(s) met the selection rule.

Legend: green polygon = corridor, green boxes = ground truth, grey = detections >= threshold, light blue = reported tracks, red = the event's track. These frames are not to be published (they show faces).
The **Why it happened** column is for a human who has looked at the strip. No setting was changed from it.

| # | Clip | Event | Open frames before loss | Opened / last inside / unresolved | Why it happened |
|---|---|---|---|---|---|
| 1 | MOT17-02-FRCNN | 1 (track #3) | 274 | 5 / 248 / 279 | TODO(human): look at `01_unresolved_MOT17-02-FRCNN_ev1_f279.png` |
| 2 | MOT17-02-FRCNN | 10 (track #37) | 175 | 414 / 558 / 589 | TODO(human): look at `02_unresolved_MOT17-02-FRCNN_ev10_f589.png` |
| 3 | MOT17-04-FRCNN | 6 (track #3) | 74 | 549 / 592 / 623 | TODO(human): look at `03_unresolved_MOT17-04-FRCNN_ev6_f623.png` |
| 4 | MOT17-02-FRCNN | 5 (track #16) | 46 | 163 / 178 / 209 | TODO(human): look at `04_unresolved_MOT17-02-FRCNN_ev5_f209.png` |
| 5 | MOT17-02-FRCNN | 7 (track #26) | 38 | 241 / 248 / 279 | TODO(human): look at `05_unresolved_MOT17-02-FRCNN_ev7_f279.png` |
| 6 | MOT17-02-FRCNN | 12 (track #2) | 34 | 437 / 440 / 471 | TODO(human): look at `06_unresolved_MOT17-02-FRCNN_ev12_f471.png` |
