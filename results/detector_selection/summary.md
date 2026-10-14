### In scope (judged view)

| Candidate | Frames | Thr @ best F1 | P | R | F1 | P / R @ 0.50 | Max R (any thr) | P,R >= bar? | fps (end to end) | Infer p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| mbv3_320fpn_640x360 | 4137 | 0.55 | 0.745 | 0.512 | 0.607 | 0.712 / 0.526 | 0.613 | no | 4.99 | 180 |
| mbv3_320fpn_960x540 | 4137 | 0.65 | 0.823 | 0.748 | 0.784 | 0.745 / 0.790 | 0.832 | no | 3.26 | 291 |
| resnet50_fpn_v2_960x540 | 4137 | 0.85 | 0.824 | 0.891 | 0.857 | 0.639 / 0.972 | 0.983 | yes | 0.19 | 3734 |
| ssdlite320_320x320 | 4137 | 0.20 | 0.664 | 0.315 | 0.427 | 0.932 / 0.184 | 0.399 | no | 12.15 | 75 |
| yolox_nano_416x416 | 4137 | 0.25 | 0.791 | 0.669 | 0.725 | 0.904 / 0.533 | 0.730 | no | 41.63 | 18 |
| yolox_tiny_416x416 | 4137 | 0.20 | 0.805 | 0.753 | 0.778 | 0.921 / 0.582 | 0.788 | no | 19.63 | 42 |
| yolox_s_640x640 | 4137 | 0.30 | 0.842 | 0.863 | 0.853 | 0.895 / 0.724 | 0.933 | yes | 6.46 | 144 |

### All boxes

| Candidate | Frames | Thr @ best F1 | P | R | F1 | P / R @ 0.50 | Max R (any thr) | P,R >= bar? | fps (end to end) | Infer p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| mbv3_320fpn_640x360 | 4137 | 0.20 | 0.536 | 0.305 | 0.389 | 0.744 / 0.247 | 0.319 | no | 4.99 | 180 |
| mbv3_320fpn_960x540 | 4137 | 0.35 | 0.700 | 0.414 | 0.521 | 0.781 / 0.385 | 0.448 | no | 3.26 | 291 |
| resnet50_fpn_v2_960x540 | 4137 | 0.60 | 0.774 | 0.595 | 0.673 | 0.737 / 0.614 | 0.664 | no | 0.19 | 3734 |
| ssdlite320_320x320 | 4137 | 0.15 | 0.490 | 0.204 | 0.288 | 0.940 / 0.084 | 0.204 | no | 12.15 | 75 |
| yolox_nano_416x416 | 4137 | 0.15 | 0.716 | 0.362 | 0.481 | 0.918 / 0.248 | 0.362 | no | 41.63 | 18 |
| yolox_tiny_416x416 | 4137 | 0.15 | 0.797 | 0.398 | 0.531 | 0.934 / 0.278 | 0.398 | no | 19.63 | 42 |
| yolox_s_640x640 | 4137 | 0.15 | 0.799 | 0.513 | 0.625 | 0.915 / 0.360 | 0.513 | no | 6.46 | 144 |

fastest candidate with P>=0.8 and R>=0.8 at some threshold on val (in-scope boxes) AND >=10.0 fps (PLAN.md s5, A1). Val holds only MOT17 so far: the bar itself is judged on held-out own corridor clips.

**NO candidate meets the rule; per PLAN.md s5 document a frame-sampling strategy and measure its effect on event recall and latency instead of lowering the bar**
