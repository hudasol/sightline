### In scope (judged view)

| Candidate | Frames | Thr @ best F1 | P | R | F1 | P / R @ 0.50 | Max R (any thr) | P,R >= bar? | fps (end to end) | Infer p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| mbv3_320fpn_640x360 | 4137 | 0.55 | 0.745 | 0.512 | 0.607 | 0.712 / 0.526 | 0.575 | no | 4.99 | 180 |
| mbv3_320fpn_960x540 | 4137 | 0.65 | 0.823 | 0.748 | 0.784 | 0.745 / 0.790 | 0.821 | no | 3.26 | 291 |
| resnet50_fpn_v2_960x540 | 4137 | 0.85 | 0.824 | 0.891 | 0.857 | 0.639 / 0.972 | 0.980 | yes | 0.19 | 3734 |

### All boxes

| Candidate | Frames | Thr @ best F1 | P | R | F1 | P / R @ 0.50 | Max R (any thr) | P,R >= bar? | fps (end to end) | Infer p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| mbv3_320fpn_640x360 | 4137 | 0.30 | 0.617 | 0.282 | 0.387 | 0.744 / 0.247 | 0.282 | no | 4.99 | 180 |
| mbv3_320fpn_960x540 | 4137 | 0.35 | 0.700 | 0.414 | 0.521 | 0.781 / 0.385 | 0.423 | no | 3.26 | 291 |
| resnet50_fpn_v2_960x540 | 4137 | 0.60 | 0.774 | 0.595 | 0.673 | 0.737 / 0.614 | 0.643 | no | 0.19 | 3734 |

fastest candidate with P>=0.8 and R>=0.8 at some threshold on val (in-scope boxes) AND >=10.0 fps (PLAN.md s5, A1). Val holds only MOT17 so far: the bar itself is judged on held-out own corridor clips.

**NO candidate meets the rule; per PLAN.md s5 document a frame-sampling strategy and measure its effect on event recall and latency instead of lowering the bar**
