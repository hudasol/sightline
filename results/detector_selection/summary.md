| Candidate | Frames | Thr @ best F1 | P | R | F1 | P / R @ 0.50 | Max R (any thr) | P,R >= bar? | fps (end to end) | Infer p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| mbv3_320fpn_640x360 | 4137 | 0.30 | 0.618 | 0.282 | 0.387 | 0.745 / 0.247 | 0.282 | no | 4.99 | 180 |
| mbv3_320fpn_960x540 | 4137 | 0.35 | 0.700 | 0.414 | 0.521 | 0.781 / 0.385 | 0.423 | no | 3.26 | 291 |
| resnet50_fpn_v2_960x540 | pending | | | | | | | | 0.19 | 3734 |

fastest candidate with P>=0.8 and R>=0.8 at some threshold on val AND >=10.0 fps (PLAN.md s5)

**NO candidate meets the rule; per PLAN.md s5 document a frame-sampling strategy and measure its effect on event recall and latency instead of lowering the bar**
