# Solo recording guide

For filming **only yourself**, with nobody else in frame. This is a short
version of [RECORDING_GUIDE.md](RECORDING_GUIDE.md); its setup, privacy and
after-session rules still apply.

## Before you start

- Pick a time when nobody else will walk through. If someone does, stop and
  delete that clip.
- Tripod, at the robot's camera height, pointing along or across the corridor.
  **Do not move it** during a session. Every clip below is from that one
  position, so one `draw-corridor` covers them all.
- Phone: landscape, 1080p, 30 fps. Stabilisation, HDR and "action" modes off.
- Note the date, the place and the camera position. All clips from one session
  go into the same split (val **or** test). Choose which split before you look
  at any results.
- Name clips `own-<location>-<yyyymmdd>-<nn>`. Keep the originals and never
  edit them.

## The clips (13 to 15)

| # | Clip | How |
|---|---|---|
| 2 | **Sensor lights switching on** | Wait until the motion-sensor lights have gone off. Start recording, then walk in so the lights switch on, and carry on through the corridor. Leave exposure on auto. |
| 2 | **Locked, lowered exposure** | Lock focus and exposure (long-press on most phones), then drag exposure down until the corridor looks dim but you can still see yourself. Walk through. Lower it in the camera only, never afterwards. |
| 3 | **Empty hallway** | 30 to 60 s each, nobody in frame at all. Lights on as normal. Their `events.json` is `[]`. |
| 6 to 8 | **Walking in, crossing, stopping at the edge** | 10 to 30 s each, one action per clip. Walk **in** from the far end and stop inside. **Cross** the corridor from side to side. Walk up to the edge and **stop just outside** it, then turn back. Spread these across both directions and both sides. |

Write down which clip is which as you go. It saves guessing later.

## Things to know

- **Faces:** your face is in these clips. Raw clips stay in `data/`
  (git-ignored). Blur faces in any frame you show anywhere else, including
  the strips written by `sightline errors`.
- **No crowding:** one person cannot film occlusion or crowding. Those
  conditions still need other people, with their written consent.
- **Empty clips and "negative":** [PLAN.md](PLAN.md) defines a negative clip as
  one where people are visible but never enter. An empty hallway does not
  match that wording. Decide whether to mark these clips `negative` **before**
  you score them, and record the decision.
- **Stopping at the edge:** if you stop just outside the polygon, the clip has no
  event. If you stop with your feet inside, it has one. Label it from what the
  frames show, not from what you meant to do.

## After the session

Follow the "After each session" steps in [RECORDING_GUIDE.md](RECORDING_GUIDE.md):
`check-clip`, `draw-corridor`, box labels, `label-events`, manifest entry,
`split-check`, then add the counts to the data card.
