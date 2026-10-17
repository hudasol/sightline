# Recording guide: own corridor footage

What to film so the own clips can carry the protocol: the held-out bar (PLAN.md
section 3, amendment A1) and the detector choice (amendment A4). Read this
before the first session. Nothing here may be decided after looking at results.
Filming only yourself? See the short [SOLO_RECORDING_GUIDE.md](SOLO_RECORDING_GUIDE.md).

## What the protocol needs from you

| Need | Why | Where it goes |
|---|---|---|
| >= 10 held-out (test) clips | the bar is 12 held-out clips, and MOT17 supplies 2 | `test.json` |
| Hard conditions **for events** in test: occlusion/crowding, low light, blur | MOT17-10 is moving-camera, so it counts for detection and tracking only | `test.json` |
| At least one negative test clip | false events per 5 min is measured on negatives only | `test.json` |
| Own **val** clips, from other sessions | amendment A4 picks the detector and interval on own val clips (for v1.0, with no own clips, it was applied on MOT17 val: A6.8) | `val.json` |

A reasonable minimum is 2 locations x 2 sessions each:
- give one session per location to val (about 4-6 clips)
- give the other sessions to test (at least 10 clips)

More is better. Decide the split **per session, before you annotate it**, and
write it down.

## Sessions, locations, cameras

- **Session** = one recording visit: one date, one place, one camera setup. Every
  clip from a session goes into the same split. So does every clip from a camera
  position. `split-check` fails otherwise.
- **Location** = a corridor. Different corridors make the test set harder to
  overfit. If there is only one corridor, vary the camera position between sessions.
- **Camera:** on a tripod, at the height the robot's camera will be, pointing
  along or across the corridor the way the robot would see it. Keep it still. A
  moving camera makes the image-space corridor meaningless, so the clip counts for
  detection and tracking only.
- **Phone settings:** landscape, 1080p, 30 fps. Turn off stabilisation, which can
  shift the frame, and any "action" or HDR mode. Lock focus and exposure when the
  phone allows it.
- **Clip length:** 20-60 s. Every non-negative clip contains at least one corridor
  entry or crossing.
- Name clips `own-<location>-<yyyymmdd>-<nn>`. Keep the original files in
  `data/own/originals/` and convert them with the README's ffmpeg command. Never
  edit originals.

## What to film

- **Clear:** one person walking into, along or across the corridor, in good light.
- **Occlusion / crowding:** 2-4 people crossing paths, walking side by side, one
  partly hiding another, someone stopping inside the corridor.
- **Low light:** a genuinely dim corridor, for example half the lights off or
  evening. Do not darken footage afterwards: that is the stress suite's job, and
  it is not a real condition.
- **Blur (for events):** keep the camera **still** and let people move fast
  (running, brisk walking close to the camera). A handheld clip cannot count for
  events.
- **Negatives:** people clearly visible who **never** enter the polygon. Examples:
  walking past just outside the edge, standing next to it, a door opening nearby,
  reflections on glass or floor. Their `events.json` is `[]`.
- **Edges of the event rule:** someone who steps in and straight back out, someone
  who lingers on the boundary, someone who leaves the frame while inside.

Tag each clip's `conditions` in the manifest from what is actually in it.

## Consent, privacy and faces

- **Written consent** from everyone identifiable, before recording, kept off the
  repo. Put its reference in the manifest `license` field (the template shows how).
- **Bystanders:** film where and when people who have not consented are unlikely
  to pass. If someone passes anyway, cut that part out or delete the clip.
- **No faces published, anywhere.** Raw clips stay in `data/` (git-ignored) and are
  never uploaded, pushed or shared. Any frame shown outside your machine must have
  faces blurred first. That includes report figures, slides, the demo video and
  the error-analysis renders written by `sightline errors`, which are real frames.
- **No audio.** The ffmpeg command drops it.
- **No identity work.** The system uses anonymous track ids only. Do not add
  names, face crops or any identity information to annotations.
- When the project ends, delete the raw clips you no longer need. Keep only what
  you are obliged to keep.

## After each session

1. Convert, then run `sightline check-clip` on every clip; fix anything marked NOT OK.
2. `draw-corridor` once per camera position.
3. Label boxes (CVAT, then `import-cvat`, recording which frames you annotated).
   Every box needs a real visibility value.
4. Label events with `label-events`.
5. Add the entries from `configs/manifests/own_clip.template.json` to val **or**
   test, then run `split-check`.
6. Record what you filmed in `docs/DATA_CARD.md` (counts table).

Lock the held-out manifest (`lock-manifest`) only once the test set is complete.
`tune` refuses a draft lock.
