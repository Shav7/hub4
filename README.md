# hub4 — a four-limb servo creature that reacts to what it sees

Four Feetech ST3215 servos daisy-chained into a central hub. A webcam looks at the
scene, YOLOv8n (run through OpenCV's DNN module) names what's there, the labels are
semantically matched to the closest animation, and that animation is choreographed
across the four limbs with smooth minimum-jerk motion.

```
camera ──▶ YOLOv8n (cv2.dnn) ──▶ labels ──▶ semantic match ──▶ animation ──▶ 4× ST3215
```

## Hardware

- 4× Feetech ST3215 (STS3215) servos, IDs 1–4, 1 Mbps, on a single bus board
- Central hub with four sockets at 90°; each servo's horn faces outward with a bracket arm
- Limb naming: ID 1 = front, 2 = right, 3 = back, 4 = left (see `SpiderConfig`)

## Setup

```sh
conda create -n lerobot python=3.12 && conda activate lerobot
pip install -r requirements.txt
./models/download.sh            # YOLOv8n ONNX weights
```

Find your serial port with `lerobot-find-port` and set it in `src/spider.py` (`DEFAULT_PORT`).
Pick the camera once with `python cameras.py` (OpenCV's camera index order on macOS does not
match the system device list, so it cannot be chosen by name).

## Run

```sh
python pose.py --list                                   # static poses
python pose.py spider flower_closed alert neutral       # play a pose sequence
python demo.py                                          # gaze-follow sweep
python live.py --show                                   # real-time loop with a window: boxes, confidence, current animation
python live.py --dry-run --show                         # vision + switching only, no servos
python cameras.py                                       # see every camera with its index; press the digit to remember the USB one
python live.py --camera 0 --remember                    # or set it directly; saved to camera.json
python live.py --confirm 6 --min-conf 0.6 --dwell 2     # less sensitive
python perform.py spider flower greet --seconds 10      # choreographed moods, no camera needed
```

## Moods and choreography

What the camera sees selects a **mood**. Each mood has a rest pose, an entry gesture, a few
**macro** animations (full-body) and a few **micro** animations (small offsets layered on the
rest pose). The choreographer alternates macros and micro bursts with randomised tempo and
repeat counts, never repeating a macro back to back, so motion stays expressive without looping.

| Mood | Motion | Triggered by |
|---|---|---|
| spider | legs planted, opposite pairs scuttle | cat, dog, bear, knife, scissors |
| flower | petals unfold in sequence, breathe, close | potted plant, vase, umbrella |
| greet | rises to attention, front limb waves | person |
| wave | a crest travels around the ring | ball, frisbee, kite, vehicles |
| scan | slow probing look-around | laptop, phone, tv, keyboard |
| rest | curls up and holds | couch, bed, teddy bear |
| idle | gentle breathing ripple | empty scene |

Add an animation in `src/animation.py`: a few keyframes (degrees per limb), an optional
per-limb phase lag for ripple choreography, and a `SemanticProfile` of the concepts it
embodies. Detected labels map to concepts in `src/semantics.py`.

## Layout

```
src/spider.py      bus controller: torque, sync read/write, minimum-jerk moves
src/poses.py       static pose library + per-limb calibration (neutral, sign, limits)
src/animation.py   keyframed animations, ring choreography, streaming player
src/vision.py      YOLOv8n via cv2.dnn, warmed-up camera
src/semantics.py   label ⇄ animation matching with hysteresis
src/realtime.py    debounced mood switcher + continuous player thread with crossfade
src/choreography.py moods (rest pose, entry, macros, micros, tempo) and the clip sequencer
tests/             70 tests against a fake bus (no hardware needed)
```

```sh
pytest -q
```

## Notes

- Servos top out near 200°/s at 12 V, so a 90° swing takes ~0.9 s regardless of the
  commanded duration. Keep neighbouring poses within ~60° for snappy motion.
- If a limb swings the wrong way, flip its `sign` in `LimbCalibration`.
