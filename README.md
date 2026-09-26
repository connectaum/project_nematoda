# project_nematoda — automatic pose tracking and locomotion analysis of nematodes

Tools for turning raw microscope videos of free-living nematodes (developed on
*Alloionema appendiculatum*, transmitted-light videos at 720p/1080p, ~20 fps) into

* **49-point midlines** for every worm in every frame, including worms that touch, cross or coil,
* **per-worm tracks** with QC flags,
* **ready-to-open DeepLabCut projects** (single- or multi-animal) with auto-generated labels,
* **locomotion features** (speed, undulation frequency, wavelength, amplitude, curvature, pauses, reversals…)
  and an exploratory clustering of behaviour.

The goal of the underlying research is to compare locomotion patterns of nematodes of different sizes and life
stages (e.g. free-living stages vs. dauers vs. post-parasitic worms) without manual annotation.

No video data, trained weights or generated projects are included in this repository — only code.

---

## Contents

| Path | What it is |
|---|---|
| `worm_pipeline.py` | Classical segmentation → skeleton → midline → tracking pipeline. One video in, `auto_labels/<video>/` out |
| `worm_split.py` | Skeleton-graph splitting of merged components (two crossing worms) — used by `worm_pipeline.py` |
| `make_dlc_project.py` | Builds or extends a DeepLabCut project (5 or 49 body parts, single or multi-animal) from pipeline output |
| `run_all.py` | Batch driver: every video in a folder → auto labels → one DLC project |
| `worm_kinematics.py` | Per-track and per-20-s-window locomotion features |
| `kin_analysis.py` | Per-video summary, PCA + hierarchical clustering of behaviour windows, figures |
| `wormbot/` | Telegram bot (Docker) that runs the whole chain on a server; `wormbot/pipeline/` holds the bot's copy of the scripts plus the contact-solving steps (DeepTangleCrawl, SAM2 episode export/merge, overlay rendering) |
| `wormsam2/` | Desktop tool (Windows/macOS/Linux) that resolves contact episodes with Meta's SAM2 video segmentation, with a click editor for manual prompts |
| `dlc-env/` | Recipe and launcher for an isolated DeepLabCut 3 environment |
| `bench/` | Research code: benchmark of crossing-worm methods, synthetic crossings, DeepTangleCrawl fine-tuning |
| `requirements.txt`, `requirements-dtc.txt` | Python dependencies (core / with DeepTangleCrawl) |

The scripts in the repository root and in `wormbot/pipeline/` are kept identical; the root copies are the
standalone entry points, the `wormbot/pipeline/` copies are what the Docker image ships.

---

## Installation

Requirements: **Python 3.10–3.12**, `git`, and `ffmpeg` (only needed by the bot for H.264 re-encoding).
[uv](https://docs.astral.sh/uv/) is recommended but plain `venv` + `pip` works too.

### macOS

```bash
git clone https://github.com/connectaum/project_nematoda.git
cd project_nematoda
python3 -m venv .venv              # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt    # add -r requirements-dtc.txt for DeepTangleCrawl
```

`brew install ffmpeg` if you want to run the bot locally.

### Linux

```bash
sudo apt install python3-venv ffmpeg      # Debian/Ubuntu; use your distro's equivalents
git clone https://github.com/connectaum/project_nematoda.git
cd project_nematoda
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # add -r requirements-dtc.txt for DeepTangleCrawl
```

On Linux the first pass of `worm_pipeline.py` runs in `PIPELINE_WORKERS` forked processes
(default `min(4, cpu_count)`); set `PIPELINE_WORKERS=1` to force a single process.

### Windows (PowerShell)

```powershell
git clone https://github.com/connectaum/project_nematoda.git
cd project_nematoda
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt       # add -r requirements-dtc.txt for DeepTangleCrawl
```

`winget install Gyan.FFmpeg` if you want to run the bot locally. The pipeline runs single-process on Windows
and macOS (multiprocessing in pass 1 is Linux-only).

### Optional components

| Component | How to install |
|---|---|
| DeepLabCut 3 (to view/correct labels and train) | see [DeepLabCut environment](#deeplabcut-environment) |
| wormsam2 (SAM2 for contact episodes) | `wormsam2/install.sh` (macOS/Linux) or `wormsam2/install.ps1` (Windows) — see [below](#resolving-contact-episodes-with-wormsam2) |
| DeepTangleCrawl weights | not included. The base checkpoint (`arrays.npy`, `eigenworms_transform.npy`, `tree.pkl`) comes from the DeepTangleCrawl / Tierpsy Tracker 2.0 release on Zenodo (record 14517883); place it in `wormbot/models/dtc/base/`. A fine-tuned checkpoint (`*.pkl`) is optional (`--ft`) |

---

## Usage

### 1. Auto-label one video

```bash
python worm_pipeline.py videos/IMG_8370.mp4 auto_labels/IMG_8370
```

Output in `auto_labels/<video>/`:

| File | Content |
|---|---|
| `labels49.npy` | `(N, 49, 2)` for one track or `(N, K, 49, 2)` for K tracks; point 0 = head, NaN = hidden |
| `curvature.npy` | dimensionless curvature κ·L along the body |
| `points5_dlc.csv`, `CollectedData_auto.csv` | 5 points per worm (head, q1, mid, q3, tail) in DeepLabCut format |
| `qc_frames.csv` | per-frame flags (`cluster`, `from_cluster`, `split`, `length_outlier`, `skeleton_branches`, …) and per-track label status |
| `tracks.json`, `summary.json` | per-track statistics, global threshold, median body length/area, fps |
| `overlay.mp4` | the video with midlines and the 5 points drawn (`OVERLAY_SCALE=0.5` for a smaller file) |

Always take fps from `summary.json`: phone videos are often variable-frame-rate and their container fps is wrong.

### 2. Resolve crossings and coils with DeepTangleCrawl (optional)

```bash
python wormbot/pipeline/dtc_infer.py videos/IMG_8370.mp4 auto_labels/IMG_8370 auto_labels/IMG_8370/dtc.pkl \
       --model-dir wormbot/models/dtc/base [--ft wormbot/models/dtc/<finetuned>.pkl]
python wormbot/pipeline/dtc_merge.py auto_labels/IMG_8370 auto_labels/IMG_8370/dtc.pkl
python wormbot/pipeline/render_overlay.py videos/IMG_8370.mp4 auto_labels/IMG_8370 auto_labels/IMG_8370/overlay.mp4
```

`dtc_merge.py` fills frames the classical pipeline could not label and repairs flagged ones; the previous labels
are kept as `labels49_before_dtc.npy`. About 0.1 s/frame on 4 CPU cores.

### 3. Build a DeepLabCut project

```bash
# one video
python make_dlc_project.py --project nematoda-ma --video videos/IMG_8370.mp4 --labels auto_labels/IMG_8370 \
       --multi --individuals 3 --nframes 40
# a whole folder (labels missing videos, skips videos already in the project)
python run_all.py --videos videos/ --project nematoda-ma --multi --individuals 3
```

`--npoints 49` gives body parts `p00…p48` instead of 5. `--no-video-copy` keeps videos out of the project;
`--project-path-in-config <path>` writes the path the project will have on the machine where DLC runs.
Frames are chosen for diversity (k-means on curvature + visibility); in multi-animal mode only frames in which
every visible worm is labelled are used, so an unlabelled worm is never learned as background.

### 4. Locomotion features

```bash
python worm_kinematics.py auto_labels/            # -> analysis/kin_tracks.csv, analysis/kin_windows.csv
python kin_analysis.py analysis/                  # -> summary CSV + figures
```

`worm_kinematics.py` writes into `<auto_labels_root>/../analysis/`.

### 5. Telegram bot (server)

The bot accepts videos in Telegram, runs steps 1–4 (plus the contact-episode export) and replies with the overlay
video, a kinematics summary and a zip with the DLC project. It uses a local Telegram Bot API server so files up
to 2 GB work. Works anywhere Docker runs (Linux server, Docker Desktop on macOS/Windows).

```bash
cd wormbot
cp .env.example .env         # fill in TELEGRAM_API_ID/HASH, BOT_TOKEN, ALLOWED_USERS, PROJECT_PATH_PREFIX
# optional: put DeepTangleCrawl weights into wormbot/models/dtc/ (or set DTC_ENABLE=0 in docker-compose.yml)
docker compose up -d --build
docker compose logs -f bot
```

Set `PROJECT_PATH_PREFIX` in `.env` to the folder where users will unpack the returned projects (it is written into
`project_path` of each project's `config.yaml`).
Without Telegram: `python wormbot/bot.py --process video1.mp4 [video2.mp4 …]` (needs `ffmpeg`; set
`DTC_ENABLE=0` if you have no DeepTangleCrawl weights). In Telegram: send videos as documents, then `/go`;
`/status`, `/cancel`, `/help`. In groups the bot only reacts to videos captioned with its @name.

---

## Manual annotation and correction

Everything above is automatic. Human input is needed in two places: worms that are **never seen separately**
in a contact episode, and **correcting labels** before training a pose model.

### Resolving contact episodes with wormsam2

When worms lie side by side, cross or coil together, the pipeline exports each contact episode
(`episodes_<video>.zip`: short clips + automatic prompts = each worm's cleanest nearby midline).
`wormsam2` propagates those prompts through the clip with SAM2 and returns `sam2_result_<video>.zip`.

Install (downloads PyTorch + SAM2, ~1–2 GB, and the model on first run, ~180 MB):

* **Windows**: right-click `wormsam2\install.ps1` → *Run with PowerShell*
  (or `Set-ExecutionPolicy -Scope Process Bypass; .\install.ps1`). Installs the CUDA build if an NVIDIA GPU is
  present, CPU otherwise, and adds `wormsam2` to the user PATH.
* **macOS / Linux**: `bash wormsam2/install.sh`. Uses MPS on Apple Silicon, CUDA on Linux with an NVIDIA GPU.
  The launcher is linked into `~/.local/bin/wormsam2`.

Check with `wormsam2 info` / `wormsam2 selftest`.

Run:

```bash
wormsam2 run episodes_IMG_8370.zip            # auto prompts; the editor opens only if some worm has none
wormsam2 run episodes_IMG_8370.zip --gui      # always open the click editor
wormsam2 run episodes_IMG_8370.zip --auto     # never open it (skip episodes without prompts)
#   --model tiny|small|base|large   --stride 2   --episode N (repeatable)   --out result.zip
```

**Click editor** (manual prompts):

| Key / mouse | Action |
|---|---|
| ← / → | browse frames — pick one where the worms can be told apart by eye |
| left click | positive point on the current worm (2–4 points along the body) |
| right click | negative point ("not this worm") |
| `n` | next worm |
| `d` | delete last point |
| `x` | delete current worm |
| Enter | run SAM2 |
| Esc | skip the episode |

Automatic prompts are marked with a star and can be kept. Check `epNN_overlay.mp4` / `epNN_sheet.jpg` inside
the result; `result.json` lists warnings (a mask about twice a worm's area = probably two worms; a tiny one = the
prompt missed).

Merge the result back — send the zip to the bot as a document, or locally:

```bash
python wormbot/pipeline/merge_sam2.py auto_labels/IMG_8370 sam2_result_IMG_8370.zip
python worm_kinematics.py auto_labels/ IMG_8370
```

Speed per 720p frame: NVIDIA GPU ≈ 0.2–0.5 s, Apple M1 Pro ≈ 1.6 s (small), 4–8 CPU cores ≈ 2–6 s (tiny).

### Correcting labels in DeepLabCut

1. Open DLC (see below): *Load project* → `<project>/config.yaml` → **Label frames** → pick
   `labeled-data/<video>/`. napari opens with the auto labels already placed.
2. Drag wrong points, delete points on hidden body parts (a missing point = "not visible", which DLC learns
   correctly), add missing worms in free individual slots; save with `Ctrl+S`
   (`Cmd+S` on macOS).
3. Check without the GUI: `deeplabcut.check_labels(config)` draws labelled frames into
   `labeled-data/<video>_labeled/`.
4. Train and iterate:

```python
import deeplabcut as dlc
cfg = "nematoda-ma/config.yaml"
dlc.create_training_dataset(cfg)                  # maDLC picks a multi-animal net; resnet_50 is enough for single
dlc.train_network(cfg, shuffle=1, epochs=200, batch_size=8)
dlc.evaluate_network(cfg, Shuffles=[1], plotting=True)
dlc.analyze_videos(cfg, ["new_video.mp4"], save_as_csv=True)
dlc.extract_outlier_frames(cfg, ["new_video.mp4"])   # frames the model got wrong
dlc.refine_labels(cfg)                               # GUI: fix only those, then merge_datasets + retrain
dlc.merge_datasets(cfg)
```

If the project was moved, fix `project_path` and the video paths in `video_sets` of `config.yaml`
(or regenerate with `--project-path-in-config`).

### DeepLabCut environment

DeepLabCut is kept in its own environment because its pins (numpy, napari, torch) clash with the rest.
Known issue: `napari >= 0.9` breaks `napari-deeplabcut 0.3.1` (`ImportError: SYMBOL_TRANSLATION_INVERTED`),
hence the `<0.7` pin.

macOS / Linux:

```bash
cd dlc-env
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "deeplabcut[gui]" "napari[pyside6]>=0.5,<0.7"
cd .. && ./dlc-env/dlc                      # GUI;  ./dlc-env/dlc script.py runs a script in this env
```

Windows:

```powershell
cd dlc-env
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe "deeplabcut[gui]" "napari[pyside6]>=0.5,<0.7"
.\.venv\Scripts\python.exe -m deeplabcut    # GUI
```

For GPU training on Windows/Linux install the CUDA build of PyTorch into this environment first
(see pytorch.org); Apple Silicon uses MPS automatically.

---

## Algorithms

### Segmentation (worm_pipeline.py, pass 1)

* **One global threshold per video.** Frames are sampled; on frames with a clear worm the threshold is
  `t_otsu + 0.55·(background − t_otsu)` (relaxed towards the background because the translucent head and tail are
  lighter than the body); the median over sampled frames is used and shifted by per-frame background drift.
  Per-frame Otsu was abandoned: on an empty frame it splits background noise and produces a giant fake worm.
* Morphological closing (disk 7), removal of components < 1500 px (debris), skeletonisation per component
  bounding box. Frames are independent, so on Linux this pass runs in parallel over contiguous frame ranges
  (workers reach their range with `grab()`, not seeking, which is unreliable on variable-frame-rate video).

### Midline and body points

* The longest path through the skeleton (two BFS passes) is taken as the body axis (frames where side branches
  exceed 15 % of it are flagged `skeleton_branches`, closed loops `loop_or_empty`), smoothed with a parametric
  spline and resampled to **49 points equally spaced in arc length**; the 5 exported
  points (head, q1, mid, q3, tail) are points 0, 12, 24, 36, 48, i.e. four equal body segments.
* Worms touching the frame border are labelled only on the visible part: points are placed by arc length from
  the visible end using the track's reference body length, and the rest are left hidden (`head_cut` / `tail_cut`).
* The width profile along the body comes from the distance transform of the mask.
* Curvature κ(s) is computed along the smoothed midline and made dimensionless as κ·L.

### Tracking and QC

* Every component is a worm candidate. Candidates are linked into tracks by centroid proximity (gate 150 px per
  frame; a lost track can be re-acquired within 450 px for up to 200 frames) with a body-size penalty.
* Components much larger than a single worm are flagged `cluster`. Branched clusters go to `worm_split.py`
  (below); two pieces of one worm broken at its translucent middle are recognised (`split`) and not labelled.
* **Head/tail**: endpoints are chained frame to frame, and each track gets one global orientation from the sign
  of its motion (centroid displacement projected on the head–tail axis): crawling worms mostly move head first.
  This is known to fail for backward-crawling worms; see *Limitations*.

### Splitting touching worms by skeleton graph (worm_split.py)

Skeleton → prune spurs → branch graph (`skan`) → enumerate every simple path from a free end. The turn at each
junction is measured *through* short junction-to-junction edges (the shared segment of an X crossing has an
intermediate direction and must not be judged on its own); a path may also end at a junction (a worm end buried
in the other body). Paths are ranked by maximum turn + a body-length prior + a termination penalty, and chosen
greedily with at most 40 % shared length. Solves ~40 % of X-crossings on the synthetic benchmark; cannot separate
worms lying side by side or two-worm rings (a single ridge, no boundary in the binary mask).

### DeepTangleCrawl ensemble (wormbot/pipeline/dtc_infer.py, dtc_merge.py)

[DeepTangle](https://github.com/kirkegaardlab/deeptangle) (Alonso & Kirkegaard) predicts worm midlines as
PCA coefficients of eigen-shapes directly from an 11-frame stack, which lets it follow worms through overlaps and
coils; DeepTangleCrawl (Tierpsy Tracker 2.0) is its version trained on crawling worms. Here:

* the video is scaled so that the median worm is ~90 px long (the shape basis is tied to 80–100 px worms),
  preprocessed like Tierpsy 2.0 (inverted grey × adaptive-threshold mask) and fed as stacks with
  `skip = round(fps/5)`;
* two models run — the shipped checkpoint and a checkpoint **fine-tuned on this project's own automatic labels**
  (clean worms from the classical pipeline, synthetic contacts made by blending two worms, rehearsal on the
  authors' training clips, pseudo-labelled coils from the base model, and a loss that supervises worms cut by the
  crop/frame edge on their visible points only — `bench/dtc/`). Each set is NMS-filtered, then the two are united
  geometrically; for duplicates the fine-tuned midline wins unless its length is implausible;
* `dtc_merge.py` links plausible midlines (0.6–1.3 × median length) into tracklets, assigns each to the pipeline
  track it overlaps best (median distance < 0.3 L) or creates a new track, fills unlabelled frames (`dtc`) and
  replaces labels on flagged frames that disagree by more than one body width (`dtc_fix`).

On three held-out contact clips the ensemble yields ≥ 2 plausible midlines in 52 % / 98 % / 85 % of frames
(base model alone: 45 / 67 / 66 %).

### SAM2 contact episodes (episode_export.py, wormsam2, merge_sam2.py)

* **Episode detection** is per track pair: a frame is in contact when two midlines come closer than ~1.5 body
  widths or the frame is flagged `cluster`; short gaps are bridged, nearby episodes merged. (A blob-area rule
  was abandoned: in videos with mixed worm sizes one large worm looked like a permanent contact.)
* **Prompts**: for each participating track, the cleanest labelled frame near the episode — fully labelled,
  ≥ 3 widths from other worms, and passing an image check (≥ 95 % of the midline inside one component whose area
  matches that worm) — searched before, after, then anywhere within a bounded range. Other worms' midlines on
  that frame become negative points.
* **SAM2** (`sam2.1-hiera` tiny/small/base) propagates each prompt forwards and backwards through the clip.
  Midlines are extracted from each mask as the **longest trail** in the skeleton graph (junction clusters
  merged, turn limit 80°), which handles self-overlapping loops where a longest path fails.
* **Merge**: plausible midlines are written into `labels49.npy` (stride gaps ≤ 4 frames interpolated), manual
  objects become new tracks, frames are flagged `from_sam2`.

### Kinematics (worm_kinematics.py, kin_analysis.py)

Two levels of frame validity: *core* (points 12–36 visible — enough for speed, frequency, pauses, reversals,
robust to worms cut by the frame edge) and *full* (all 49 points — wavelength, amplitude, curvature, coiling).
Features per track and per 20-s window (10-s step), all normalised by body length L:

| Feature | Definition |
|---|---|
| `speed_bl_s` | median centroid speed while moving, L/s (moving ≥ 0.04 L/s) |
| `frac_move`, `frac_pause`, `frac_thrash` | pause = slow and not bending; thrash = slow but bending in place |
| `freq_hz` | dominant undulation frequency of mid-body curvature (Welch PSD over moving runs ≥ 2 s); `freq_zc_hz` from zero crossings as a check |
| `wavelength_bl` | 2 × mean spacing of zero crossings of κ(s), / L |
| `amp_bl` | half peak-to-peak displacement perpendicular to the body's principal axis, / L |
| `curv_rms`, `kmax_p90`, `frac_coil` | RMS and 90th-percentile |κ·L|; coil = max |κ·L| > 12 |
| `bend_act` | median |d(κ·L)/dt| at mid-body |
| `rev_per_min`, `frac_backward` | sustained (≥ 0.5 s) movement against the head direction |

`kin_analysis.py` standardises window features, runs PCA and Ward clustering (k by silhouette) and draws per-video
dot plots, the PCA map and a dendrogram of tracks.

### Benchmark (bench/)

`make_synth.py` generates synthetic crossings by pasting a worm cut from another frame of the same video with
multiplicative (transmitted-light) blending, in four geometries (X, shallow angle, parallel, tip contact), with
ground-truth midlines and instance masks. Methods (pipeline baseline, skeleton graph, erosion + watershed,
ridge filter, Omnipose, SAM2, DeepTangle) are scored by "all worms recovered with mean midline error < 5 % L"
on synthetic frames and by plausible-midline counts on real contact frames.

---

## Limitations

* Head/tail orientation relies on motion only; backward crawling (mutants, males) flips it. Planned: consensus of
  motion, tip-angle variability and intensity profile (as in Tierpsy) and tail tapering.
* Thresholds that depend on a median worm size assume roughly uniform sizes within a video.
* Worms that are never seen separately in a contact episode need a manual click in wormsam2.
* Tested on transmitted-light videos of one species; other setups may need a different threshold rule.

## License and citation

The code in this repository is released under the [MIT License](LICENSE), except `wormbot/pipeline/deeptangle/`,
which keeps its own MIT license (see below). Model weights and datasets are not part of the repository and are
subject to their own terms.

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff)
(GitHub shows it as *Cite this repository*).

## Acknowledgements and third-party code

* `wormbot/pipeline/deeptangle/` — a vendored subset of [DeepTangle](https://github.com/kirkegaardlab/deeptangle)
  (MIT, © 2023 Albert Alonso; see its `LICENSE`), as used by DeepTangleCrawl / Tierpsy Tracker 2.0.
* [SAM2](https://github.com/facebookresearch/sam2) (Meta), [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut),
  [skan](https://github.com/jni/skan), [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) and
  [SLEAP](https://sleap.ai) (design reference for head/tail and tracking).
