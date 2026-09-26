"""
Build (or extend) a DeepLabCut project from auto-generated labels (worm_pipeline.py output).

Usage:
  python make_dlc_project.py --project <dir> --video <video.mp4> --labels <auto_labels/<video>/> \
        [--npoints 5|49] [--nframes 40] [--scorer auto] [--multi] [--individuals 3] [--project-path-in-config <path>]

What it does
  * creates <project>/{config.yaml, videos/, labeled-data/<video>/, training-datasets/, dlc-models/}
    (same layout as deeplabcut.create_new_project), or adds a video to an existing one
  * picks --nframes diverse frames (k-means on body curvature + visibility), extracts them as PNG
  * writes labeled-data/<video>/CollectedData_<scorer>.{csv,h5} in DLC's native format
    (hidden points = empty cells, which DLC treats as "not visible")
  * --npoints 49: bodyparts p00..p48 (arc length 0..1 from the head); 5: head, q1, mid, q3, tail
  * --multi: multi-animal project (maDLC). Columns scorer/individuals/bodyparts/coords, individuals
    worm1..worm<--individuals>. Worms present in a frame fill the slots in track order (identity=False,
    so slot assignment does not need to be consistent between frames). Only frames in which EVERY
    visible worm is labelled are used (an unlabelled worm would be learned as background).
"""
import argparse, os, shutil, datetime
from pathlib import Path
import numpy as np, pandas as pd, cv2, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--labels", required=True, help="output dir of worm_pipeline.py for this video")
ap.add_argument("--npoints", type=int, default=5, choices=[5, 49])
ap.add_argument("--nframes", type=int, default=40)
ap.add_argument("--scorer", default="auto")
ap.add_argument("--task", default="nematoda")
ap.add_argument("--multi", action="store_true", help="multi-animal project")
ap.add_argument("--individuals", type=int, default=3, help="number of individual slots (multi only)")
ap.add_argument("--project-path-in-config", default=None)
ap.add_argument("--no-video-copy", action="store_true", help="do not copy the video into <project>/videos")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

proj = Path(a.project); vid = Path(a.video); stem = vid.stem
for sub in ["videos", "labeled-data/" + stem, "training-datasets", "dlc-models"]:
    (proj / sub).mkdir(parents=True, exist_ok=True)
if not a.no_video_copy and not (proj / "videos" / vid.name).exists():
    shutil.copy(vid, proj / "videos" / vid.name)

# ---------------------------------------------------------------- labels
lab49 = np.load(Path(a.labels) / "labels49.npy")           # (N,49,2) or (N,K,49,2), NaN = hidden
curv = np.load(Path(a.labels) / "curvature.npy")           # (N,49) or (N,K,49)
if lab49.ndim == 3: lab49, curv = lab49[:, None], curv[:, None]
N, K = lab49.shape[:2]
qc = pd.read_csv(Path(a.labels) / "qc_frames.csv"); qc["flags"] = qc["flags"].fillna("")
qc["label_status"] = qc["label_status"].fillna("")
if a.npoints == 5:
    idx = np.linspace(0, 48, 5).round().astype(int); names = ["head", "q1", "mid", "q3", "tail"]
else:
    idx = np.arange(49); names = [f"p{i:02d}" for i in range(49)]
labels = lab49[:, :, idx]                                    # (N,K,P,2)
n_lab = (~np.isnan(labels[:, :, :, 0])).sum(2)              # labelled points per worm slot
present = n_lab >= 2                                        # (N,K) worm with usable labels

BAD = {"cluster", "skeleton_branches", "loop_or_empty", "spline_fail", "length_outlier", "no_worm"}
BAD_STATUS = {"both_ends_cut", "visible_longer_than_body", "no_midline"}
def frame_ok(k):
    fl = set(f for f in qc.loc[k, "flags"].split(";") if f)
    if fl & BAD: return False
    st = set(s for s in str(qc.loc[k, "label_status"]).split(";") if s and s != "-")
    if st & BAD_STATUS: return False
    if a.multi and present[k].sum() > a.individuals: return False   # more worms than individual slots -> one would train as background
    return present[k].sum() >= 1
usable = np.array([frame_ok(k) for k in range(N)])
# two labelled worms whose midline ends point at each other across a gap are probably one worm broken in two -> skip the frame
SPLIT_D, SPLIT_ANG = 350, 60
def pieces_of_one_worm(da, db):
    da, db = da[~np.isnan(da[:, 0])], db[~np.isnan(db[:, 0])]
    if len(da) < 6 or len(db) < 6: return False
    for ea, ta in ((da[0], da[0]-da[4]), (da[-1], da[-1]-da[-5])):
        for eb, tb in ((db[0], db[0]-db[4]), (db[-1], db[-1]-db[-5])):
            v = eb - ea; n = np.linalg.norm(v)
            if n == 0 or n > SPLIT_D: continue
            ca = np.dot(ta, v) / (np.linalg.norm(ta)*n + 1e-9); cb = np.dot(tb, -v) / (np.linalg.norm(tb)*n + 1e-9)
            if ca > np.cos(np.radians(SPLIT_ANG)) and cb > np.cos(np.radians(SPLIT_ANG)): return True
    return False
for k in np.flatnonzero(usable):
    js = np.flatnonzero(present[k])
    for i1 in range(len(js)):
        for i2 in range(i1 + 1, len(js)):
            if pieces_of_one_worm(lab49[k, js[i1]], lab49[k, js[i2]]): usable[k] = False
cand = np.flatnonzero(usable)
print(f"{stem}: {N} frames, {K} track(s), {len(cand)} usable frames "
      f"({(present[cand].sum(1) == 1).sum()} with 1 worm, {(present[cand].sum(1) >= 2).sum()} with 2+)")
MANUAL_ONLY = len(cand) == 0
if MANUAL_ONLY:
    print("no frame with all worms auto-labelled -> extracting evenly spaced frames for MANUAL labelling (no CollectedData)")

# ---------------------------------------------------------------- diverse frame selection
# feature = curvature profile (NaN->0) + visible fraction per worm slot (sorted by x), k-means, one frame per cluster
M = a.individuals if a.multi else 1
X = np.zeros((len(cand), M * 50))
for r, k in enumerate(cand):
    js = [j for j in range(K) if present[k, j]]
    js.sort(key=lambda j: np.nanmean(lab49[k, j, :, 0]))
    for s, j in enumerate(js[:M]):
        X[r, s*50:s*50+49] = np.nan_to_num(curv[k, j]) / 8.0
        X[r, s*50+49] = (~np.isnan(lab49[k, j, :, 0])).mean() * 3.0
kk = min(a.nframes, len(cand))
if MANUAL_ONLY:
    chosen = [int(f) for f in np.linspace(0, N - 1, a.nframes).round()]      # evenly spaced, for manual labelling
else:
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=kk, n_init=10, random_state=a.seed).fit(X)
        chosen = [cand[np.argmin(np.linalg.norm(X - c, axis=1))] for c in km.cluster_centers_]
    except ImportError:
        rng = np.random.default_rng(a.seed); chosen = rng.choice(cand, kk, replace=False)
chosen = sorted(set(int(c) for c in chosen))
print(f"selected {len(chosen)} frames: {chosen}")

# ---------------------------------------------------------------- extract frames
width = len(str(N))                                         # DLC zero-pads to the digit count of the frame total
cap = cv2.VideoCapture(str(vid)); W, H = int(cap.get(3)), int(cap.get(4))
imgnames = []
for f in chosen:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    if not ok: continue
    name = f"img{str(f).zfill(width)}.png"
    cv2.imwrite(str(proj / "labeled-data" / stem / name), fr); imgnames.append((f, name))
cap.release()

# ---------------------------------------------------------------- CollectedData csv/h5
if not MANUAL_ONLY:
    rows = pd.MultiIndex.from_tuples([("labeled-data", stem, nm) for _, nm in imgnames])
    if a.multi:
        indiv = [f"worm{i+1}" for i in range(M)]
        cols = pd.MultiIndex.from_product([[a.scorer], indiv, names, ["x", "y"]],
                                          names=["scorer", "individuals", "bodyparts", "coords"])
        data = np.full((len(imgnames), M, len(names), 2), np.nan)
        for r, (f, _) in enumerate(imgnames):
            js = [j for j in range(K) if present[f, j]]
            for s, j in enumerate(js[:M]): data[r, s] = labels[f, j]
        data = data.reshape(len(imgnames), -1)
    else:
        cols = pd.MultiIndex.from_product([[a.scorer], names, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
        data = np.array([labels[f, np.flatnonzero(present[f])[0]].reshape(-1) for f, _ in imgnames])
    df = pd.DataFrame(data, index=rows, columns=cols)
    fn = proj / "labeled-data" / stem / f"CollectedData_{a.scorer}"
    df.to_csv(fn.with_suffix(".csv"))
    df.to_hdf(fn.with_suffix(".h5"), key="df_with_missing", mode="w")
    print(f"wrote {fn}.csv/.h5  ({len(df)} frames, {len(names)} bodyparts, "
          f"{int((~np.isnan(data)).sum()//2)} labelled points)")

# ---------------------------------------------------------------- config.yaml
cfg_path = proj / "config.yaml"
project_path = a.project_path_in_config or str(proj.resolve())
video_key = str(Path(project_path) / "videos" / vid.name)
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["video_sets"][video_key] = {"crop": f"0, {W}, 0, {H}"}
    have = cfg["multianimalbodyparts"] if cfg.get("multianimalproject") else cfg["bodyparts"]
    if have != names: raise SystemExit("bodyparts in existing config differ from --npoints choice")
    if bool(cfg.get("multianimalproject")) != a.multi: raise SystemExit("existing project single/multi-animal type differs from --multi")
else:
    cfg = {
        "Task": a.task, "scorer": a.scorer, "date": datetime.date.today().strftime("%b%d"),
        "multianimalproject": a.multi, "identity": False if a.multi else None,
        "project_path": project_path, "engine": "pytorch",
        "video_sets": {video_key: {"crop": f"0, {W}, 0, {H}"}},
    }
    if a.multi:
        cfg.update({"individuals": [f"worm{i+1}" for i in range(M)], "uniquebodyparts": [],
                    "multianimalbodyparts": names, "bodyparts": "MULTI!"})
    else:
        cfg["bodyparts"] = names
    cfg.update({
        "start": 0, "stop": 1, "numframes2pick": 20,
        "skeleton": [[names[i], names[i + 1]] for i in range(len(names) - 1)],
        "skeleton_color": "black", "pcutoff": 0.6, "dotsize": 6 if a.npoints == 49 else 12,
        "alphavalue": 0.7, "colormap": "rainbow",
        "TrainingFraction": [0.95], "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "albumentations" if a.multi else "default",
        "snapshotindex": -1, "detector_snapshotindex": -1,
        "batch_size": 8, "detector_batch_size": 1,
        "cropping": False, "x1": 0, "x2": W, "y1": 0, "y2": H,
        "corner2move2": [50, 50], "move2corner": True,
    })
    if a.multi: cfg["default_track_method"] = "ellipse"
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print("config:", cfg_path)
