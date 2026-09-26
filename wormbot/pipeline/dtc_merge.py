"""Merge DeepTangleCrawl midlines (dtc_infer.py) into a video's auto_labels.

  python dtc_merge.py <auto_labels_dir> <dtc.pkl> [--fix-flagged] [--min-new 20]

What it does
* plausible DTC midlines (length 0.6–1.3 × median worm length) are linked frame-to-frame into tracklets;
* each tracklet is assigned to the pipeline track it overlaps best (median midline distance on frames where
  both exist < 0.3 L), tracklets that match no track and last >= --min-new frames become NEW track columns;
* labels49.npy: frames where the assigned track has NO label get the DTC midline (status 'dtc');
  with --fix-flagged, labels on frames flagged skeleton_branches / loop_or_empty / length_outlier that differ
  from the DTC midline by more than one body width are REPLACED (status 'dtc_fix'); the previous file is kept
  as labels49_before_dtc.npy;
* qc_frames.csv: flag 'from_dtc' on touched frames; tracks.json / summary.json updated (summary['dtc']).
Prints one summary line. Re-run worm_kinematics.py / render_overlay.py afterwards.
"""
import csv, json, os, sys, pickle, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("labels"); ap.add_argument("dtc")
ap.add_argument("--fix-flagged", action="store_true", default=os.environ.get("DTC_FIX_FLAGGED", "1") == "1")
ap.add_argument("--min-new", type=int, default=60)
ap.add_argument("--min-new-score", type=float, default=0.5, help="median score a tracklet needs to become a NEW track")
ap.add_argument("--min-score", type=float, default=0.3)
a = ap.parse_args()
L = a.labels
FIX_FLAGS = {"skeleton_branches", "loop_or_empty", "length_outlier"}


def mdist(u, v):
    return min(np.linalg.norm(u - v, axis=1).mean(), np.linalg.norm(u - v[::-1], axis=1).mean())


def orient_like(p, ref):
    if ref is None or np.isnan(ref).any():
        return p
    return p if np.mean(np.hypot(*(p - ref).T)) <= np.mean(np.hypot(*(p[::-1] - ref).T)) else p[::-1]


d = pickle.load(open(a.dtc, "rb")); frames = d["frames"]
lab = np.load(os.path.join(L, "labels49.npy")).astype(np.float32)
if lab.ndim == 3: lab = lab[:, None]
N, K = lab.shape[:2]
summ = json.load(open(os.path.join(L, "summary.json"))); tracks = json.load(open(os.path.join(L, "tracks.json")))
L_med = float(summ["median_length_px"]); width = float(summ.get("median_area_px", 20 * L_med)) / L_med
rows = list(csv.DictReader(open(os.path.join(L, "qc_frames.csv")))); fields = list(rows[0].keys())
flags_of = {int(r["frame"]): set(x for x in r["flags"].split(";") if x) for r in rows}

# ---------------------------------------------------------------- 1. plausible midlines → tracklets
def length(w): return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
dets = {f: [m for m in ms if m["s"] >= a.min_score and 0.6 * L_med <= length(m["w"]) <= 1.3 * L_med]
        for f, ms in frames.items()}
LINK = max(60.0, 0.15 * L_med); GAP = 3
tracklets = []                       # each: dict(items={f: (w, s)}, last=f)
for f in sorted(dets):
    active = [t for t in tracklets if f - t["last"] <= GAP]
    pairs = sorted(((mdist(m["w"], t["items"][t["last"]][0]), i, j) for i, m in enumerate(dets[f]) for j, t in enumerate(active)),
                   key=lambda p: p[0])
    used_m, used_t = set(), set()
    for dist, i, j in pairs:
        if dist > LINK or i in used_m or j in used_t: continue
        t = active[j]; t["items"][f] = (dets[f][i]["w"], dets[f][i]["s"]); t["last"] = f
        used_m.add(i); used_t.add(j)
    for i, m in enumerate(dets[f]):
        if i not in used_m: tracklets.append(dict(items={f: (m["w"], m["s"])}, last=f))
tracklets = [t for t in tracklets if len(t["items"]) >= 3]

# ---------------------------------------------------------------- 2. tracklet → pipeline track
ASSIGN = 0.3 * L_med
assign = {}                          # tracklet index -> column
new_cols = []
for ti, t in enumerate(tracklets):
    best = None
    for col in range(K):
        ds = [mdist(w, lab[f, col]) for f, (w, s) in t["items"].items() if not np.isnan(lab[f, col]).any()]
        if len(ds) >= 3:
            med = float(np.median(ds))
            if med < ASSIGN and (best is None or med < best[0]): best = (med, col)
    if best is not None: assign[ti] = best[1]
    elif len(t["items"]) >= a.min_new and np.median([sc for (w, sc) in t["items"].values()]) >= a.min_new_score: new_cols.append(ti)
if new_cols:
    lab = np.concatenate([lab, np.full((N, len(new_cols), 49, 2), np.nan, np.float32)], axis=1)
    for j, ti in enumerate(new_cols): assign[ti] = K + j
K2 = lab.shape[1]
if not os.path.exists(os.path.join(L, "labels49_before_dtc.npy")):
    np.save(os.path.join(L, "labels49_before_dtc.npy"), lab[:, :K] if K2 > K else lab)

# ---------------------------------------------------------------- 3. write midlines
touched = {}                         # (f, col) -> 'dtc' | 'dtc_fix'
cands = {}                           # (f, col) -> list of (score, w)
for ti, col in assign.items():
    for f, (w, s) in tracklets[ti]["items"].items():
        cands.setdefault((f, col), []).append((s, w))
n_fill = n_fix = 0
for col in range(K2):
    fs = sorted(f for (f, c) in cands if c == col)
    for f in fs:
        s, w = max(cands[(f, col)], key=lambda t: t[0])
        have = not np.isnan(lab[f, col]).any()
        if have:
            if not a.fix_flagged or not (flags_of.get(f, set()) & FIX_FLAGS) or s < 0.5: continue
            if mdist(w, lab[f, col]) <= width: continue
            kind = "dtc_fix"
        else:
            kind = "dtc"
        # orientation: nearest untouched-or-already-set label of this track, before then after
        ref = None
        before = [g for g in range(f - 1, max(-1, f - 30), -1) if not np.isnan(lab[g, col]).any()]
        after = [g for g in range(f + 1, min(N, f + 30)) if not np.isnan(lab[g, col]).any()]
        if before: ref = lab[before[0], col]
        elif after: ref = lab[after[0], col]
        lab[f, col] = orient_like(w.astype(np.float32), ref)
        touched[(f, col)] = kind
        if kind == "dtc": n_fill += 1
        else: n_fix += 1
np.save(os.path.join(L, "labels49.npy"), lab if K2 > 1 else lab[:, 0])

# curvature.npy must stay aligned with labels49 (make_dlc_project indexes both by track): pad new columns and
# recompute the touched frames with the pipeline's formula (smoothed gradients, kappa * arc length)
from scipy.ndimage import gaussian_filter1d
cpath = os.path.join(L, "curvature.npy")
curv = np.load(cpath).astype(np.float32) if os.path.exists(cpath) else np.full((N, K, 49), np.nan, np.float32)
if curv.ndim == 2: curv = curv[:, None]
if curv.shape[1] < K2: curv = np.concatenate([curv, np.full((N, K2 - curv.shape[1], 49), np.nan, np.float32)], axis=1)
for (f, col) in touched:
    dpt = lab[f, col]
    x = gaussian_filter1d(dpt[:, 0], 1.5); y = gaussian_filter1d(dpt[:, 1], 1.5)
    dx, dy = np.gradient(x), np.gradient(y); ddx, ddy = np.gradient(dx), np.gradient(dy)
    Lk = float(np.linalg.norm(np.diff(dpt, axis=0), axis=1).sum())
    curv[f, col] = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5) * Lk
np.save(cpath, curv if K2 > 1 else curv[:, 0])

# ---------------------------------------------------------------- 4. qc_frames.csv / tracks.json / summary.json
frames_touched = {}
for (f, col), kind in touched.items(): frames_touched.setdefault(f, {})[col] = kind
for r in rows:
    f = int(r["frame"])
    st = r["label_status"].split(";") if K > 1 else [r["label_status"]]
    st = (st + ["-"] * K2)[:K2]
    if f in frames_touched:
        for col, kind in frames_touched[f].items(): st[col] = kind
        fl = set(x for x in r["flags"].split(";") if x); fl.add("from_dtc"); r["flags"] = ";".join(sorted(fl))
        r["n_worms"] = str(sum(1 for s in st if s and s not in ("-", "no_midline", "both_ends_cut", "visible_longer_than_body")))
    r["label_status"] = ";".join(st) if K2 > 1 else st[0]
with open(os.path.join(L, "qc_frames.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
for j, ti in enumerate(new_cols):
    tracks.append(dict(id=f"dtc_{K + j + 1}", n_frames=0, first_frame=None, last_frame=None, body_length_px=None,
                       origin="DeepTangleCrawl tracklet with no matching pipeline track"))
for col in range(K2):
    fr = np.where(~np.isnan(lab[:, col, 0, 0]))[0]
    if col < len(tracks):
        tracks[col]["n_frames"] = int(len(fr))
        if len(fr): tracks[col]["first_frame"], tracks[col]["last_frame"] = int(fr[0]), int(fr[-1])
        tracks[col]["dtc_frames"] = int(sum(1 for (f, c) in touched if c == col))
json.dump(tracks, open(os.path.join(L, "tracks.json"), "w"), indent=2)
summ["n_tracks"] = K2; summ["tracks"] = tracks
summ.setdefault("flag_counts", {})["from_dtc"] = len(frames_touched)
summ["frames_labelled"] = int(np.any(~np.isnan(lab[:, :, 0, 0]), axis=1).sum())
summ["dtc"] = dict(ensemble=d.get("ensemble"), ft=os.path.basename(d.get("ft") or ""), scale=d.get("scale"), thr=d.get("thr"),
                   tracklets=len(tracklets), assigned=len(assign), new_tracks=len(new_cols), frames_filled=n_fill,
                   frames_fixed=n_fix, secs=d.get("secs"))
json.dump(summ, open(os.path.join(L, "summary.json"), "w"), indent=2, ensure_ascii=False)
print(f"DTC merged: {n_fill} frames filled + {n_fix} fixed in {len(frames_touched)} frames, "
      f"{len(tracklets)} tracklets ({len(assign) - len(new_cols)} matched, {len(new_cols)} new tracks); tracks now {K2}")
