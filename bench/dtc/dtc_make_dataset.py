"""Build a DeepTangleCrawl fine-tuning set from our auto_labels (2026-09-19).

  python dtc_make_dataset.py <project_root> <out.hdf5> [--videos IMG_8370,IMG_8373] [--step 3] [--per-video 1500]
                             [--crop 128x224] [--target-L 70,120] [--seed 0] [--sheet out.jpg]

Sample = 11 frames (centre c, skip=4 → c-20..c+20) of one video, downscaled so that the median worm is
target_L px long (random in [70,120] like the authors' ~80 px), preprocessed exactly like inference
(inverted gray × adaptive-threshold mask, "bright worm on black"), cropped to HxW around a labelled worm.
Labels = every track that has a 'full' 49-point midline on frames c-4, c, c+4 → (N,3,49,2) in crop px
(x, y), padded to NL=6 with -1. Foreground components that carry NO label (clusters, border-cut worms,
head_cut tracks) are erased from the frames, so the network never sees an unlabelled worm.

hdf5 layout: x (S,11,H,W) uint8, y (S,6,3,49,2) float32, meta_video (S) str, meta_frame (S) int,
meta_track (S) int, meta_scale (S) float; attrs crop, skip, target_L.
"""
import sys, os, json, argparse, time
import numpy as np, cv2, h5py, pandas as pd
from pathlib import Path
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--videos", default="")
ap.add_argument("--step", type=int, default=3, help="frame step between sample centres per track")
ap.add_argument("--per-video", type=int, default=1500)
ap.add_argument("--crop", default="128x224")
ap.add_argument("--target-L", default="70,120")
ap.add_argument("--skip", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sheet", default="")
ap.add_argument("--limit", type=int, default=0, help="debug: stop after N samples total")
ap.add_argument("--gate", default="video", help="label-length gate: 'video' = 0.75–1.25 x video median (v1, best), 'track' = per-track median (v2)")
ap.add_argument("--cut", type=float, default=0.3, help="prob. that the crop cuts the central worm (field-of-view edge cases)")
a = ap.parse_args()
ROOT = Path(a.root); H, W = map(int, a.crop.split("x")); TL0, TL1 = map(float, a.target_L.split(","))
SKIP = a.skip; NL = 6; NF = 11; HALF = 5 * SKIP
rng = np.random.default_rng(a.seed)
BLOCKS = [15, 21, 31]; CS = [1, 5, 15]

def preprocess(gray, block, C):
    """inference preprocessing (run_dtc.py): bright worm on black, thresholded on the light-bg image"""
    inv = 255 - gray
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C) == 0
    return mask, inv

def load_video_labels(v):
    d = ROOT / "auto_labels" / v
    lab = np.load(d / "labels49.npy")
    if lab.ndim == 3: lab = lab[:, None]
    qc = pd.read_csv(d / "qc_frames.csv")
    status = [str(s).split(";") for s in qc["label_status"].fillna("")]
    flags = [str(f) if isinstance(f, str) else "" for f in qc["flags"]]
    summ = json.load(open(d / "summary.json"))
    return lab, status, flags, summ

def full_at(status, f, k):
    return f < len(status) and k < len(status[f]) and status[f][k] == "full"

def rasterize(pts, shape, thick):
    """polyline of (x,y) points → boolean mask"""
    m = np.zeros(shape, np.uint8)
    p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [p], False, 1, thickness=max(1, int(thick)))
    return m.astype(bool)

vids = a.videos.split(",") if a.videos else sorted(p.name for p in (ROOT / "auto_labels").iterdir() if (p / "labels49.npy").exists())
h5 = h5py.File(a.out, "w")
X = h5.create_dataset("x", (0, NF, H, W), np.uint8, maxshape=(None, NF, H, W), chunks=(1, NF, H, W), compression="lzf")
Y = h5.create_dataset("y", (0, NL, 3, 49, 2), np.float32, maxshape=(None, NL, 3, 49, 2))
MV = h5.create_dataset("meta_video", (0,), h5py.string_dtype(), maxshape=(None,))
MF = h5.create_dataset("meta_frame", (0,), np.int32, maxshape=(None,))
MT = h5.create_dataset("meta_track", (0,), np.int32, maxshape=(None,))
MS = h5.create_dataset("meta_scale", (0,), np.float32, maxshape=(None,))
h5.attrs.update(dict(crop=f"{H}x{W}", skip=SKIP, target_L=a.target_L, nframes=NF))
def append(x, y, v, f, k, s):
    n = X.shape[0]
    for ds in (X, Y, MV, MF, MT, MS): ds.resize(n + 1, axis=0)
    X[n] = x; Y[n] = y; MV[n] = v; MF[n] = f; MT[n] = k; MS[n] = s

sheet = []; total = 0; T0 = time.time()
for v in vids:
    vp = next((p for p in (ROOT / "videos").glob(v + ".*") if p.suffix.lower() in (".mp4", ".mov")), None)
    if vp is None: print("no video for", v); continue
    lab, status, flags, summ = load_video_labels(v)
    N, K = lab.shape[:2]; L_med = summ["median_length_px"]; A_med = summ["median_area_px"]
    # width proxy (px, original): area / length
    width0 = A_med / L_med
    # per-track reference length (videos mix worm sizes: IMG_8374 has 255–330 px worms next to 530 px ones)
    L_ref = np.full(K, L_med)
    for kk in range(K):
        Ls_k = [np.hypot(*np.diff(lab[f, kk], axis=0).T).sum() for f in range(N) if full_at(status, f, kk) and np.isfinite(lab[f, kk]).all()]
        if len(Ls_k) >= 10: L_ref[kk] = float(np.median(Ls_k))
    # candidate centres: (c, k) with full labels on c-SKIP, c, c+SKIP
    cands = []
    for k in range(K):
        for c in range(HALF, N - HALF, a.step):
            if all(full_at(status, f, k) and np.isfinite(lab[f, k]).all() for f in (c - SKIP, c, c + SKIP)):
                cands.append((c, k))
    if not cands: print(v, "no candidates"); continue
    rng.shuffle(cands); cands = sorted(cands[:a.per_video])
    print(f"{v}: {len(cands)} centres (of {N} frames, {K} tracks, L_med {L_med:.0f})", flush=True)
    cap = cv2.VideoCapture(str(vp)); buf = {}; fi = -1
    def get_frame(f):
        global fi
        while fi < f:
            ok, fr = cap.read()
            if not ok: return None
            fi += 1; buf[fi] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            for old in [j for j in buf if j < fi - 2 * HALF - 2]: buf.pop(old)
        return buf.get(f)
    made = 0
    for c, k in cands:
        frames_idx = list(range(c - HALF, c + HALF + 1, SKIP))
        grays = [get_frame(f) for f in frames_idx]
        if any(g is None for g in grays): break
        tL = rng.uniform(TL0, TL1); s = tL / L_med
        block = BLOCKS[rng.integers(len(BLOCKS))]; C = CS[rng.integers(len(CS))]
        rot = rng.random() < 0.25          # rot90 of the whole frame before cropping
        thick = max(2, width0 * s * 0.8)
        # scaled frames + per-frame masks
        pre = []; ok_sample = True
        # labels available per frame (any track, any status full) in scaled px, for masking unlabelled foreground
        def labels_scaled(f):
            out = {}
            for kk in range(K):
                ff = None
                for d in (0, -1, 1, -2, 2, -3, 3, -4, 4):
                    if full_at(status, f + d, kk) and np.isfinite(lab[f + d, kk]).all(): ff = f + d; break
                if ff is not None: out[kk] = lab[ff, kk] * s
            return out
        for f, g in zip(frames_idx, grays):
            small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if rot: small = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
            mask, inv = preprocess(small, block, C)
            hs, ws = small.shape
            # erase foreground components that carry no label
            labs = labels_scaled(f)
            if rot:  # (x,y) → (hs_orig - y, x) after clockwise rotation: new_x = h_old-1-y, new_y = x
                labs = {kk: np.stack([ (ws - 1) - p[:, 1], p[:, 0] ], -1) for kk, p in labs.items()}
                # careful: after rotation, new width ws == old height
            comp, ncomp = ndi.label(mask)
            keep = np.zeros(ncomp + 1, bool)
            for kk, p in labs.items():
                r = rasterize(p, (hs, ws), thick)
                ids = np.unique(comp[r]); keep[ids] = True
            keep[0] = False
            mask2 = keep[comp]
            pre.append(mask2 * inv.astype(np.float32) / 255.0)
            if f == c:
                # the labelled worm must sit in a component of single-worm size (else it is a cluster child)
                pk = lab[c, k] * s
                if rot: pk = np.stack([(ws - 1) - pk[:, 1], pk[:, 0]], -1)
                r = rasterize(pk, (hs, ws), thick); ids = [i for i in np.unique(comp[r]) if i > 0]
                if not ids: ok_sample = False
                else:
                    area = sum((comp == i).sum() for i in ids)
                    if area > 1.6 * A_med * s * s: ok_sample = False
        if not ok_sample: continue
        pre = np.stack(pre); _, hs, ws = pre.shape
        # label-quality gate on the centre frame: every labelled worm's component must be covered by its
        # own midline (a loop/coil with a wrong straight-through skeleton leaves the ring uncovered) and the
        # label length must be close to the video's median (0.75–1.25 L_med)
        mid = pre[NF // 2] > 0; comp, ncomp = ndi.label(mid); wpx = max(2.0, width0 * s)
        bad = False
        for kk in range(K):
            if not (full_at(status, c, kk) and np.isfinite(lab[c, kk]).all()): continue
            P = lab[c, kk] * s
            if rot: P = np.stack([(ws - 1) - P[:, 1], P[:, 0]], -1)
            Lk = np.hypot(*np.diff(P, axis=0).T).sum() / s
            if a.gate == "track":
                if not (0.8 * L_ref[kk] <= Lk <= 1.2 * L_ref[kk]) or Lk < 0.35 * L_med: bad = True; break
            elif not (0.75 * L_med <= Lk <= 1.25 * L_med): bad = True; break
            r = rasterize(P, (hs, ws), 1); ids = [i for i in np.unique(comp[r]) if i > 0]
            if not ids: continue
            own = np.isin(comp, ids)
            near = cv2.dilate(r.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(round(wpx * 1.3))).astype(bool)
            if (own & near).sum() < 0.92 * own.sum(): bad = True; break
        if bad: continue
        # labels for the 3 central frames, all tracks with full labels there
        Ls = []
        for kk in range(K):
            fs = (c - SKIP, c, c + SKIP)
            if all(full_at(status, f, kk) and np.isfinite(lab[f, kk]).all() for f in fs):
                P = np.stack([lab[f, kk] for f in fs]) * s        # (3,49,2)
                if rot: P = np.stack([(ws - 1) - P[..., 1], P[..., 0]], -1)
                Ls.append((kk, P))
        # crop placement: around worm k's centre on frame c, worm fully inside with margin
        Pk = dict(Ls)[k]; lo = Pk.reshape(-1, 2).min(0); hi = Pk.reshape(-1, 2).max(0); m = 4
        # x0 range so that [lo-m, hi+m] ⊂ [x0, x0+W)
        x0_lo, x0_hi = hi[0] + m - W, lo[0] - m; y0_lo, y0_hi = hi[1] + m - H, lo[1] - m
        if x0_lo > x0_hi or y0_lo > y0_hi: continue    # worm larger than crop (should not happen at these scales)
        x0 = int(round(rng.uniform(x0_lo, x0_hi))); y0 = int(round(rng.uniform(y0_lo, y0_hi)))
        if rng.random() < a.cut:                        # push the crop so that 15–50 % of the worm's extent is outside
            ext = hi - lo; side = rng.integers(4); frac = rng.uniform(0.15, 0.5)
            if side == 0: x0 = int(round(hi[0] + m - W + frac * ext[0]))
            elif side == 1: x0 = int(round(lo[0] - m - frac * ext[0]))
            elif side == 2: y0 = int(round(hi[1] + m - H + frac * ext[1]))
            else: y0 = int(round(lo[1] - m - frac * ext[1]))
        # if the scaled frame is smaller than the crop, allow negative offsets (zero padding)
        x0 = int(np.clip(x0, min(0, ws - W), max(0, ws - W))) if ws >= W else int(np.clip(x0, ws - W, 0))
        y0 = int(np.clip(y0, min(0, hs - H), max(0, hs - H))) if hs >= H else int(np.clip(y0, hs - H, 0))
        crop = np.zeros((NF, H, W), np.float32)
        sy0, sy1 = max(0, y0), min(hs, y0 + H); sx0, sx1 = max(0, x0), min(ws, x0 + W)
        crop[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = pre[:, sy0:sy1, sx0:sx1]
        y = np.full((NL, 3, 49, 2), -1.0, np.float32)
        # order: worm k first, then others by distance of their centre to the crop centre
        cc = np.array([x0 + W / 2, y0 + H / 2])
        Ls.sort(key=lambda t: (t[0] != k, np.linalg.norm(t[1][1].mean(0) - cc)))
        for i, (kk, P) in enumerate(Ls[:NL]):
            y[i] = P - [x0, y0]
        # a label entirely outside the crop is useless → mark as padding
        for i in range(NL):
            out = (y[i][..., 0] >= W) | (y[i][..., 1] >= H) | (y[i][..., 0] < 0) | (y[i][..., 1] < 0)
            if (y[i] >= 0).any() and out.all(): y[i] = -1
        append(np.clip(crop * 255 + 0.5, 0, 255).astype(np.uint8), y, v, c, k, s)
        made += 1; total += 1
        if a.sheet and len(sheet) < 24 and rng.random() < 0.05: sheet.append((crop[NF // 2], y, f"{v} f{c} k{k} s{s:.3f}"))
        if a.limit and total >= a.limit: break
    cap.release()
    print(f"  {v}: {made} samples ({time.time() - T0:.0f} s, total {total})", flush=True)
    if a.limit and total >= a.limit: break
h5.close()
print("done", total, "samples →", a.out)
if a.sheet and sheet:
    tiles = []
    for img, y, txt in sheet:
        t = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i in range(NL):
            if (y[i] < 0).all(): continue
            p = np.round(y[i][1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(t, [p], False, (0, 255, 0) if i == 0 else (0, 200, 255), 1)
            cv2.circle(t, tuple(p[0, 0]), 2, (255, 0, 0), -1)
        cv2.putText(t, txt, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(a.sheet, cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print("sheet", a.sheet)
