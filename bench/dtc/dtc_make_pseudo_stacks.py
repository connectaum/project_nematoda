"""Stage 1 of the coil pseudo-label set (2026-09-19): pick frames where the classical pipeline failed
(qc flags skeleton_branches / loop_or_empty, or a present track without a 'full' label) and dump 11-frame
preprocessed stacks (whole frame, scale 90/L_med, block 15 / C 1, "bright worm on black", uint8) to an hdf5.
Stage 2 (dtc_make_pseudo.py on apps, needs JAX) runs the shipped model on them and builds the training set.

  python dtc_make_pseudo_stacks.py <root> <out.hdf5> [--videos a,b] [--exclude IMG_8373,ga17Trimfortraining]
                                   [--per-video 300] [--min-gap 3] [--target-L 90]
hdf5: group per video: x (S,11,hs,ws) uint8, frame (S) int, flag (S) str; attrs scale, L_med, A_med, width, height
"""
import sys, json, argparse, time
import numpy as np, cv2, h5py, pandas as pd
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--videos", default=""); ap.add_argument("--exclude", default="IMG_8373,ga17Trimfortraining")
ap.add_argument("--per-video", type=int, default=300); ap.add_argument("--min-gap", type=int, default=3)
ap.add_argument("--target-L", type=float, default=90.0); ap.add_argument("--skip", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); ROOT = Path(a.root); SKIP = a.skip; HALF = 5 * SKIP; rng = np.random.default_rng(a.seed)
excl = set(a.exclude.split(","))
vids = a.videos.split(",") if a.videos else sorted(p.name for p in (ROOT / "auto_labels").iterdir() if (p / "qc_frames.csv").exists() and p.name not in excl)
h5 = h5py.File(a.out, "w"); T0 = time.time()
for v in vids:
    vp = next((p for p in (ROOT / "videos").glob(v + ".*") if p.suffix.lower() in (".mp4", ".mov")), None)
    if vp is None: print("no video", v); continue
    d = ROOT / "auto_labels" / v; qc = pd.read_csv(d / "qc_frames.csv"); su = json.load(open(d / "summary.json"))
    L_med = su["median_length_px"]; s = a.target_L / L_med
    fl = qc["flags"].fillna("").astype(str); st = qc["label_status"].fillna("").astype(str)
    coil = fl.str.contains("skeleton_branches|loop_or_empty").values
    nofull = (st.apply(lambda t: "full" not in t.split(";")) & (qc["n_worms"] >= 1)).values
    N = len(qc); cand = [c for c in range(HALF, N - HALF) if coil[c] or nofull[c]]
    rng.shuffle(cand); chosen = []
    for c in sorted(cand, key=lambda c: (not coil[c], rng.random())):   # coil frames first
        if all(abs(c - o) >= a.min_gap for o in chosen): chosen.append(c)
        if len(chosen) >= a.per_video: break
    chosen.sort()
    if not chosen: print(v, "no candidates"); continue
    print(f"{v}: {len(chosen)} centres of {len(cand)} candidates (N {N}, L_med {L_med:.0f}, scale {s:.3f})", flush=True)
    cap = cv2.VideoCapture(str(vp)); buf = {}; fi = -1
    def get_frame(f):
        global fi
        while fi < f:
            ok, fr = cap.read()
            if not ok: return None
            fi += 1; g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            mask = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1) == 0
            buf[fi] = (mask * (255 - small)).astype(np.uint8)
            for old in [j for j in buf if j < fi - 2 * HALF - 2]: buf.pop(old)
        return buf.get(f)
    xs, fs, fl_out = [], [], []
    for c in chosen:
        grays = [get_frame(f) for f in range(c - HALF, c + HALF + 1, SKIP)]
        if any(g is None for g in grays): break
        xs.append(np.stack(grays)); fs.append(c); fl_out.append(("coil;" if coil[c] else "") + ("nofull;" if nofull[c] else "") + fl[c])
    cap.release()
    g = h5.create_group(v); x = np.stack(xs)
    g.create_dataset("x", data=x, chunks=(1,) + x.shape[1:], compression="lzf"); g.create_dataset("frame", data=np.array(fs, np.int32))
    g.create_dataset("flag", data=np.array(fl_out, dtype=h5py.string_dtype()))
    g.attrs.update(dict(scale=s, L_med=L_med, A_med=su["median_area_px"], width=su["width"], height=su["height"], fps=su["fps"]))
    print(f"  {v}: {len(xs)} stacks {x.shape[2]}x{x.shape[3]} ({time.time() - T0:.0f} s)", flush=True)
h5.close(); print("done", a.out)
