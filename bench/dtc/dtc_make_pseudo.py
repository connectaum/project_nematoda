"""Stage 2 of the coil pseudo-label set: run the SHIPPED model on pseudo_stacks.hdf5 (dtc_make_pseudo_stacks.py on the
Mac: whole-frame 11-frame stacks at 90 px / L_med, inference preprocessing) and build samples in our training format.

  docker run --rm -v ~/dtc:/work dtc:latest python /work/dtc_make_pseudo.py /work/authors/pseudo_stacks.hdf5 /work/dtc_pseudo.hdf5 [--sheet x.jpg]

Labels = base-model predictions (NMS thr 0.5) with score >= 0.6 and middle-frame length 0.7–1.2 x 90 px.
Components of the foreground that carry no kept label, or that are ambiguous (a prediction with 0.3 <= score < 0.6
on them that is not a kept one, or area > 1.6 x single-worm area x n_labels) are erased together with their labels.
One 128x224 crop per stack around a random kept label (labels shifted, padded to 6 with -1).
"""
import sys, time, pickle, argparse; sys.path.insert(0, "/work/tierpsy_tracker_2.0")
import numpy as np, cv2, h5py, jax, jax.numpy as jnp
from pathlib import Path
from scipy import ndimage as ndi
from deeptangle.forward import build_model
from deeptangle import utils
from deeptangle.predict import non_max_suppression
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--sheet", default="")
ap.add_argument("--crop", default="128x224"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--target-L", type=float, default=90.0)
a = ap.parse_args(); H, W = map(int, a.crop.split("x")); NL, NF = 6, 11; rng = np.random.default_rng(a.seed)
MP = Path("/work/model/DT_C_data_model/model/parameters_Final_mixed_train_data_wt_hand_ann_pca_92_epochs_250_2024-04-19_96_cutoff")
A = jnp.load(MP / "eigenworms_transform.npy"); forward_fn = build_model(A, 8, 8, 11)
tree = pickle.load(open(MP / "tree.pkl", "rb")); leaves, treedef = jax.tree_util.tree_flatten(tree)
with open(MP / "arrays.npy", "rb") as f: flat = [jnp.load(f) for _ in leaves]
params, st, _ = utils.single_from_sharded(jax.tree_util.tree_unflatten(treedef, flat))
fwd = jax.jit(lambda x: forward_fn.apply(params, st, x, is_training=False)[0])

def rasterize(pts, shape, thick):
    m = np.zeros(shape, np.uint8); cv2.polylines(m, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 1, thickness=max(1, int(thick))); return m.astype(bool)

src = h5py.File(a.inp, "r"); h5 = h5py.File(a.out, "w")
X = h5.create_dataset("x", (0, NF, H, W), np.uint8, maxshape=(None, NF, H, W), chunks=(1, NF, H, W), compression="lzf")
Y = h5.create_dataset("y", (0, NL, 3, 49, 2), np.float32, maxshape=(None, NL, 3, 49, 2))
MV = h5.create_dataset("meta_video", (0,), h5py.string_dtype(), maxshape=(None,)); MF = h5.create_dataset("meta_frame", (0,), np.int32, maxshape=(None,))
MT = h5.create_dataset("meta_track", (0,), np.int32, maxshape=(None,)); MS = h5.create_dataset("meta_scale", (0,), np.float32, maxshape=(None,))
MG = h5.create_dataset("meta_flag", (0,), h5py.string_dtype(), maxshape=(None,))
h5.attrs.update(dict(crop=f"{H}x{W}", skip=4, target_L=str(a.target_L), nframes=NF, source="pseudo labels from shipped model"))
def append(x, y, v, f, k, s, fl):
    n = X.shape[0]
    for ds in (X, Y, MV, MF, MT, MS, MG): ds.resize(n + 1, axis=0)
    X[n] = x; Y[n] = y; MV[n] = v; MF[n] = f; MT[n] = k; MS[n] = s; MG[n] = fl
sheet = []; stats = dict(stacks=0, no_label=0, samples=0, labels=0, erased=0, coil_samples=0); T0 = time.time()
for v in src:
    g = src[v]; xs = g["x"]; frames = g["frame"][:]; flags = g["flag"][:].astype(str); s = float(g.attrs["scale"]); A1 = float(g.attrs["A_med"]) * s * s
    S, _, hs, ws = xs.shape; pb, pr = (-hs) % 16, (-ws) % 16; width = A1 / a.target_L; thick = max(2.0, width * 0.8)
    made = 0
    for b0 in range(0, S, 8):
        xb = xs[b0:b0 + 8].astype(np.float32) / 255.0; xb = np.pad(xb, ((0, 0), (0, 0), (0, pb), (0, pr)))
        wa, sa, pa = map(np.asarray, fwd(jnp.asarray(xb)))
        for i in range(len(xb)):
            stats["stacks"] += 1
            keep = non_max_suppression((wa[i], sa[i], pa[i]), 0.3, 0.5, 96)
            preds = [(wa[i][k], float(sa[i][k])) for k in np.nonzero(keep)[0]] if keep.dtype == bool else [(wa[i][k], float(sa[i][k])) for k in keep]
            good = []; weak = []
            for w, sc in preds:
                L = np.hypot(*np.diff(w[1], axis=0).T).sum()
                if sc >= 0.6 and 0.7 * a.target_L <= L <= 1.2 * a.target_L: good.append(w)
                else: weak.append(w)
            if not good: stats["no_label"] += 1; continue
            pre = xb[i, :, :hs, :ws].copy()
            # per-frame cleanup: components not covered by a good label (of the nearest labelled frame) → erased;
            # ambiguous components (weak prediction on them, or too big for their labels) → erased + labels dropped
            drop = np.zeros(len(good), bool)
            for f in list(range(NF)) * 2:   # pass 1 decides which labels to drop, pass 2 erases with the final label set
                ref = 0 if f < 4 else (1 if f < 7 else 2); comp, nc = ndi.label(pre[f] > 0)
                cover = np.zeros((len(good), nc + 1), bool)
                for j, w in enumerate(good):                        # covered = >= 25 % of the midline pixels lie in the component
                    ids, cnt = np.unique(comp[rasterize(w[ref], (hs, ws), 1)], return_counts=True); tot = cnt.sum()
                    cover[j, ids[cnt >= 0.25 * tot]] = True
                cover[:, 0] = False; keepc = cover.any(0); area = ndi.sum(np.ones_like(comp), comp, np.arange(nc + 1))
                for w in weak:
                    ids = np.unique(comp[rasterize(w[ref], (hs, ws), 1)]); ids = ids[ids > 0]
                    for ci in ids:
                        if keepc[ci]: keepc[ci] = False; drop |= cover[:, ci]
                for ci in range(1, nc + 1):
                    if keepc[ci] and area[ci] > 1.3 * A1 * cover[:, ci].sum(): keepc[ci] = False; drop |= cover[:, ci]
                if f == NF - 1 and not drop.any(): pass
                keepc[0] = False
                if drop.any(): keepc &= cover[~drop].any(0) | (np.arange(nc + 1) == 0)
                stats["erased"] += int((~keepc[1:]).sum()); pre[f] *= keepc[comp]
            labs = [w for w, d in zip(good, drop) if not d]
            if not labs: stats["no_label"] += 1; continue
            k = rng.integers(len(labs)); P = labs[k]; lo, hi = P[1].min(0), P[1].max(0); m = 6
            x0_lo, x0_hi = hi[0] + m - W, lo[0] - m; y0_lo, y0_hi = hi[1] + m - H, lo[1] - m
            x0 = int(round(rng.uniform(min(x0_lo, x0_hi), max(x0_lo, x0_hi)))); y0 = int(round(rng.uniform(min(y0_lo, y0_hi), max(y0_lo, y0_hi))))
            x0 = int(np.clip(x0, min(0, ws - W), max(0, ws - W))) if ws >= W else int(np.clip(x0, ws - W, 0))
            y0 = int(np.clip(y0, min(0, hs - H), max(0, hs - H))) if hs >= H else int(np.clip(y0, hs - H, 0))
            crop = np.zeros((NF, H, W), np.float32); sy0, sy1 = max(0, y0), min(hs, y0 + H); sx0, sx1 = max(0, x0), min(ws, x0 + W)
            crop[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = pre[:, sy0:sy1, sx0:sx1]
            cc = np.array([x0 + W / 2, y0 + H / 2]); order = sorted(range(len(labs)), key=lambda j: (j != k, np.linalg.norm(labs[j][1].mean(0) - cc)))
            y = np.full((NL, 3, 49, 2), -1.0, np.float32)
            for n, j in enumerate(order[:NL]):
                yy = labs[j] - [x0, y0]
                if ((yy[..., 0] >= W) | (yy[..., 1] >= H) | (yy < 0).any(-1)).all(): continue
                y[n] = yy
            append(np.clip(crop * 255 + 0.5, 0, 255).astype(np.uint8), y, v, int(frames[b0 + i]), int(k), s, flags[b0 + i])
            made += 1; stats["samples"] += 1; stats["labels"] += len(labs); stats["coil_samples"] += int("coil" in flags[b0 + i])
            if a.sheet and len(sheet) < 32 and rng.random() < 0.06: sheet.append((crop[NF // 2], y, f"{v} f{frames[b0 + i]} {flags[b0 + i][:12]}"))
    print(f"{v}: {made}/{S} samples ({time.time() - T0:.0f} s)", flush=True)
h5.close(); print("done", stats)
if a.sheet and sheet:
    tiles = []
    for img, y, txt in sheet:
        t = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i in range(NL):
            if (y[i] < 0).all(): continue
            p = np.round(y[i][1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(t, [p], False, (0, 255, 0) if i == 0 else (0, 200, 255), 1); cv2.circle(t, tuple(p[0, 0]), 2, (255, 0, 0), -1)
        cv2.putText(t, txt, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1); tiles.append(t)
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(a.sheet, cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)); print("sheet", a.sheet)
