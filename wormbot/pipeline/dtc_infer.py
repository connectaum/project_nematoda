"""DeepTangleCrawl (DTC) inference on a whole video: per-frame 49-point midlines through contacts and coils.

  python dtc_infer.py <video.mp4> <auto_labels_dir> <out.pkl> [--model-dir DIR] [--ft ckpt.pkl] [--thr 0.3]
                      [--target-L 90] [--batch 8] [--no-ensemble]

* the video is downscaled so that the median worm (summary.json median_length_px) is --target-L px long
  (the shipped model is tied to ~80–100 px worms), preprocessed like Tierpsy 2.0 (inverted gray × adaptive
  threshold mask), and fed as 11-frame stacks with skip = round(fps/5) frames;
* two models are run: the shipped checkpoint (--model-dir) and our fine-tuned one (--ft); each gets its own
  NMS, then the two sets are united geometrically (midlines closer than --dedup px belong to the same worm).
  Which midline of such a duplicate survives is --merge / DTC_MERGE: the base model's midlines are
  systematically ~7 % too short, so the old "highest score wins" rule (score) made the ensemble inherit that
  bias. ftsafe (default) keeps the fine-tuned model's midline, but falls back to the base one when the ft
  length is outside 0.85-1.15 x L_med and the base one is closer to L_med (guards against a midline that ran
  onto a touching worm); ftfirst always keeps the ft one; longest keeps the longest.
  Either model alone with --no-ensemble / no --ft;
* output pickle: dict(frames={f: [dict(w=(49,2) float32 original px, s=score, m='base'|'ft')]}, scale, skip,
  target_L, L_med, model_dir, ft, secs).
Frames closer than 5·skip to the start/end use a clamped (edge-repeated) window.
"""
import sys, os, time, json, pickle, argparse
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax, jax.numpy as jnp
from deeptangle.forward import build_model
from deeptangle import utils
from deeptangle.predict import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("labels"); ap.add_argument("out")
ap.add_argument("--model-dir", default=os.environ.get("DTC_MODEL_DIR", "/models/dtc/base"))
ap.add_argument("--ft", default=os.environ.get("DTC_FT", ""))
ap.add_argument("--no-ensemble", action="store_true")
ap.add_argument("--thr", type=float, default=float(os.environ.get("DTC_THR", "0.3")))
ap.add_argument("--target-L", type=float, default=90.0)
ap.add_argument("--block", type=int, default=15); ap.add_argument("--C", type=int, default=1)
ap.add_argument("--cutoff", type=int, default=96); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--dedup", type=float, default=60.0)
ap.add_argument("--merge", default=os.environ.get("DTC_MERGE", "ftsafe"),
                choices=["ftsafe", "ftfirst", "longest", "score"])
a = ap.parse_args()

summ = json.load(open(Path(a.labels) / "summary.json"))
L_med = float(summ["median_length_px"]); fps = float(summ.get("fps") or 20.0)
scale = a.target_L / L_med; skip = max(1, int(round(fps / 5)))
MP = Path(a.model_dir)
A = jnp.load(MP / "eigenworms_transform.npy"); forward_fn = build_model(A, 8, 8, 11)
tree = pickle.load(open(MP / "tree.pkl", "rb")); leaves, treedef = jax.tree_util.tree_flatten(tree)
with open(MP / "arrays.npy", "rb") as f: flat = [jnp.load(f) for _ in leaves]
params, st, _ = utils.single_from_sharded(jax.tree_util.tree_unflatten(treedef, flat))
models = [("base", jax.jit(lambda x, p=params, s=st: forward_fn.apply(p, s, x, is_training=False)[0]))]
if a.ft:
    ck = pickle.load(open(a.ft, "rb"))
    p2, s2 = jax.tree_util.tree_map(jnp.asarray, ck["params"]), jax.tree_util.tree_map(jnp.asarray, ck["state"])
    ft_fn = jax.jit(lambda x, p=p2, s=s2: forward_fn.apply(p, s, x, is_training=False)[0])
    models = [("ft", ft_fn)] if a.no_ensemble else models + [("ft", ft_fn)]
print(f"DTC: L_med {L_med:.0f} px → scale {scale:.4f}, skip {skip}, models {[m for m, _ in models]}", flush=True)

def mdist(u, v):
    return min(np.linalg.norm(u - v, axis=1).mean(), np.linalg.norm(u - v[::-1], axis=1).mean())

def mlen(w):
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())

# ---------------------------------------------------------------- read + preprocess the whole video at network scale
cap = cv2.VideoCapture(a.video); pre = []
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, a.block, a.C) == 0
    pre.append((mask * (255 - small)).astype(np.uint8))
cap.release()
T = len(pre); h, w = pre[0].shape; pb, pr = (-h) % 16, (-w) % 16
pre = np.pad(np.stack(pre), ((0, 0), (0, pb), (0, pr)))
print(f"  {T} frames at {w}x{h} (+pad → {w + pr}x{h + pb})", flush=True)

def window(c):
    idx = np.clip(np.arange(c - 5 * skip, c + 5 * skip + 1, skip), 0, T - 1)
    return pre[idx].astype(np.float32) / 255.0

frames = {}; t0 = time.time(); centers = list(range(T))
for b0 in range(0, T, a.batch):
    cs = centers[b0:b0 + a.batch]
    x = jnp.asarray(np.stack([window(c) for c in cs]))
    per_model = []
    for name, fn in models:
        wa, sa, pa = map(np.asarray, fn(x)); per_model.append((name, wa, sa, pa))
    for i, c in enumerate(cs):
        cand = []
        for name, wa, sa, pa in per_model:
            keep = non_max_suppression((wa[i], sa[i], pa[i]), a.thr, 0.5, a.cutoff)
            for wk, sk in zip(wa[i][keep][:, 1] / scale, sa[i][keep]):
                cand.append((wk.astype(np.float32), float(sk), name))
        cand.sort(key=lambda t: -t[1]); cl = []
        for wk, sk, name in cand:
            for cluster in cl:
                if mdist(wk, cluster[0][0]) <= a.dedup: cluster.append((wk, sk, name)); break
            else: cl.append([(wk, sk, name)])
        out = []
        for cluster in cl:
            if len(cluster) == 1 or a.merge == "score": rep = cluster[0]
            elif a.merge == "longest": rep = max(cluster, key=lambda t: mlen(t[0]))
            else:
                ft = [t for t in cluster if t[2] == "ft"]
                rep = max(ft, key=lambda t: t[1]) if ft else cluster[0]
                if a.merge == "ftsafe" and ft and not 0.85 * L_med <= mlen(rep[0]) <= 1.15 * L_med:
                    others = [t for t in cluster if t[2] != "ft"]
                    if others:
                        alt = max(others, key=lambda t: t[1])
                        if abs(mlen(alt[0]) - L_med) < abs(mlen(rep[0]) - L_med): rep = alt
            out.append(dict(w=rep[0], s=rep[1], m=rep[2]))
        frames[c] = out
    if (b0 // a.batch) % 50 == 0:
        print(f"  {b0 + len(cs)}/{T}  {(time.time() - t0) / (b0 + len(cs)) * 1000:.0f} ms/frame", flush=True)
secs = time.time() - t0
n = sum(len(v) for v in frames.values())
print(f"DTC done: {T} frames, {n} midlines ({n / max(T, 1):.2f}/frame) in {secs:.0f} s = {secs / max(T, 1) * 1000:.0f} ms/frame", flush=True)
pickle.dump(dict(frames=frames, scale=scale, skip=skip, target_L=a.target_L, L_med=L_med, model_dir=str(MP), ft=a.ft,
                 secs=secs, thr=a.thr, ensemble=len(models) > 1), open(a.out, "wb"))
