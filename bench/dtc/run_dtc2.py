import sys, time, pickle, json, argparse; sys.path.insert(0, "/work/tierpsy_tracker_2.0")
import cv2, numpy as np, jax, jax.numpy as jnp
from pathlib import Path
from deeptangle.forward import build_model
from deeptangle import utils
from deeptangle.predict import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("clip"); ap.add_argument("--scale", type=float, required=True, help="downscale factor of clip vs original, e.g. 6")
ap.add_argument("--skip", type=int, default=4); ap.add_argument("--step", type=int, default=1)
ap.add_argument("--block", type=int, default=15); ap.add_argument("--C", type=int, default=1)
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--cutoff", type=int, default=96)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--tag", default="")
ap.add_argument("--bgd", type=float, default=0, help="mean-background subtraction threshold (0=off)")
ap.add_argument("--resize", type=float, default=1.0, help="extra downscale applied in-script (e.g. 0.5)")
ap.add_argument("--nothresh", action="store_true", help="feed inverted raw frames instead of thresholded")
ap.add_argument("--params", default="", help="fine-tuned checkpoint ckpt_*.pkl from finetune.py")
ap.add_argument("--params2", default="", help="second model (e.g. the shipped one = 'base') for an ensemble: per-model NMS, then geometric union")
ap.add_argument("--dedup", type=float, default=60.0, help="ensemble: midlines closer than this mean distance (orig px, min over flip) are duplicates")
ap.add_argument("--lmed", type=float, default=0.0, help="median worm length in original px (enables --merge ftsafe)")
ap.add_argument("--merge", default="ftsafe", choices=["ftsafe", "ftfirst", "longest", "score"],
                help="ensemble: which midline of a duplicate cluster to keep. ftsafe = the --params (fine-tuned) "
                     "model's one, falling back to base when the ft length is outside 0.85-1.15 x --lmed and base "
                     "is closer (without --lmed behaves as ftfirst); ftfirst = always the ft one, base then only "
                     "adds recall; longest = the longest; score = highest score (old behaviour, inherits the base "
                     "model's ~7%% too short midlines)")
a = ap.parse_args()

MP = Path("/work/model/DT_C_data_model/model/parameters_Final_mixed_train_data_wt_hand_ann_pca_92_epochs_250_2024-04-19_96_cutoff")
A = jnp.load(MP/"eigenworms_transform.npy"); forward_fn = build_model(A, 8, 8, 11)
tree = pickle.load(open(MP/"tree.pkl","rb")); leaves, treedef = jax.tree_util.tree_flatten(tree)
with open(MP/"arrays.npy","rb") as f: flat = [jnp.load(f) for _ in leaves]
params, st, _ = utils.single_from_sharded(jax.tree_util.tree_unflatten(treedef, flat))
if a.params:
    ck = pickle.load(open(a.params, "rb")); params, st = jax.tree_util.tree_map(jnp.asarray, ck["params"]), jax.tree_util.tree_map(jnp.asarray, ck["state"]); print("params from", a.params)
fwd = jax.jit(lambda x: forward_fn.apply(params, st, x, is_training=False)[0])
fwd2 = None
if a.params2:
    if a.params2 == "base":
        params2, st2, _ = utils.single_from_sharded(jax.tree_util.tree_unflatten(treedef, flat))
    else:
        ck2 = pickle.load(open(a.params2, "rb")); params2, st2 = jax.tree_util.tree_map(jnp.asarray, ck2["params"]), jax.tree_util.tree_map(jnp.asarray, ck2["state"])
    fwd2 = jax.jit(lambda x: forward_fn.apply(params2, st2, x, is_training=False)[0]); print("ensemble with", a.params2)
def mdist(u, v):
    return min(np.linalg.norm(u - v, axis=1).mean(), np.linalg.norm(u - v[::-1], axis=1).mean())

def mlen(w):
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())

def merge_cands(cand, dedup, rule, lmed=0.0):
    """cand: [(midline, score, model)] from all models; greedy score-descending clustering, then one per cluster."""
    cl = []
    for w, sc, m in sorted(cand, key=lambda t: -t[1]):
        for c in cl:
            if mdist(w, c[0][0]) <= dedup: c.append((w, sc, m)); break
        else: cl.append([(w, sc, m)])
    out = []
    for c in cl:
        if len(c) == 1 or rule == "score": out.append(c[0][:2]); continue
        if rule == "longest": out.append(max(c, key=lambda t: mlen(t[0]))[:2]); continue
        ft = [t for t in c if t[2] == "ft"]
        rep = max(ft, key=lambda t: t[1]) if ft else c[0]
        if rule == "ftsafe" and ft and lmed > 0 and not 0.85 * lmed <= mlen(rep[0]) <= 1.15 * lmed:
            others = [t for t in c if t[2] != "ft"]
            if others:
                alt = max(others, key=lambda t: t[1])
                if abs(mlen(alt[0]) - lmed) < abs(mlen(rep[0]) - lmed): rep = alt
        out.append(rep[:2])
    return out

cap = cv2.VideoCapture(a.clip); frames = []
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
if a.resize != 1.0:
    frames = [cv2.resize(f, None, fx=a.resize, fy=a.resize, interpolation=cv2.INTER_AREA) for f in frames]; a.scale /= a.resize
frames = np.array(frames); T, H, W = frames.shape
is_light = frames[0].mean() > 127
print(f"{a.clip}: {T} frames {W}x{H} light_bg={is_light}", flush=True)
inv = (255 - frames) if is_light else frames                     # bright worms on dark
if a.nothresh:
    pre = inv.astype(np.float32)
else:
    # same as tierpsynn._adaptive_thresholding: threshold on light-bg image, keep dark pixels
    mask = np.array([cv2.adaptiveThreshold(255 - im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, a.block, a.C) == 0 for im in inv])
    pre = mask * inv.astype(np.float32)
    if a.bgd > 0:
        bg = inv.astype(np.float32).mean(0); pre *= (inv.astype(np.float32) - bg) > a.bgd
pb, pr = (-H) % 16, (-W) % 16
pre = np.pad(pre, ((0,0),(0,pb),(0,pr))) / 255.0
name = Path(a.clip).stem + a.tag
cv2.imwrite(f"/work/out/{name}_pre.png", (pre[T//2]*255).astype(np.uint8))

span = a.skip * 10
centers = list(range(5*a.skip, T - 5*a.skip, a.step))
results = {}; t0 = time.time(); nfwd = 0
for b0 in range(0, len(centers), a.batch):
    cs = centers[b0:b0+a.batch]
    x = jnp.asarray(np.stack([pre[c-5*a.skip:c+5*a.skip+1:a.skip] for c in cs]), jnp.float32)
    t1 = time.time(); preds = fwd(x); jax.block_until_ready(preds); tf = time.time()-t1; nfwd += len(cs)
    w_all, s_all, p_all = map(np.asarray, preds)
    if fwd2 is not None: w2_all, s2_all, p2_all = map(np.asarray, fwd2(x))
    for i, c in enumerate(cs):
        pr_i = (w_all[i], s_all[i], p_all[i])
        keep = non_max_suppression(pr_i, a.thr, 0.5, a.cutoff)
        W1 = w_all[i][keep][:, 1] * a.scale; S1 = s_all[i][keep]
        if fwd2 is not None:
            keep2 = non_max_suppression((w2_all[i], s2_all[i], p2_all[i]), a.thr, 0.5, a.cutoff)
            W2 = w2_all[i][keep2][:, 1] * a.scale; S2 = s2_all[i][keep2]
            cand = [(w, s, "ft") for w, s in zip(W1, S1)] + [(w, s, "base") for w, s in zip(W2, S2)]
            out = merge_cands(cand, a.dedup, a.merge, a.lmed)
            W1 = np.array([o[0] for o in out]).reshape(-1, 49, 2); S1 = np.array([o[1] for o in out])
        results[c] = dict(w=W1, s=S1, nraw3=int((s_all[i] > 0.3).sum()))
    if b0 % (a.batch*10) == 0: print(f"  {b0+len(cs)}/{len(centers)} fwd {tf/len(cs)*1000:.0f} ms/frame", flush=True)
dt = time.time()-t0
print(f"done {len(centers)} frames in {dt:.1f}s = {dt/len(centers)*1000:.0f} ms/frame (incl. NMS)", flush=True)
pickle.dump(dict(results=results, scale=a.scale, T=T, H=H, W=W, args=vars(a), secs=dt), open(f"/work/out/{name}.pkl","wb"))
