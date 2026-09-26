"""finetune2.py = finetune.py with REHEARSAL: each batch element is drawn from one of several sources with given weights.

  python finetune2.py OUT --src ours:/work/dtc_train.hdf5:0.5 --src authors:/work/authors/X.hdf5:0.3 --src ours:/work/dtc_pseudo.hdf5:0.2
                     [--steps 6000] [--batch 8] [--lr 1e-4] [--holdout IMG_8373] [--pair 0.5] [--init ckpt.pkl]
                     [--eval-every 500] [--save-every 500] [--seed 0] [--crop 128x224]

Source kinds: `ours` = our hdf5 (x (S,11,H,W) uint8, y (S,NL,3,49,2) crop px padded -1, meta_video; the holdout video is
excluded and, for the first `ours` source, forms the validation set); `authors` = the DeepTangleCrawl training file
(x_train/<id> (11,512,512) [0,1], y_train/<id> (N,3,49,2)): a random HxW crop around a random worm is taken on the fly,
labels shifted, labels fully outside dropped, cut worms handled by the partial-label logic of the loss.
Batch label slots NL = max over sources (authors' crops may hold more worms than ours). Augmentation + pairing as before.
"""
import sys, os, time, pickle, argparse, threading, queue, json; sys.path.insert(0, "/work/tierpsy_tracker_2.0")
import numpy as np, h5py, jax, jax.numpy as jnp, optax
from jax import grad, jit, vmap
from jax.tree_util import tree_map, tree_reduce
from collections import namedtuple
from pathlib import Path
from deeptangle.forward import build_model
from deeptangle import utils

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--src", action="append", required=True, help="kind:path:weight")
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--holdout", default="IMG_8373")
ap.add_argument("--pair", type=float, default=0.5); ap.add_argument("--init", default="")
ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nval", type=int, default=64)
ap.add_argument("--crop", default="128x224"); ap.add_argument("--nl", type=int, default=10)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
MP = Path("/work/model/DT_C_data_model/model/parameters_Final_mixed_train_data_wt_hand_ann_pca_92_epochs_250_2024-04-19_96_cutoff")
sigma, cutoff = 15.0, 96.0
MIN_VIS = float(os.environ.get("MIN_VIS", "0.35"))   # v3 loss: a cut worm needs this fraction of its points inside the crop to be supervised
wl = (1.0, 1e2, 1e5)
Losses = namedtuple("Losses", ["w", "s", "p"])
H, W = map(int, a.crop.split("x")); NL = a.nl; NF = 11
rng = np.random.default_rng(a.seed)

# ---------------------------------------------------------------- data
def pad_labels(y):
    out = np.full((NL, 3, 49, 2), -1.0, np.float32); n = min(NL, len(y)); out[:n] = y[:n]; return out

class Ours:
    def __init__(self, path, first):
        self.h5 = h5py.File(path, "r"); self.X, self.Y = self.h5["x"], self.h5["y"]; vids = self.h5["meta_video"][:].astype(str)
        S = self.X.shape[0]; assert self.X.shape[2:] == (H, W), self.X.shape
        hold = np.array([v.startswith(a.holdout) for v in vids]) if a.holdout else np.zeros(S, bool)
        self.idx = np.nonzero(~hold)[0]; self.val = np.nonzero(hold)[0] if first else np.array([], int)
        print(f"{path}: {S} samples, train {len(self.idx)}, val {len(self.val)}", flush=True)
    def load(self, i): return self.X[i].astype(np.float32) / 255.0, pad_labels(self.Y[i])
    def sample(self): return self.load(self.idx[rng.integers(len(self.idx))])

class Authors:
    def __init__(self, path):
        self.h5 = h5py.File(path, "r"); self.ids = list(self.h5["x_train"].keys()); print(f"{path}: {len(self.ids)} clips", flush=True)
    def sample(self):
        for _ in range(10):
            k = self.ids[rng.integers(len(self.ids))]; y = np.asarray(self.h5["y_train"][k], np.float32)   # (N,3,49,2) full-frame px
            if y.ndim == 4 and len(y) > 0: break
        x = np.asarray(self.h5["x_train"][k], np.float32); F, hs, ws = x.shape
        c = y[rng.integers(len(y)), 1].mean(0)                                                               # a random worm's centre
        x0 = int(np.clip(round(c[0] - W / 2 + rng.uniform(-0.4, 0.4) * W), 0, ws - W)); y0 = int(np.clip(round(c[1] - H / 2 + rng.uniform(-0.4, 0.4) * H), 0, hs - H))
        crop = np.ascontiguousarray(x[:, y0:y0 + H, x0:x0 + W]); yy = y - [x0, y0]
        inside_any = ((yy[..., 0] >= 0) & (yy[..., 0] < W) & (yy[..., 1] >= 0) & (yy[..., 1] < H)).any(axis=(1, 2))
        yy = yy[inside_any]; cc = np.array([W / 2, H / 2])
        yy = yy[np.argsort(np.linalg.norm(yy[:, 1].mean(1) - cc, axis=1))]
        return crop, pad_labels(yy)

sources, weights = [], []; first = True
for spec in a.src:
    kind, path, w = spec.rsplit(":", 2)
    sources.append(Ours(path, first) if kind == "ours" else Authors(path)); weights.append(float(w))
    if kind == "ours": first = False
weights = np.array(weights) / np.sum(weights); print("source weights", dict(zip(a.src, np.round(weights, 3))), flush=True)
val_src = next(s for s in sources if isinstance(s, Ours) and len(s.val))
val_idx = np.sort(rng.choice(val_src.val, min(a.nval, len(val_src.val)), replace=False))
val_x = np.stack([val_src.load(i)[0] for i in val_idx]); val_y = np.stack([val_src.load(i)[1] for i in val_idx])
print(f"val {len(val_idx)} samples (holdout={a.holdout}); crop {H}x{W}, NL {NL}", flush=True)

def flip_labels(y, hflip, vflip):
    y = y.copy(); valid = y[..., 0] >= 0
    if hflip: y[..., 0] = np.where(valid, W - 1 - y[..., 0], -1)
    if vflip: y[..., 1] = np.where(valid, H - 1 - y[..., 1], -1)
    return y

def augment(x, y):
    if rng.random() < 0.5: x = x[:, :, ::-1]; y = flip_labels(y, True, False)
    if rng.random() < 0.5: x = x[:, ::-1, :]; y = flip_labels(y, False, True)
    if rng.random() < 0.5: x = x[::-1]; y = y[:, ::-1]                       # time reversal
    x = np.clip(x * rng.uniform(0.7, 1.3), 0, 1)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)

def shift(x, y, dx, dy):
    out = np.zeros_like(x)
    sy0, sy1 = max(0, dy), min(H, H + dy); sx0, sx1 = max(0, dx), min(W, W + dx)
    out[:, sy0:sy1, sx0:sx1] = x[:, sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
    y = y.copy(); valid = y[..., 0] >= 0
    y[..., 0] = np.where(valid, y[..., 0] + dx, -1); y[..., 1] = np.where(valid, y[..., 1] + dy, -1)
    return out, y

def n_valid(y): return int((y[:, 0, 0, 0] >= 0).sum())

def merge_labels(ya, yb):
    labs = [l for l in list(ya) + list(yb) if l[0, 0, 0] >= 0]
    def inside(l): return bool(((l >= 0) & (l < [W, H])).all())
    labs.sort(key=lambda l: not inside(l))
    y = np.full((NL, 3, 49, 2), -1.0, np.float32)
    for i, l in enumerate(labs[:NL]): y[i] = l
    return y

def draw(): return sources[rng.choice(len(sources), p=weights)].sample()

def make_sample():
    x, y = augment(*draw())
    if rng.random() < a.pair:
        xb, yb = augment(*draw())
        if n_valid(y) and n_valid(yb):
            ca = y[0, 1].mean(0); cb = yb[0, 1].mean(0)
            L = np.hypot(*np.diff(y[0, 1], axis=0).T).sum()
            target = ca + rng.uniform(-0.5, 0.5, 2) * L * np.array([1, 0.6])
            dx, dy = np.round(target - cb).astype(int); dx = int(np.clip(dx, -W // 2, W // 2)); dy = int(np.clip(dy, -H // 2, H // 2))
            xb, yb = shift(xb, yb, dx, dy)
            x = np.maximum(x, xb); y = merge_labels(y, yb)
    return x, y

def batch_iter():
    while True:
        xs, ys = zip(*[make_sample() for _ in range(a.batch)])
        yield np.stack(xs), np.stack(ys)
q = queue.Queue(maxsize=6)
def feeder():
    for b in batch_iter(): q.put(b)
threading.Thread(target=feeder, daemon=True).start()

# ---------------------------------------------------------------- model
A = jnp.load(MP / "eigenworms_transform.npy"); forward = build_model(A, 8, 8, 11)
if a.init:
    ck = pickle.load(open(a.init, "rb")); params, st = tree_map(jnp.asarray, ck["params"]), tree_map(jnp.asarray, ck["state"])
    print("init from", a.init)
else:
    tree = pickle.load(open(MP / "tree.pkl", "rb")); leaves, treedef = jax.tree_util.tree_flatten(tree)
    with open(MP / "arrays.npy", "rb") as f: flat = [jnp.load(f) for _ in leaves]
    params, st, _ = utils.single_from_sharded(jax.tree_util.tree_unflatten(treedef, flat))
opt = optax.adamw(a.lr); opt_state = opt.init(params)

def _iw(n):
    w = 1 / (jnp.abs(jnp.arange(-n // 2 + 1, n // 2 + 1)) + 1); return w / w.sum()
def multi_loss_fn(Y_pred, Y_label):
    """Authors' loss with POSITIVE supervision for worms cut by the crop edge (v3, 2026-09-20):
    distances between a prediction and a label are averaged over the label points that lie INSIDE the crop
    (per frame), so a partially visible worm still gets a point loss and a positive score target on its
    visible part; labels with < MIN_VIS of their points visible are ignored (no Loss_X, no Loss_S for the
    predictions that target them); padded labels (-1) never attract predictions."""
    X_pred, S_pred, P_pred = Y_pred
    lim = jnp.array([W, H], jnp.float32)
    vis = jnp.all((Y_label >= 0) & (Y_label < lim), axis=-1)                # (B, NL, 3, 49) point visible
    valid = Y_label[:, :, 0, 0, 0] >= 0                                      # padded labels are all -1
    fvis = vis.mean(axis=(-1, -2))                                           # visible fraction per label
    fw = _iw(X_pred.shape[2])                                                # frame weights (0.25, 0.5, 0.25)
    wv = vis.astype(jnp.float32) * fw[None, None, :, None]                   # (B, NL, 3, 49)
    wvf = jnp.flip(wv, axis=-1)
    @vmap
    def dm(p, q_): return jnp.sum((p[None] - q_[:, None]) ** 2, axis=-1)    # (NL, Np, 3, 49)
    d = dm(X_pred, Y_label); df = dm(X_pred, jnp.flip(Y_label, axis=-2))
    num = jnp.sum(d * wv[:, :, None], axis=(-1, -2)); numf = jnp.sum(df * wvf[:, :, None], axis=(-1, -2))
    den = jnp.sum(wv, axis=(-1, -2))[:, :, None] + 1e-6                     # same for flipped
    distances = jnp.minimum(num, numf) / den                                 # (B, NL, Np) masked mean sq. dist
    supervised = valid & (fvis >= MIN_VIS)                                   # full + sufficiently visible cut worms
    ignored = valid & ~supervised
    distances = jnp.where(valid[:, :, None], distances, 1e9)                 # padded labels never attract predictions
    sup_count = jnp.sum(supervised) + 1e-6
    Loss_X = jnp.sum(jnp.min(distances * supervised[:, :, None], axis=2)) / sup_count
    distances = jax.lax.stop_gradient(distances); Xs = jax.lax.stop_gradient(X_pred)
    T = jnp.argmin(distances, axis=1)                                        # (B, Np): nearest label of each prediction
    scores = jnp.exp(-jnp.min(distances, axis=1) / sigma)
    keep = (~jnp.take_along_axis(ignored, T, axis=1)).astype(jnp.float32)    # predictions aimed at an ignored label: no score loss
    Loss_S = jnp.sum(keep * (scores - S_pred) ** 2) / (keep.sum() + 1e-6)
    same_T = T[:, None, :] == T[:, :, None]
    dls = dm(P_pred, P_pred); K = Xs.shape[3]
    Xcm = Xs[:, :, X_pred.shape[2] // 2, K // 2, :]
    visible = dm(Xcm, Xcm) < cutoff ** 2; factor = visible / visible.sum(axis=2)[:, :, None]
    safe_log = lambda v: jnp.log(jnp.where(v > 0.0, v, 1.0))
    Loss_P = factor * jnp.where(same_T, dls, -safe_log(1 - jnp.exp(-dls)))
    sm = scores[:, :, None] * scores[:, None, :]
    Loss_P = jnp.sum(sm * Loss_P) / sm.sum()
    return Losses(Loss_X, Loss_S, Loss_P)
def loss_fn(params, state, x, y, training):
    preds, state = forward.apply(params, state, x, is_training=training)
    L = multi_loss_fn(preds, y); L = tree_map(jnp.multiply, Losses(*wl), L)
    return tree_reduce(jnp.add, L), (state, L)
@jit
def train_step(params, state, opt_state, x, y):
    grads, (state, L) = grad(loss_fn, has_aux=True)(params, state, x, y, True)
    upd, opt_state = opt.update(grads, opt_state, params=params)
    return optax.apply_updates(params, upd), state, opt_state, L
@jit
def eval_loss(params, state, x, y):
    return loss_fn(params, state, x, y, False)[1][1]

def evaluate(params, state):
    Ls = [eval_loss(params, state, jnp.asarray(val_x[i:i + 8]), jnp.asarray(val_y[i:i + 8])) for i in range(0, len(val_x), 8)]
    return Losses(*[float(np.mean([float(getattr(l, k)) for l in Ls])) for k in ("w", "s", "p")])

def save(step, params, state):
    ck = dict(params=tree_map(np.asarray, params), state=tree_map(np.asarray, state), step=step, args=vars(a))
    pickle.dump(ck, open(OUT / f"ckpt_{step}.pkl", "wb"))

log = open(OUT / "log.jsonl", "a")
v0 = evaluate(params, st); print(f"step 0 val w={v0.w:.3f} s={v0.s:.3f} p={v0.p:.3f}", flush=True)
log.write(json.dumps(dict(step=0, val=v0._asdict())) + "\n"); log.flush()
t0 = time.time(); acc = []
for step in range(1, a.steps + 1):
    x, y = q.get()
    params, st, opt_state, L = train_step(params, st, opt_state, jnp.asarray(x), jnp.asarray(y))
    acc.append([float(L.w), float(L.s), float(L.p)])
    if step % 50 == 0:
        m = np.mean(acc, 0); acc = []
        print(f"step {step} train w={m[0]:.3f} s={m[1]:.3f} p={m[2]:.3f}  {(time.time() - t0) / step:.2f} s/step", flush=True)
        log.write(json.dumps(dict(step=step, train=dict(w=m[0], s=m[1], p=m[2]))) + "\n"); log.flush()
    if step % a.eval_every == 0:
        v = evaluate(params, st); print(f"step {step} VAL w={v.w:.3f} s={v.s:.3f} p={v.p:.3f}", flush=True)
        log.write(json.dumps(dict(step=step, val=v._asdict())) + "\n"); log.flush()
    if step % a.save_every == 0 or step == a.steps: save(step, params, st)
print("done", flush=True)
