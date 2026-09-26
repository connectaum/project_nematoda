"""Fine-tune the shipped DeepTangleCrawl checkpoint on our hdf5 (dtc_make_dataset.py) — CPU, JAX.

  python finetune.py data.hdf5 outdir [--steps 5000] [--batch 8] [--lr 1e-4] [--holdout IMG_8373] [--pair 0.5]
                     [--init ckpt.pkl] [--eval-every 250] [--save-every 500] [--seed 0]

Loss and model are the authors' (train.py) with `inside` computed for a rectangular crop. Batch augmentation:
h/v flips, time reversal, intensity jitter, and with prob --pair a second sample is max-blended on top with its
worm-0 shifted next to the first one's worm-0 (synthetic contact/overlap, the authors' recipe). Labels padded
with -1 (→ not inside → no Loss_X, but they still shape the score targets). Checkpoints: <outdir>/ckpt_<step>.pkl
= dict(params, state) as numpy trees, loadable by run_dtc.py --params.
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
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--holdout", default="IMG_8373")
ap.add_argument("--pair", type=float, default=0.5); ap.add_argument("--init", default="")
ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nval", type=int, default=64)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
MP = Path("/work/model/DT_C_data_model/model/parameters_Final_mixed_train_data_wt_hand_ann_pca_92_epochs_250_2024-04-19_96_cutoff")
sigma, cutoff = 15.0, 96.0; wl = (1.0, 1e2, 1e5)
Losses = namedtuple("Losses", ["w", "s", "p"])

# ---------------------------------------------------------------- data
h5 = h5py.File(a.data, "r"); X, Y = h5["x"], h5["y"]; vids = h5["meta_video"][:].astype(str)
S, NF, H, W = X.shape; NL = Y.shape[1]
hold = np.array([v.startswith(a.holdout) for v in vids]) if a.holdout else np.zeros(S, bool)
train_idx = np.nonzero(~hold)[0]; val_idx = np.nonzero(hold)[0]
rng = np.random.default_rng(a.seed)
if len(val_idx) == 0: val_idx = rng.choice(train_idx, min(a.nval, len(train_idx)), replace=False); train_idx = np.setdiff1d(train_idx, val_idx)
val_idx = np.sort(rng.choice(val_idx, min(a.nval, len(val_idx)), replace=False))
print(f"{S} samples: train {len(train_idx)}, val {len(val_idx)} (holdout={a.holdout}); crop {H}x{W}", flush=True)

def load(i):
    return X[i].astype(np.float32) / 255.0, Y[i].copy()

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
    """translate frames by integer (dx, dy) with zero fill; labels follow (invalid stay -1)"""
    out = np.zeros_like(x)
    sy0, sy1 = max(0, dy), min(H, H + dy); sx0, sx1 = max(0, dx), min(W, W + dx)
    out[:, sy0:sy1, sx0:sx1] = x[:, sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
    y = y.copy(); valid = y[..., 0] >= 0
    y[..., 0] = np.where(valid, y[..., 0] + dx, -1); y[..., 1] = np.where(valid, y[..., 1] + dy, -1)
    return out, y

def n_valid(y): return int((y[:, 0, 0, 0] >= 0).sum())

def merge_labels(ya, yb):
    """concat valid labels, prefer those fully inside the crop, cap at NL"""
    labs = [l for l in list(ya) + list(yb) if l[0, 0, 0] >= 0]
    def inside(l): return bool(((l >= 0) & (l < [W, H])).all())
    labs.sort(key=lambda l: not inside(l))
    y = np.full((NL, 3, 49, 2), -1.0, np.float32)
    for i, l in enumerate(labs[:NL]): y[i] = l
    return y

def make_sample():
    i = train_idx[rng.integers(len(train_idx))]; x, y = augment(*load(i))
    if rng.random() < a.pair:
        j = train_idx[rng.integers(len(train_idx))]; xb, yb = augment(*load(j))
        if n_valid(y) and n_valid(yb):
            ca = y[0, 1].mean(0); cb = yb[0, 1].mean(0)               # worm-0 centres on the middle frame
            L = np.hypot(*np.diff(y[0, 1], axis=0).T).sum()
            target = ca + rng.uniform(-0.5, 0.5, 2) * L * np.array([1, 0.6])
            dx, dy = np.round(target - cb).astype(int)
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
val_x = np.stack([load(i)[0] for i in val_idx]); val_y = np.stack([load(i)[1] for i in val_idx])

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
    X_pred, S_pred, P_pred = Y_pred
    lim = jnp.array([W, H], jnp.float32)
    inside = jnp.all((Y_label >= 0) & (Y_label < lim), axis=(-1, -2, -3))
    valid = Y_label[:, :, 0, 0, 0] >= 0                       # padded labels are all -1
    partial = valid & ~inside                                  # worm cut by the crop edge: no supervision at all
    @vmap
    def dm(p, q_): return jnp.sum((p[None] - q_[:, None]) ** 2, axis=-1)
    d = dm(X_pred, Y_label).mean(-1); df = dm(X_pred, jnp.flip(Y_label, axis=-2)).mean(-1)
    distances = jnp.average(jnp.minimum(d, df), axis=-1, weights=_iw(X_pred.shape[2]))
    distances = jnp.where(valid[:, :, None], distances, 1e9)  # padded labels never attract predictions
    inside_count = jnp.sum(inside) + 1e-6
    Loss_X = jnp.sum(jnp.min(distances * inside[:, :, None], axis=2)) / inside_count
    distances = jax.lax.stop_gradient(distances); Xs = jax.lax.stop_gradient(X_pred)
    T = jnp.argmin(distances, axis=1)                          # (B, Np): nearest label of each prediction
    scores = jnp.exp(-jnp.min(distances, axis=1) / sigma)
    ignore = jnp.take_along_axis(partial, T, axis=1)           # predictions aimed at a cut worm: no score loss
    keep = (~ignore).astype(jnp.float32)
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
