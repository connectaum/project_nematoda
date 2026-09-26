import glob, os, numpy as np, pandas as pd
from common import *; from methods import longest_path, crop
from skimage.morphology import skeletonize
from trail import longest_trail
rows = []
for d in sorted(glob.glob("results/sam2ep/*")):
    b = os.path.basename(d); v = "_".join(b.split("_")[:2]); ctx = VideoCtx(v)
    z = np.load(f"{d}/masks.npz"); M = np.unpackbits(z["masks"], axis=-1)[..., :ctx.W].astype(bool); T, K = M.shape[:2]
    sk = np.zeros((T, K)); tr = np.zeros((T, K)); area = M.reshape(T, K, -1).sum(-1) / ctx.A_med
    for t in range(T):
        for k in range(K):
            m = M[t, k]
            if m.sum() < 0.15*ctx.A_med: continue
            sub, off = crop(m); p = longest_path(skeletonize(sub)); sk[t, k] = arclen(p)/ctx.L_med if p is not None else 0
            q = longest_trail(sub, L_ref=ctx.L_med, spur=0); tr[t, k] = arclen(q)/ctx.L_med if q is not None else 0
    ok = lambda a: ((a >= 0.6) & (a <= 1.3))
    rows.append(dict(run=b, frames=T, objs=K, area_per_obj=str(area.mean(0).round(2)), all_plaus_skeleton=ok(sk).all(1).mean().round(2),
                     all_plaus_trail=ok(tr).all(1).mean().round(2), obj_lost=(area < 0.15).mean().round(2)))
df = pd.DataFrame(rows); df.to_csv("results/sam2_summary.csv", index=False); print(df.to_string(index=False))
