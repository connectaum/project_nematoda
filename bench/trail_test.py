import sys, glob, numpy as np, pandas as pd
from common import *; from methods import longest_path, crop
from skimage.morphology import skeletonize
from trail import longest_trail
d = sys.argv[1]; v = d.split("/")[-1].split("_")[0] + "_" + d.split("/")[-1].split("_")[1]; ctx = VideoCtx(v)
z = np.load(f"{d}/masks.npz"); M = np.unpackbits(z["masks"], axis=-1)[..., :ctx.W].astype(bool); T, K = M.shape[:2]
res = {"skeleton": [], "trail": []}
for t in range(T):
    for k in range(K):
        m = M[t, k]
        if m.sum() < 0.15*ctx.A_med: res["skeleton"].append(0); res["trail"].append(0); continue
        sub, off = crop(m)
        p = longest_path(skeletonize(sub)); res["skeleton"].append(arclen(p)/ctx.L_med if p is not None else 0)
        q = longest_trail(sub, L_ref=ctx.L_med, spur=0); res["trail"].append(arclen(q)/ctx.L_med if q is not None else 0)
for k, vals in res.items():
    a = np.array(vals).reshape(T, K); ok = (a >= 0.6) & (a <= 1.3)
    print(f"{k:9s} plausible per obj {ok.mean(0).round(2)}  all-objs {ok.all(1).mean():.2f}  median L/Lmed {np.median(a[a>0]):.2f}")
