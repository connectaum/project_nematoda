"""Morphometry of every fully-visible worm instance in the clean bench frames -> per-track features,
clustering, figures and a crop gallery for expert review.   python morphometry.py
Outputs: bench/morpho/instances.csv, tracks.csv, fig_*.png, gallery_*.jpg"""
import os, json, csv, numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from common import *
from methods import longest_path

OUT = os.path.join(ROOT, "bench", "morpho"); os.makedirs(OUT, exist_ok=True)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948", "#7a7a7a", "#8a5a2b", "#00a0c0", "#b0006d"]
SCALE_TO_720 = {"ga": 720/1080}          # ga series is 1080p, IMG is 720p — assumed same optics (see report)

rows = []; crops = []
for v in videos():
    ctx = VideoCtx(v); sc = SCALE_TO_720["ga"] if v.startswith("ga") else 1.0
    for r in load_index(v):
        if r["kind"] != "clean": continue
        f = int(r["frame"]); img = load_frame(v, f); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(gray, (5, 5), 0); bg = float(np.median(g))
        m = ctx.segment(gray); comps = components(m)
        for k, p in ctx.gt_midlines(f):
            xi = np.clip(p[:, 0].round().astype(int), 0, ctx.W-1); yi = np.clip(p[:, 1].round().astype(int), 0, ctx.H-1)
            best = max(comps, key=lambda c: c["mask"][yi, xi].sum(), default=None)
            if best is None or best["mask"][yi, xi].mean() < 0.9 or best["area"] > 1.5*ctx.A_big: continue
            y0, x0, y1, x1 = best["bbox"]
            if x0 <= 3 or y0 <= 3 or x1 >= ctx.W-3 or y1 >= ctx.H-3: continue
            sub = best["mask"][y0:y1, x0:x1]; sk = skeletonize(sub); lp = longest_path(sk)
            if lp is None or (sk.sum() - len(lp)) > 0.12*len(lp): continue
            L = arclen(p); L_ref = ctx.tracks[k]["body_length_px"] if k < len(ctx.tracks) else ctx.L_med
            if abs(L/L_ref - 1) > 0.2: continue
            dist = cv2.distanceTransform(best["mask"].astype(np.uint8), cv2.DIST_L2, 5)
            w = 2*dist[yi, xi]                                   # width profile head(0) -> tail(48)
            wm = np.median(w[12:37])
            dark = float(1 - g[best["mask"]].mean()/bg)          # optical density proxy (0 = transparent)
            dark_core = float(1 - np.percentile(g[best["mask"]], 10)/bg)
            # curvature-independent shape: fraction of body that is "thin" (< 0.5 wm) at each end
            def taper(seg): return float(np.mean(seg)/wm)
            rows.append(dict(video=v, track=k, frame=f, L_px=L, L_720=L*sc, W_px=wm, W_720=wm*sc, W_over_L=wm/L,
                             area_over_L2=best["area"]/L**2, taper_head=taper(w[1:6]), taper_tail=taper(w[-6:-1]),
                             taper_head_q=taper(w[6:13]), taper_tail_q=taper(w[-13:-6]), tail_len_thin=float(np.sum(w[24:] < 0.5*wm)/25),
                             head_len_thin=float(np.sum(w[:25] < 0.5*wm)/25), dark=dark, dark_core=dark_core,
                             profile=";".join(f"{x:.2f}" for x in w/wm)))
            pad = 12; crops.append(dict(video=v, track=k, frame=f, img=img[max(y0-pad,0):y1+pad, max(x0-pad,0):x1+pad].copy(), sc=sc))
inst = pd.DataFrame(rows); inst.to_csv(f"{OUT}/instances.csv", index=False)
print("instances", len(inst), "by video:", inst.groupby("video").size().to_dict())

# per-track aggregation
feat = ["L_720", "W_720", "W_over_L", "area_over_L2", "taper_head", "taper_tail", "taper_head_q", "taper_tail_q", "tail_len_thin", "head_len_thin", "dark", "dark_core"]
tr = inst.groupby(["video", "track"]).agg(n=("frame", "size"), **{f: (f, "median") for f in feat}).reset_index()
prof = inst.groupby(["video", "track"])["profile"].apply(lambda s: np.median(np.array([[float(x) for x in p.split(";")] for p in s]), axis=0))
tr = tr[tr.n >= 3].reset_index(drop=True)
print("tracks with >=3 instances:", len(tr))

# clustering at the INSTANCE level (tracks are too few in the sampled clean frames); track-level = majority vote
FEATS = ["L_720", "W_over_L", "taper_head_q", "taper_tail_q", "dark"]
Xs = StandardScaler().fit_transform(inst[FEATS].values); Z = linkage(Xs, "ward")
best = None
for k in (2, 3, 4):
    lab = fcluster(Z, k, "maxclust"); s = silhouette_score(Xs, lab); sizes = np.bincount(lab)[1:]
    print(f"k={k} silhouette={s:.2f} sizes={sizes.tolist()}")
    if sizes.min() >= 0.05*len(inst) and (best is None or s > best[1]): best = (k, s, lab)
k, sil, lab = best; inst["cluster"] = lab
# second pass without the single giant ga10 worm (it dominates the first split): clusters 2.. within the rest
sub = inst[inst.video != "ga10Trim"]; Xs2 = StandardScaler().fit_transform(sub[FEATS].values); Z2 = linkage(Xs2, "ward"); best2 = None
for k2 in (2, 3, 4):
    lab2 = fcluster(Z2, k2, "maxclust"); s2 = silhouette_score(Xs2, lab2); sizes = np.bincount(lab2)[1:]
    print(f"[no ga10] k={k2} silhouette={s2:.2f} sizes={sizes.tolist()}")
    if sizes.min() >= 0.05*len(sub) and (best2 is None or s2 > best2[1]): best2 = (k2, s2, lab2)
k, sil, lab2 = best2; inst["cluster"] = 0; inst.loc[sub.index, "cluster"] = lab2; inst.loc[inst.video == "ga10Trim", "cluster"] = k + 1
inst.to_csv(f"{OUT}/instances.csv", index=False)
tr = tr.merge(inst.groupby(["video", "track"])["cluster"].agg(lambda x: x.mode().iloc[0]).reset_index(), on=["video", "track"])
tr.to_csv(f"{OUT}/tracks.csv", index=False)
print(inst.groupby("cluster")[["L_720", "W_720", "W_over_L", "taper_head_q", "taper_tail_q", "dark"]].median().round(3).to_string())
print(pd.crosstab(inst.video, inst.cluster).to_string())

# ---------------- figures
vids = sorted(tr.video.unique()); vcol = {v: PAL[i % len(PAL)] for i, v in enumerate(vids)}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for v in vids:
    d = tr[tr.video == v]
    ax[0].scatter(d.L_720, d.W_720, s=30 + 2*d.n, color=vcol[v], label=v, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax[1].scatter(d.L_720, d.W_over_L, s=30 + 2*d.n, color=vcol[v], alpha=0.85, edgecolor="white", linewidth=0.8)
    ax[2].scatter(d.taper_head_q, d.taper_tail_q, s=30 + 2*d.n, color=vcol[v], alpha=0.85, edgecolor="white", linewidth=0.8)
ax[0].set(xlabel="body length, px (720p scale)", ylabel="mid-body width, px", title="size per track (marker ∝ frames)")
ax[1].set(xlabel="body length, px", ylabel="width / length", title="aspect ratio")
ax[2].set(xlabel="head quarter width / mid", ylabel="tail quarter width / mid", title="taper of the two ends")
ax[0].legend(fontsize=7, frameon=False, ncol=2)
for a in ax: a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_size_by_video.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ccol = {c: PAL[c-1] for c in sorted(inst.cluster.unique())}
for c in sorted(inst.cluster.unique()):
    d = inst[inst.cluster == c]
    ax[0].scatter(d.L_720, d.W_over_L, s=18, color=ccol[c], label=f"cluster {c} (n={len(d)} instances)", alpha=0.7, edgecolor="white", linewidth=0.5)
    P = np.array([[float(x) for x in p.split(";")] for p in d.profile])[::3]
    s = np.linspace(0, 1, 49)
    for p in P: ax[1].plot(s, p, color=ccol[c], alpha=0.25, lw=1)
    ax[1].plot(s, np.median(P, 0), color=ccol[c], lw=2.5, label=f"cluster {c}")
ax[0].set(xlabel="body length, px (720p scale)", ylabel="width / length", title=f"Ward clusters on instances (ga10 held out), k={k}, silhouette={sil:.2f}")
ax[1].set(xlabel="position along body (0 = head as oriented by motion)", ylabel="width / mid-body width", title="width profiles (every 3rd instance) + cluster median")
for a in ax: a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_clusters.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist([inst[inst.video.str.startswith("IMG")].L_720, inst[inst.video.str.startswith("ga")].L_720], bins=30, stacked=True,
        color=[PAL[0], PAL[1]], label=["IMG series (720p)", "ga series (1080p, scaled ×2/3)"])
ax.set(xlabel="body length, px (720p scale)", ylabel="instances", title="length distribution of all measured instances"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{OUT}/fig_length_hist.png", dpi=150); plt.close(fig)

# ---------------- gallery: per cluster, 16 instances spread over the length range, common px scale
H = 140
for c in sorted(inst.cluster.unique()):
    d = inst[inst.cluster == c].sort_values("L_720"); pick = d.iloc[np.unique(np.linspace(0, len(d)-1, 16).astype(int))]
    tiles = []
    for r in pick.itertuples():
        x = next(x for x in crops if x["video"] == r.video and x["track"] == r.track and x["frame"] == r.frame); im = x["img"]
        im = cv2.resize(im, None, fx=x["sc"]*0.5, fy=x["sc"]*0.5, interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]; canvas = np.full((H+40, max(w, 200), 3), 245, np.uint8); canvas[20:20+min(h, H), :min(w, canvas.shape[1])] = im[:H, :canvas.shape[1]]
        cv2.putText(canvas, f"{r.video} t{r.track} f{r.frame} L{r.L_720:.0f} W/L{r.W_over_L:.3f}", (3, 14), 0, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    W_ = max(t.shape[1] for t in tiles); tiles = [np.pad(t, ((0, 0), (0, W_-t.shape[1]), (0, 0)), constant_values=245) for t in tiles]
    per = 4; rows_ = [np.hstack(tiles[i:i+per] + [np.full_like(tiles[0], 245)]*(per - len(tiles[i:i+per]))) for i in range(0, len(tiles), per)]
    cv2.imwrite(f"{OUT}/gallery_cluster{c}.jpg", np.vstack(rows_), [1, 85])
print("done")
