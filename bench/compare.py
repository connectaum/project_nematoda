"""Aggregate all methods: synthetic metrics table, real-cluster proxy metrics, contact sheets.
  python compare.py   -> bench/results/summary_synth.csv, summary_real.csv, contact_*.jpg, fig_bench.png"""
import os, glob, csv, json, numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

RES = os.path.join(ROOT, "bench", "results")
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948"]
syn = pd.concat([pd.read_csv(f) for f in sorted([f for f in glob.glob(f"{RES}/*_synth.csv") if "summary" not in f])], ignore_index=True)
real = pd.concat([pd.read_csv(f) for f in sorted([f for f in glob.glob(f"{RES}/*_real.csv") if "summary" not in f])], ignore_index=True)
methods = list(dict.fromkeys(syn.method))

# ---- synthetic summary
g = syn.groupby(["method", "type"]).agg(all_recovered=("all_recovered", "mean"), base_ok=("base_ok", "mean"), donor_ok=("donor_ok", "mean"),
                                       extra=("extra", "mean"), miss=("miss", "mean"), err=("err_mean", "median")).reset_index()
tot = syn.groupby("method").agg(all_recovered=("all_recovered", "mean"), base_ok=("base_ok", "mean"), donor_ok=("donor_ok", "mean"),
                                extra=("extra", "mean"), miss=("miss", "mean"), err=("err_mean", "median")).reset_index(); tot["type"] = "ALL"
summ = pd.concat([tot, g], ignore_index=True).round(3); summ.to_csv(f"{RES}/summary_synth.csv", index=False)
print(summ.pivot(index="method", columns="type", values="all_recovered").round(2).to_string())
# per-video worm recovery (both worms) for the best methods
print(syn.groupby(["method", "video"]).all_recovered.mean().unstack().round(2).to_string())

# ---- real clusters: expected worm count from the context frames (labelled worms just before / after the episode)
exp = {}
for v in videos():
    qc = {int(r["frame"]): r for r in csv.DictReader(open(f"{ROOT}/auto_labels/{v}/qc_frames.csv"))}
    idx = load_index(v)
    for ep in sorted({int(r["episode"]) for r in idx if r["kind"] == "cluster"}):
        ctxf = [int(r["frame"]) for r in idx if r["kind"] == "context" and int(r["episode"]) == ep]
        n = [int(qc[f]["n_worms"]) + int(int(qc[f]["n_comp"]) > int(qc[f]["n_worms"])) for f in ctxf if f in qc]   # labelled worms + one partial/cluster comp
        exp[(v, ep)] = max(n) if n else np.nan
real["expected"] = [exp.get((r.video, r.episode), np.nan) for r in real.itertuples()]
cl = real[real.kind == "cluster"].copy()
cl["count_ok"] = (cl.n_plausible == cl.expected).astype(float); cl.loc[cl.expected.isna(), "count_ok"] = np.nan
rs = cl.groupby("method").agg(frames=("frame", "size"), worms_per_frame=("n_pred", "mean"), plausible_per_frame=("n_plausible", "mean"),
                              two_plus=("n_plausible", lambda x: (x >= 2).mean()), count_ok=("count_ok", "mean")).round(3)
rs.to_csv(f"{RES}/summary_real.csv"); print(rs.to_string())
print(cl.groupby(["method", "video"]).n_plausible.mean().unstack().round(2).to_string())

# ---- figure: all_recovered by type per method
fig, ax = plt.subplots(figsize=(9, 4)); types = ["X", "shallow", "parallel", "tip"]; w = 0.8/len(methods)
for i, m in enumerate(methods):
    vals = [summ[(summ.method == m) & (summ.type == t)].all_recovered.values[0] for t in types]
    ax.bar(np.arange(len(types)) + i*w, vals, w*0.92, color=PAL[i % 8], label=m)
ax.set_xticks(np.arange(len(types)) + w*(len(methods)-1)/2); ax.set_xticklabels(types)
ax.set(ylabel="both worms recovered (err < 5% L)", title="synthetic crossings: fraction of frames fully solved", ylim=(0, 1))
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{RES}/fig_bench.png", dpi=150); plt.close(fig)

# ---- contact sheets on real cluster frames: one row per frame, one column per method
def sheet(frames, fname, size=300):
    rows = []
    for v, f in frames:
        ctx = VideoCtx(v); img = load_frame(v, f); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); m = ctx.segment(gray)
        comps = components(m)
        if not comps: continue
        c = max(comps, key=lambda c: c["area"]); y0, x0, y1, x1 = c["bbox"]; cx, cy = (x0+x1)//2, (y0+y1)//2; s = max(y1-y0, x1-x0)//2 + 50
        X0, Y0 = max(cx-s, 0), max(cy-s, 0); X1, Y1 = min(cx+s, ctx.W), min(cy+s, ctx.H)
        tiles = []
        base = img[Y0:Y1, X0:X1].copy(); base = cv2.resize(base, (size, size)); cv2.putText(base, f"{v} f{f}", (4, 16), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA); tiles.append(base)
        for mth in methods:
            pf = f"{RES}/pred_{mth}_{v}_{f:06d}.npy"
            arr = np.load(pf, allow_pickle=True) if os.path.exists(pf) else np.array([], dtype=object)
            preds = [np.asarray(p, float) for p in (arr if arr.dtype == object else list(arr))]
            ov = draw_midlines(img, [np.asarray(p) for p in preds], thick=3)[Y0:Y1, X0:X1]; ov = cv2.resize(ov, (size, size))
            cv2.putText(ov, f"{mth[:14]} n={len(preds)}", (4, 16), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA); tiles.append(ov)
        rows.append(np.hstack(tiles))
    if rows: cv2.imwrite(fname, np.vstack(rows), [1, 80])

sel = []
for v in videos():
    idx = [r for r in load_index(v) if r["kind"] == "cluster"]
    if not idx: continue
    for r in [idx[int(i)] for i in np.linspace(0, len(idx)-1, min(6, len(idx)))]: sel.append((v, int(r["frame"])))
for i in range(0, len(sel), 12): sheet(sel[i:i+12], f"{RES}/contact_real_{i//12+1}.jpg")
print("contact sheets:", len(sel), "frames")
