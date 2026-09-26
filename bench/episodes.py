"""Contact-episode statistics from auto_labels/<v>/qc_frames.csv (hypothesis 6: describe contacts instead of resolving them).
Episode = run of frames (gap<=10) where a frame is flagged 'cluster' OR its largest component area > 1.5*A_med (merged pair w/o branching)."""
import csv, json, glob, os, numpy as np, pandas as pd
rows = []
for d in sorted(glob.glob("auto_labels/*/qc_frames.csv")):
    v = d.split("/")[1]; s = json.load(open(f"auto_labels/{v}/summary.json")); A = s["median_area_px"]; fps = s["fps"]
    q = list(csv.DictReader(open(d)))
    def contact(r):
        return ("cluster" in r["flags"]) or (float(r["area"] or 0) > 1.5*A)
    eps = []; cur = None
    for r in q:
        f = int(r["frame"])
        if contact(r):
            if cur is None: cur = dict(start=f, end=f, n=0, big=0, branch=0, amax=0.0)
            elif f - cur["end"] > 10: eps.append(cur); cur = dict(start=f, end=f, n=0, big=0, branch=0, amax=0.0)
            cur["end"] = f; cur["n"] += 1; cur["big"] += float(r["area"] or 0) > 1.5*A
            cur["branch"] += "skeleton_branches" in r["flags"] or "cluster" in r["flags"]; cur["amax"] = max(cur["amax"], float(r["area"] or 0)/A)
    if cur: eps.append(cur)
    for e in eps:
        dur = (e["end"] - e["start"] + 1)
        if dur < 5: continue
        kind = "blob≈2 worms, unbranched (parallel?)" if e["branch"] < 0.3*e["n"] and e["amax"] > 1.5 else ("branched, area≈1 worm (self-coil/ring?)" if e["amax"] < 1.4 else "branched, ≥2 worms (crossing/tangle)")
        rows.append(dict(video=v, start=e["start"], end=e["end"], frames=dur, sec=round(dur/fps, 1), area_max=round(e["amax"], 2),
                         frac_branched=round(e["branch"]/e["n"], 2), guess=kind))
df = pd.DataFrame(rows); df.to_csv("bench/results/contact_episodes.csv", index=False)
print(df.to_string(index=False))
print("\nper video:"); tot = {v: json.load(open(f"auto_labels/{v}/summary.json"))["n_frames"] for v in df.video.unique()}
g = df.groupby("video").agg(episodes=("frames", "size"), frames_in_contact=("frames", "sum"), longest_s=("sec", "max"))
g["frac_of_video"] = [round(g.loc[v, "frames_in_contact"]/tot[v], 2) for v in g.index]; print(g.to_string())
