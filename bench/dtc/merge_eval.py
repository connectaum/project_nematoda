"""Offline study of the ensemble merge rule (DTC base + fine-tuned model).

Loads the two SINGLE-model run_dtc2.py outputs for a video (same thr/NMS), reproduces the
ensemble merge frame by frame and compares representative-selection rules:
  score   - highest score wins (what run_dtc2.py / dtc_infer.py do today)
  longest - longest midline wins
  nearL   - length closest to the video's L_med wins
  mean    - average of the cluster members after flip alignment (score = max)
  ftfirst - the fine-tuned model's midline wins whenever it is in the cluster (base is then used
            only for recall: worms the ft model missed entirely)
Prints the usual metrics plus length-accuracy ones, and duplicate-cluster diagnostics.

usage: python merge_eval.py <base_tag> <ft_tag> [dedup]
       e.g. python merge_eval.py _b03 r5_6000 60
"""
import sys, pickle, json
import numpy as np
from pathlib import Path

LMED = {"IMG_8370": 536.8, "IMG_8373": 539.6, "IMG_8374": 527.1}
OUT = Path("/work/out")
base_tag = sys.argv[1] if len(sys.argv) > 1 else "_b03"
ft_tag   = sys.argv[2] if len(sys.argv) > 2 else "r5_6000"
DEDUP    = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
RULES = ["score", "longest", "nearL", "mean", "ftfirst", "ftsafe"]


def length(w):
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())


def mdist(u, v):
    return min(np.linalg.norm(u - v, axis=1).mean(), np.linalg.norm(u - v[::-1], axis=1).mean())


def clusters(cands):
    """cands: list of (w, score, model). Greedy, score-descending, same order as run_dtc2.py."""
    cl = []
    for w, s, m in sorted(cands, key=lambda t: -t[1]):
        for c in cl:
            if mdist(w, c[0][0]) <= DEDUP:
                c.append((w, s, m)); break
        else:
            cl.append([(w, s, m)])
    return cl


def pick(cluster, rule, L):
    if len(cluster) == 1 or rule == "score":
        return cluster[0][0], cluster[0][1]
    if rule == "longest":
        w, s, _ = max(cluster, key=lambda t: length(t[0])); return w, s
    if rule == "nearL":
        w, s, _ = min(cluster, key=lambda t: abs(length(t[0]) - L)); return w, s
    if rule == "ftfirst":
        ft = [c for c in cluster if c[2] == "ft"]
        w, s, _ = (max(ft, key=lambda t: t[1]) if ft else cluster[0]); return w, s
    if rule == "ftsafe":
        ft = [c for c in cluster if c[2] == "ft"]
        if not ft: return cluster[0][0], cluster[0][1]
        w, s, _ = max(ft, key=lambda t: t[1]); l = length(w)
        if 0.85 * L <= l <= 1.15 * L: return w, s
        others = [c for c in cluster if c[2] != "ft"]
        if others:
            wb, sb, _ = max(others, key=lambda t: t[1])
            if abs(length(wb) - L) < abs(l - L): return wb, sb
        return w, s
    if rule == "mean":
        ref = cluster[0][0]; acc = [ref]
        for w, _, _ in cluster[1:]:
            d = np.linalg.norm(ref - w, axis=1).mean(); df = np.linalg.norm(ref - w[::-1], axis=1).mean()
            acc.append(w if d <= df else w[::-1])
        return np.mean(acc, axis=0), max(c[1] for c in cluster)
    raise ValueError(rule)


for vid, L in LMED.items():
    pb, pf = OUT / f"{vid}_s6{base_tag}.pkl", OUT / f"{vid}_s6{ft_tag}.pkl"
    if not (pb.exists() and pf.exists()):
        print(f"{vid}: missing {pb.name if not pb.exists() else pf.name}"); continue
    db, df_ = pickle.load(open(pb, "rb")), pickle.load(open(pf, "rb"))
    rb, rf = db["results"], df_["results"]
    cs = sorted(set(rb) & set(rf))
    stats = {r: dict(n=[], npl=[], Ls=[]) for r in RULES}
    stats["base"] = dict(n=[], npl=[], Ls=[]); stats["ft"] = dict(n=[], npl=[], Ls=[])
    ndup = nclu = 0; base_wins = 0; dLs = []
    for c in cs:
        cands = [(w, float(s), "base") for w, s in zip(rb[c]["w"], rb[c]["s"])] + \
                [(w, float(s), "ft") for w, s in zip(rf[c]["w"], rf[c]["s"])]
        cl = clusters(cands)
        nclu += len(cl)
        for cluster in cl:
            models = {m for _, _, m in cluster}
            if len(cluster) > 1 and models == {"base", "ft"}:
                ndup += 1
                wb = [w for w, _, m in cluster if m == "base"][0]
                wf = [w for w, _, m in cluster if m == "ft"][0]
                base_wins += int(cluster[0][2] == "base")
                dLs.append(length(wb) - length(wf))
        for r in RULES:
            ws = [pick(cluster, r, L)[0] for cluster in cl]
            Ls = [length(w) for w in ws]
            stats[r]["n"].append(len(Ls)); stats[r]["npl"].append(sum(0.6 * L <= l <= 1.3 * L for l in Ls))
            stats[r]["Ls"] += Ls
        for key, res in (("base", rb), ("ft", rf)):
            Ls = [length(w) for w in res[c]["w"]]
            stats[key]["n"].append(len(Ls)); stats[key]["npl"].append(sum(0.6 * L <= l <= 1.3 * L for l in Ls))
            stats[key]["Ls"] += Ls
    print(f"\n=== {vid} (L_med {L:.0f}, {len(cs)} frames, dedup {DEDUP:.0f}px) ===")
    print(f"clusters {nclu}, of them base+ft duplicates {ndup} ({ndup/max(nclu,1):.0%}); "
          f"base wins by score in {base_wins}/{max(ndup,1)} ({base_wins/max(ndup,1):.0%}); "
          f"median L(base)-L(ft) on duplicates {np.median(dLs) if dLs else float('nan'):+.0f} px")
    print(f"{'rule':8s} {'det/frame':>9s} {'plaus/frame':>11s} {'ge2':>6s} {'ge1':>6s} "
          f"{'medL':>6s} {'dL%':>6s} {'|dL|% med':>9s} {'<0.85':>6s} {'>1.15':>6s}")
    for r in ["base", "ft"] + RULES:
        d = stats[r]; Ls = np.array(d["Ls"]); npl = np.array(d["npl"]); n = np.array(d["n"])
        pl = Ls[(Ls >= 0.6 * L) & (Ls <= 1.3 * L)]
        print(f"{r:8s} {n.mean():9.3f} {npl.mean():11.3f} {(npl>=2).mean():6.3f} {(npl>=1).mean():6.3f} "
              f"{np.median(Ls):6.0f} {(np.median(Ls)-L)/L*100:+6.1f} "
              f"{np.median(np.abs(pl-L))/L*100:9.1f} {np.mean(Ls<0.85*L):6.3f} {np.mean(Ls>1.15*L):6.3f}")
