"""Run methods on the synthetic benchmark and on real cluster frames.
  python run_bench.py [method,method,...]  -> bench/results/<method>_synth.csv, <method>_real.csv
"""
import sys, os, json, glob, time, numpy as np, cv2, pandas as pd
from common import *
import methods as M
try: import method_m4
except Exception as ex: print("method_m4 not loaded:", ex)

which = sys.argv[1].split(",") if len(sys.argv) > 1 else list(M.METHODS)
RES = os.path.join(ROOT, "bench", "results"); os.makedirs(RES, exist_ok=True)
SYN = os.path.join(ROOT, "bench", "synth")

for name in which:
    fn = M.METHODS[name]; rows = []; t0 = time.time()
    for v in videos():
        if not os.path.isdir(f"{SYN}/{v}"): continue
        ctx = VideoCtx(v)
        for jf in sorted(glob.glob(f"{SYN}/{v}/*.json")):
            j = json.load(open(jf)); img = cv2.imread(jf.replace(".json", ".jpg"))
            try: preds = fn(img, ctx)
            except Exception as e: preds = []; print("ERR", name, jf, e)
            gts = [np.array(g["mid"]) for g in j["gt"]]; L = float(np.mean([g["L"] for g in j["gt"]]))
            s = score_frame(preds, gts, L, (ctx.W, ctx.H))
            # per-role recovery: was the base / donor worm itself recovered?
            pairs, _, _ = match(preds, gts)
            role_ok = {g["role"]: 0 for g in j["gt"]}
            for gi, pj, e in pairs:
                if e < 0.05*L: role_ok[j["gt"][gi]["role"]] = 1
            rows.append(dict(method=name, video=v, id=j["id"], type=j["type"], angle=j["angle"], overlap_px=j["overlap_px"],
                             base_ok=role_ok.get("base", 0), donor_ok=role_ok.get("donor", 0), **s))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/{name}_synth.csv", index=False)
    print(f"{name}: synth {len(df)} frames in {time.time()-t0:.0f}s  all_recovered={df.all_recovered.mean():.2f} "
          f"base_ok={df.base_ok.mean():.2f} donor_ok={df.donor_ok.mean():.2f} extra/frame={df.extra.mean():.2f}", flush=True)
    print(df.groupby("type")[["all_recovered", "base_ok", "donor_ok", "extra"]].mean().round(2).to_string())
    # real cluster frames
    rows = []; t0 = time.time()
    for v in videos():
        ctx = VideoCtx(v)
        for r in load_index(v):
            if r["kind"] not in ("cluster", "context"): continue
            img = load_frame(v, r["frame"])
            try: preds = fn(img, ctx)
            except Exception as e: preds = []; print("ERR", name, v, r["frame"], e)
            Ls = [arclen(p)/ctx.L_med for p in preds]
            rows.append(dict(method=name, video=v, frame=int(r["frame"]), kind=r["kind"], episode=int(r["episode"]),
                             n_pred=len(preds), n_plausible=sum(0.6 <= l <= 1.3 for l in Ls), lens=";".join(f"{l:.2f}" for l in Ls)))
            np.save(f"{RES}/pred_{name}_{v}_{int(r['frame']):06d}.npy", np.array(preds, dtype=object), allow_pickle=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/{name}_real.csv", index=False)
    cl = df[df.kind == "cluster"]
    print(f"{name}: real cluster frames {len(cl)} in {time.time()-t0:.0f}s  mean worms/frame={cl.n_pred.mean():.2f} "
          f"plausible={cl.n_plausible.mean():.2f} frames>=2 plausible={(cl.n_plausible>=2).mean():.2f}", flush=True)
