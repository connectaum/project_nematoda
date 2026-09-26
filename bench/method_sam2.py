"""Hypothesis 1: temporal propagation. SAM2 video predictor, prompted with the per-worm masks of the
last clean context frame before a cluster episode (or the first after it, propagating backwards),
then masks -> skeleton -> midline per object. Real cluster/context frames only (no synth: single-frame).
  PYTORCH_ENABLE_MPS_FALLBACK=1 python method_sam2.py [model] [videos...]"""
import sys, os, glob, time, shutil, warnings, numpy as np, cv2, pandas as pd, torch; warnings.filterwarnings("ignore")
from sam2.sam2_video_predictor import SAM2VideoPredictor
from skimage.morphology import skeletonize
from common import *
from methods import longest_path, crop

mt = sys.argv[1] if len(sys.argv) > 1 else "facebook/sam2.1-hiera-small"
only = sys.argv[2:]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
predictor = SAM2VideoPredictor.from_pretrained(mt, device=dev)
name = "SAM2_" + mt.split("-")[-1]
RES = os.path.join(ROOT, "bench", "results"); os.makedirs(RES, exist_ok=True)
TMP = os.path.join(ROOT, "bench", "_sam2_tmp")

def midlines(masks, ctx):
    out = []
    for m in masks:
        if m.sum() < 0.15*ctx.A_med: continue
        sub, off = crop(m); p = longest_path(skeletonize(sub))
        if p is None: continue
        p = p + off
        if 0.45*ctx.L_med <= arclen(p) <= 1.35*ctx.L_med: out.append(resample_polyline(p))
    return out

def prompt_masks(img, ctx):
    """separate worm-sized components of the clean-context frame; None if something is already merged"""
    comps = components(ctx.segment(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), a_min=int(0.4*ctx.A_med))
    if not comps or any(c["area"] > 1.7*ctx.A_med for c in comps): return None
    return [c["mask"] for c in comps]

rows = []; t0 = time.time()
for v in videos():
    if only and v not in only: continue
    ctx = VideoCtx(v); idx = load_index(v)
    eps = sorted({int(r["episode"]) for r in idx if r["kind"] in ("cluster", "context")})
    for e in eps:
        fr = sorted({int(r["frame"]) for r in idx if r["kind"] in ("cluster", "context") and int(r["episode"]) == e})
        kind = {int(r["frame"]): r["kind"] for r in idx if int(r["episode"]) == e}
        cl = [f for f in fr if kind[f] == "cluster"]
        pre = [f for f in fr if kind[f] == "context" and f < cl[0]]; post = [f for f in fr if kind[f] == "context" and f > cl[-1]]
        start, rev, pm = None, False, None
        for cand, r in [(pre[::-1], False), (post, True)]:
            for f in cand:
                pm = prompt_masks(load_frame(v, f), ctx)
                if pm: start, rev = f, r; break
            if pm: break
        if not pm:
            print(v, e, "no clean prompt frame", flush=True)
            for f in fr: rows.append(dict(method=name, video=v, frame=f, kind=kind[f], episode=e, n_pred=0, n_plausible=0, lens="", note="no_prompt"))
            continue
        shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
        for i, f in enumerate(fr): shutil.copy(f"{FRAMES}/{v}/{f:06d}.jpg", f"{TMP}/{i:05d}.jpg")
        with torch.inference_mode():
            st = predictor.init_state(video_path=TMP, offload_video_to_cpu=True)
            si = fr.index(start)
            for k, m in enumerate(pm): predictor.add_new_mask(st, frame_idx=si, obj_id=k+1, mask=torch.from_numpy(m))
            res = {}
            for fi, oids, logits in predictor.propagate_in_video(st, start_frame_idx=si, reverse=rev):
                res[fi] = [(logits[i, 0] > 0).cpu().numpy() for i in range(len(oids))]
            if rev:  # also cover frames after the start frame (if any)
                for fi, oids, logits in predictor.propagate_in_video(st, start_frame_idx=si, reverse=False):
                    res.setdefault(fi, [(logits[i, 0] > 0).cpu().numpy() for i in range(len(oids))])
            predictor.reset_state(st)
        for i, f in enumerate(fr):
            ms = res.get(i, []); preds = midlines(ms, ctx); Ls = [arclen(p)/ctx.L_med for p in preds]
            rows.append(dict(method=name, video=v, frame=f, kind=kind[f], episode=e, n_pred=len(preds),
                             n_plausible=sum(0.6 <= l <= 1.3 for l in Ls), lens=";".join(f"{l:.2f}" for l in Ls),
                             note=f"{'rev' if rev else 'fwd'}@{start}:{len(pm)}obj"))
            np.save(f"{RES}/pred_{name}_{v}_{f:06d}.npy", np.array(preds, dtype=object), allow_pickle=True)
            if ms: np.save(f"{RES}/mask_{name}_{v}_{f:06d}.npy", np.packbits(np.stack(ms), axis=-1))
        print(v, e, f"{len(fr)} frames, {'rev' if rev else 'fwd'} from {start}, {len(pm)} objs, {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{RES}/{name}_real.csv", index=False); cl = df[df.kind == "cluster"]
print(f"{name}: real cluster frames {len(cl)} mean worms/frame={cl.n_pred.mean():.2f} plausible={cl.n_plausible.mean():.2f} frames>=2 plausible={(cl.n_plausible>=2).mean():.2f}")
