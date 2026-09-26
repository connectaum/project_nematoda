"""Hypothesis 3: cheap heuristics on top of M1.
  M4a_ridge : skeleton graph built from a GRAYSCALE ridge filter (Sato) instead of the binary mask -> side-by-side worms give two ridges
  M4b_trail : + ring/coil handling: single-worm-sized components whose graph search fails get the longest TRAIL of the unfilled mask
Registers both into methods.METHODS so run_bench.py can run them:  python run_bench.py M4a_ridge,M4b_trail"""
import numpy as np, cv2, scipy.ndimage as ndi
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
import methods as M
from common import *
from trail import longest_trail

def segment_nofill(ctx, gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0); bg = float(np.median(g))
    m = (g < ctx.thr + (bg - ctx.bg_ref)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, DISK7).astype(bool)
    return remove_small_objects(m, max_size=800)

def ridge_mask(gray_sub, mask_sub, width):
    inv = (255 - gray_sub).astype(float)
    r = sato(inv, sigmas=[max(1.5, width/6), width/4, width/3], black_ridges=False)
    r[~ndi.binary_dilation(mask_sub, iterations=2)] = 0
    if r.max() <= 0: return mask_sub
    rm = r > 0.25*r.max()
    rm = ndi.binary_closing(rm, iterations=2)
    return remove_small_objects(rm, max_size=int(width*width))

def m4(img, ctx, use_ridge=True, use_trail=True, **kw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); out = []
    full = ctx.segment(gray); nofill = segment_nofill(ctx, gray) if use_trail else None
    for c in components(full):
        sub, off = M.crop(c["mask"]); sk = skeletonize(sub); w = M.comp_width(sub, sk)
        y0, x0 = off[1], off[0]; h, wd = sub.shape
        gsub = gray[y0:y0+h, x0:x0+wd]
        src = ridge_mask(gsub, sub, w) if use_ridge else sub
        rng = (0.45*ctx.L_med, 1.35*ctx.L_med)
        cands, _ = M.graph_paths(src, w, rng)
        if not cands and use_ridge: cands, _ = M.graph_paths(sub, w, rng)      # ridge failed: fall back to the binary graph
        if cands:
            for p in M.select_paths(cands, ctx.L_med):
                r = resample_polyline(p["co"] + off)
                if r is not None: out.append(r)
            continue
        # nothing plausible: single worm (maybe coiled) or partial
        if use_trail and c["area"] <= 1.4*ctx.A_med:
            nf = nofill[y0:y0+h, x0:x0+wd] & sub
            q = longest_trail(nf, L_ref=ctx.L_med, spur=0)
            if q is not None and 0.45*ctx.L_med <= arclen(q) <= 1.35*ctx.L_med: out.append(resample_polyline(q + off)); continue
        p = M.longest_path(sk)
        if p is not None and 0.45*ctx.L_med <= arclen(p) <= 1.35*ctx.L_med: out.append(resample_polyline(p + off))
    return out

M.METHODS["M4a_ridge"] = lambda img, ctx, **kw: m4(img, ctx, use_ridge=True, use_trail=False)
M.METHODS["M4b_trail"] = lambda img, ctx, **kw: m4(img, ctx, use_ridge=True, use_trail=True)
M.METHODS["M4c_trailonly"] = lambda img, ctx, **kw: m4(img, ctx, use_ridge=False, use_trail=True)
