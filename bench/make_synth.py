"""Synthetic crossings: paste a real worm (cut from another clean frame of the same video) onto a clean frame
with multiplicative (transmitted-light) blending, so the overlap region darkens like a real overlap.
Ground truth = both 49-pt midlines + instance masks.  Output: bench/synth/<video>/<id>.jpg, <id>.json, <id>_inst.png

  python make_synth.py [n_per_type=20] [seed=0]
"""
import sys, os, json, numpy as np, cv2
from common import *

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
OUT = os.path.join(ROOT, "bench", "synth")
TYPES = ["X", "shallow", "parallel", "tip"]

def worm_instances(ctx, f):
    """(mask, midline, width) for each fully labelled worm in a clean frame."""
    img = load_frame(ctx.v, f); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = ctx.segment(gray); comps = components(m); out = []
    for k, p in ctx.gt_midlines(f):
        xi = np.clip(p[:, 0].round().astype(int), 0, ctx.W - 1); yi = np.clip(p[:, 1].round().astype(int), 0, ctx.H - 1)
        best = max(comps, key=lambda c: c["mask"][yi, xi].sum(), default=None)
        if best is None or best["mask"][yi, xi].mean() < 0.9: continue
        if best["area"] > 1.5 * ctx.A_big: continue
        y0, x0, y1, x1 = best["bbox"]
        if x0 <= 3 or y0 <= 3 or x1 >= ctx.W - 3 or y1 >= ctx.H - 3: continue
        # strict instance quality: clean skeleton (few spurs) and length consistent with the track
        from skimage.morphology import skeletonize
        from methods import longest_path
        sub = best["mask"][y0:y1, x0:x1]; sk = skeletonize(sub); lp = longest_path(sk)
        if lp is None or (sk.sum() - len(lp)) > 0.12 * len(lp): continue
        L_ref = ctx.tracks[k]["body_length_px"] if k < len(ctx.tracks) else ctx.L_med
        if abs(arclen(p) / L_ref - 1) > 0.2: continue
        dist = cv2.distanceTransform(best["mask"].astype(np.uint8), cv2.DIST_L2, 5)
        w = 2 * np.median(dist[yi[8:-8], xi[8:-8]])
        out.append(dict(mask=best["mask"], mid=p, width=float(w), frame=int(f), img=img, gray=gray))
    return out

def tangent(p, s):
    i = int(round(s * (len(p) - 1))); i = min(max(i, 2), len(p) - 3)
    t = p[i + 2] - p[i - 2]; return p[i], t / (np.linalg.norm(t) + 1e-9)

def compose(base, donor, typ, ctx):
    """Return (image, gt_list, inst_mask, meta) or None."""
    A, B = base, donor
    if typ == "X": ang = rng.uniform(60, 120); sa, sb = rng.uniform(0.25, 0.75, 2); off = 0
    elif typ == "shallow": ang = rng.uniform(15, 45); sa, sb = rng.uniform(0.25, 0.75, 2); off = 0
    elif typ == "parallel":
        ang = rng.uniform(-8, 8) + rng.choice([0, 180]); sa = rng.uniform(0.35, 0.65); sb = sa + rng.uniform(-0.1, 0.1)
        off = 0.5 * (A["width"] + B["width"]) * rng.uniform(0.75, 1.0)
    else:  # tip: donor end touches base body
        ang = rng.uniform(-180, 180); sa = rng.uniform(0.15, 0.85); sb = rng.choice([0.02, 0.98]); off = 0
    ang *= rng.choice([-1, 1])
    pa, ta = tangent(A["mid"], sa); pb, tb = tangent(B["mid"], sb)
    # rotation that takes tb to direction ta rotated by ang
    target = np.array([[np.cos(np.radians(ang)), -np.sin(np.radians(ang))], [np.sin(np.radians(ang)), np.cos(np.radians(ang))]]) @ ta
    rot = np.degrees(np.arctan2(target[1], target[0]) - np.arctan2(tb[1], tb[0]))
    normal = np.array([-ta[1], ta[0]]) * rng.choice([-1, 1])
    dest = pa + normal * off
    M = cv2.getRotationMatrix2D((float(pb[0]), float(pb[1])), -rot, 1.0)   # cv2 uses y-down: negative = CCW in image coords
    M[:, 2] += dest - pb
    # transform donor midline & mask
    midB = (M[:, :2] @ B["mid"].T).T + M[:, 2]
    if midB.min() < 8 or midB[:, 0].max() > ctx.W - 8 or midB[:, 1].max() > ctx.H - 8: return None
    maskB = cv2.warpAffine(B["mask"].astype(np.uint8), M, (ctx.W, ctx.H), flags=cv2.INTER_NEAREST) > 0
    # ratio image of the donor crop: gray / local background, 1 outside a dilated mask
    ys, xs = np.nonzero(B["mask"]); y0, y1, x0, x1 = max(ys.min() - 20, 0), min(ys.max() + 21, ctx.H), max(xs.min() - 20, 0), min(xs.max() + 21, ctx.W)
    g = B["gray"].astype(np.float32)
    dil = cv2.dilate(B["mask"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bgB = np.median(g[y0:y1, x0:x1][~dil[y0:y1, x0:x1]])
    ratio = np.ones_like(g)
    # soft edge: full ratio inside mask, blend to 1 over the dilation ring
    soft = cv2.GaussianBlur(B["mask"].astype(np.float32), (7, 7), 0)
    r = np.clip(g / bgB, 0, 1.2)
    ratio = 1 + (r - 1) * np.clip(soft * 1.4, 0, 1)
    ratio[~dil] = 1.0
    ratioW = cv2.warpAffine(ratio, M, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR, borderValue=1.0)
    img = A["img"].astype(np.float32) * ratioW[..., None]
    img = np.clip(img + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)
    inst = np.zeros((ctx.H, ctx.W), np.uint8); inst[A["mask"]] = 1; inst[maskB] += 2     # 3 = overlap
    overlap = int((inst == 3).sum()); touching = overlap > 0 or (cv2.dilate(maskB.astype(np.uint8), np.ones((5, 5), np.uint8)) & A["mask"]).any()
    if not touching: return None
    meta = dict(type=typ, angle=float(abs(ang)), s_base=float(sa), s_donor=float(sb), overlap_px=overlap,
                base_frame=A["frame"], donor_frame=B["frame"], width_base=A["width"], width_donor=B["width"])
    gts = [dict(mid=A["mid"].tolist(), L=arclen(A["mid"]), role="base")]
    for k, p in ctx.gt_midlines(A["frame"]):                  # other fully labelled worms already in the base frame
        if np.abs(p - A["mid"]).max() > 1: gts.append(dict(mid=p.tolist(), L=arclen(p), role="other"))
    gts.append(dict(mid=midB.tolist(), L=arclen(midB), role="donor"))
    return img, gts, inst, meta

total = 0
for v in videos():
    ctx = VideoCtx(v)
    clean = [r["frame"] for r in load_index(v) if r["kind"] == "clean"]
    inst = []
    for f in clean: inst += worm_instances(ctx, f)
    if len(inst) < 2: print(v, "skip: <2 clean worm instances"); continue
    os.makedirs(f"{OUT}/{v}", exist_ok=True); n_v = 0
    for typ in TYPES:
        made = 0; tries = 0
        while made < N_PER and tries < N_PER * 15:
            tries += 1
            i, j = rng.choice(len(inst), 2, replace=False)
            res = compose(inst[i], inst[j], typ, ctx)
            if res is None: continue
            img, gts, im, meta = res; sid = f"{typ}_{made:03d}"
            cv2.imwrite(f"{OUT}/{v}/{sid}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{OUT}/{v}/{sid}_inst.png", im)
            json.dump(dict(video=v, id=sid, gt=gts, **meta), open(f"{OUT}/{v}/{sid}.json", "w"))
            made += 1
        n_v += made
    print(v, "clean instances", len(inst), "synthetic", n_v); total += n_v
print("total", total)
