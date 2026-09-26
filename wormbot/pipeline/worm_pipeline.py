"""
Classical worm-pose pipeline (v2, multi-worm aware): mask -> components -> skeleton -> midline -> 49/5 points.

  python worm_pipeline.py <video.mp4> <out_dir>      [env OVERLAY_SCALE=0.5 for a smaller overlay.mp4]

Outputs in <out_dir>:
  points5_dlc.csv       5 points per worm per frame (DLC format; 4-level header scorer/individuals/bodyparts/coords
                        when the video has more than one worm track, plain 3-level header otherwise)
  CollectedData_auto.csv  same, indexed by labeled-data/<video>/img####.png
  labels49.npy          (N,49,2) for one track, (N,K,49,2) for K tracks; arc length from the head; NaN = hidden
  curvature.npy         (N,49) / (N,K,49) dimensionless curvature (kappa * body length)
  qc_frames.csv         per-frame QC: flags, n_comp, n_worms, per-track label_status
  tracks.json           per-track statistics (frames, body length, orientation score)
  summary.json, overlay.mp4

Changes vs v2 (v3, 2026-09-05)
  * clusters are no longer dropped outright: branched cluster components are handed to worm_split.split_cluster
    (skeleton graph, straight-through rule at junctions); if it finds >= 2 plausible-length midlines, they become
    'from_cluster' worm candidates (likelihood 0.5) and are tracked like ordinary components. Side-by-side worms and
    two-worm rings still end up as 'cluster' with no labels.

Changes vs v1
  * 'split': two pieces whose midline ends nearly touch (one worm broken by its translucent middle) get no labels
  * one global threshold per video (median over frames with a clear worm) instead of per-frame Otsu:
    per-frame Otsu on an EMPTY frame splits background noise in half and produces a giant fake "worm"
  * every connected component above a minimum area is a worm candidate; candidates are linked across frames
    into tracks by centroid proximity (worm1, worm2, ...); components much larger than a single worm
    (worms touching / coiled together) are flagged 'cluster' and get no labels
  * cv2 morphology instead of skimage closing (identical result, ~100x faster), skeleton on the bounding box
"""
import sys, os, json, collections, time
if sys.platform == "linux" and os.environ.get("PIPELINE_WORKERS", "0") != "1":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")   # fork-safe: no BLAS/OpenMP thread pools in the parent
import numpy as np, cv2, pandas as pd
if sys.platform == "linux" and os.environ.get("PIPELINE_WORKERS", "0") != "1": cv2.setNumThreads(1)          # idem for OpenCV (v3.1 multiprocessing pass 1)
from skimage.morphology import skeletonize, remove_small_objects, disk
from skimage.measure import label, regionprops
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worm_split import split_cluster, HAVE_SKAN       # v3: skeleton-graph splitting of clusters (bench M1, 2026-09-05)

VIDEO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
N_PTS = 5          # points to export (4 equal segments)
N_DENSE = 49       # dense midline for curvature
BORDER = 3         # px from frame edge counts as "touching border"
A_MIN = 1500       # px: smaller components are debris
GATE = 150         # px: max centroid jump between consecutive frames for the same track
GATE_LOST = 450    # px: re-acquire a track that was hidden (cluster / off-frame) for up to LOST_MAX frames
LOST_MAX = 200
OVERLAY_SCALE = float(os.environ.get("OVERLAY_SCALE", "1.0"))
DISK7 = disk(7).astype(np.uint8)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- global threshold
def frame_stats(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    t_otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return g, float(t_otsu), float(np.median(g))

cap = cv2.VideoCapture(VIDEO)
W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5)
NF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, NF // 150)
st = []
for f in range(0, NF, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    if not ok: continue
    _, t, bg = frame_stats(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)); st.append((t, bg))
cap.release()
st = np.array(st); contrast = st[:, 1] - st[:, 0]
keep = contrast >= 0.6 * np.percentile(contrast, 90)          # frames with a clear worm
THR_REL = np.median(st[keep, 0] + 0.55 * (st[keep, 1] - st[keep, 0]))   # relaxed: translucent head/tail are lighter than body
BG_REF = np.median(st[keep, 1])
print(f"global threshold {THR_REL:.1f} (bg {BG_REF:.1f}, worm contrast {np.median(contrast[keep]):.0f} levels, "
      f"{keep.sum()}/{len(keep)} sampled frames with a worm)", flush=True)

# ---------------------------------------------------------------- segmentation
def segment(gray):
    g, _, bg = frame_stats(gray)
    m = (g < THR_REL + (bg - BG_REF)).astype(np.uint8)       # follow slow illumination drift
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, DISK7).astype(bool)   # == skimage closing(m, disk(7))
    m = ndi.binary_fill_holes(m)
    m = remove_small_objects(m, max_size=800)
    return m

# ---------------------------------------------------------------- skeleton graph
NBR = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def skeleton_longest_path(sk):
    """Return ordered pixel list of the longest endpoint-to-endpoint path, #endpoints, #spur pixels."""
    pts = set(zip(*np.nonzero(sk)))
    if not pts: return None, 0, 0
    def nb(p): return [(p[0]+dy, p[1]+dx) for dy,dx in NBR if (p[0]+dy, p[1]+dx) in pts]
    deg = {p: len(nb(p)) for p in pts}
    ends = [p for p,d in deg.items() if d == 1]
    if len(ends) < 2:           # loop / ring -> fail
        return None, len(ends), 0
    def bfs(src):
        prev = {src: None}; dq = collections.deque([src]); last = src
        while dq:
            p = dq.popleft(); last = p
            for q in nb(p):
                if q not in prev: prev[q] = p; dq.append(q)
        return last, prev
    a, _ = bfs(ends[0]); b, prev = bfs(a)
    path = []; p = b
    while p is not None: path.append(p); p = prev[p]
    spur = len(pts) - len(path)                      # skeleton pixels not on the main path
    return np.array(path, float)[:, ::-1], len(ends), spur   # (x,y)

def resample(path, n, smooth=None):
    if len(path) < 8: return None
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    u = d / d[-1]
    s = smooth if smooth is not None else len(path) * 2.0
    try:
        tck, _ = splprep([path[:,0], path[:,1]], u=u, s=s, k=3)
    except Exception:
        return None
    uu = np.linspace(0, 1, 500); xy = np.array(splev(uu, tck)).T
    dd = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    target = np.linspace(0, dd[-1], n)
    return np.c_[np.interp(target, dd, xy[:,0]), np.interp(target, dd, xy[:,1])], dd[-1]

def width_profile(mask, pts):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.array([dist[int(round(y)), int(round(x))] if 0<=int(round(y))<mask.shape[0] and 0<=int(round(x))<mask.shape[1] else 0 for x,y in pts])

# ---------------------------------------------------------------- pass 1: components per frame
# v3.1 (2026-09-14): frames are independent in pass 1, so the video is cut into ranges processed by PIPELINE_WORKERS
# forked processes (default: min(4, cpu_count) on Linux, 1 elsewhere). Each worker decodes its own range (no frame
# IPC); results are concatenated in order, so the output is identical to the sequential run. Workers skip to their
# range with grab() instead of CAP_PROP_POS_FRAMES: on these (variable-frame-rate) phone videos seeking lands one
# frame off at some positions (checked on IMG_8374), which shifted labels by a frame in later ranges.
def analyse_frame(gray):
    m = segment(gray)
    lab = label(m); comps = []
    for r in sorted(regionprops(lab), key=lambda r: -r.area):
        if r.area < A_MIN: continue
        y0,x0,y1,x1 = r.bbox
        c = dict(area=int(r.area), centroid=(float(r.centroid[1]), float(r.centroid[0])),
                 touches_border=bool(x0<=BORDER or y0<=BORDER or x1>=W-BORDER or y1>=H-BORDER),
                 flags=[], dense=None, width=None, length=np.nan)
        # work on the bounding box (pad 1 px so skeleton endpoints are not clipped)
        oy, ox = max(y0-1,0), max(x0-1,0)
        sub = (lab[oy:y1+1, ox:x1+1] == r.label)
        sk = skeletonize(sub)
        path, n_end, spur = skeleton_longest_path(sk)
        c["n_end"] = n_end; c["spur_frac"] = 0.0
        if path is None: c["flags"].append("loop_or_empty")
        else:
            path = path + [ox, oy]
            c["spur_frac"] = spur / len(path)
            if spur > 0.15*len(path): c["flags"].append("skeleton_branches")
            if n_end >= 3 and HAVE_SKAN: c["sub"], c["sub_off"] = np.packbits(sub, axis=None), (ox, oy, sub.shape)   # kept for cluster splitting
            res = resample(path, N_DENSE)
            if res is None: c["flags"].append("spline_fail")
            else:
                c["dense"], L = res; c["length"] = float(L)
                c["width"] = width_profile(sub, c["dense"] - [ox, oy])
                mid = c["width"][5:-5]; c["width_cv"] = float(mid.std() / max(mid.mean(), 1e-6))
                c["area_per_len"] = r.area / L
        comps.append(c)
    return comps

def _pass1_range(bounds):
    a, b = bounds                                   # frames [a, b); b >= NF means "read to the end"
    if WORKERS > 1: cv2.setNumThreads(1)
    cap = cv2.VideoCapture(VIDEO)
    for _ in range(a):
        if not cap.grab(): cap.release(); return []
    out = []; t0 = time.time()
    for _ in range(b - a):
        ok, frame = cap.read()
        if not ok: break
        out.append(analyse_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if len(out) % 500 == 0: print(f"  [{a}..] {len(out)} frames, {time.time()-t0:.0f} s", flush=True)
    cap.release()
    return out

WORKERS = int(os.environ.get("PIPELINE_WORKERS", "0")) or (min(4, os.cpu_count() or 1) if sys.platform == "linux" else 1)
T0 = time.time(); frames = []            # per frame: list of component dicts
if WORKERS > 1 and NF >= 4 * WORKERS:
    import multiprocessing as mp
    edges = [round(NF * i / WORKERS) for i in range(WORKERS)] + [10**9]   # one contiguous range per worker; the last reads to EOF
    bounds = list(zip(edges[:-1], edges[1:]))                             # (container frame count can be off by a few)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for (a, b), res in zip(bounds, pool.imap(_pass1_range, bounds)):
            frames.extend(res)
            print(f"  {len(frames)}/{NF} frames, {time.time()-T0:.0f} s", flush=True)
            if len(res) < b - a and b < 10**9: break       # EOF inside a range: stop where the sequential reader would
else:
    frames = _pass1_range((0, 10**9))
N = len(frames)
print(f"pass 1 done: {N} frames in {time.time()-T0:.0f} s ({WORKERS} worker(s))", flush=True)

# ---------------------------------------------------------------- size sanity: single-worm reference area / length
def clean_comps():
    for comps in frames:
        for c in comps:
            if not c["touches_border"] and not c["flags"] and c["dense"] is not None: yield c
areas = np.array([c["area"] for c in clean_comps()]); lens = np.array([c["length"] for c in clean_comps()])
if len(areas) < 20:                      # fall back to all components with a midline
    areas = np.array([c["area"] for f in frames for c in f if c["dense"] is not None] or [np.nan])
    lens = np.array([c["length"] for f in frames for c in f if c["dense"] is not None] or [np.nan])
A_MED, L_MED = float(np.nanmedian(areas)), float(np.nanmedian(lens))
A_BIG = float(np.nanpercentile(areas, 90))          # size of the largest single worm in this video (worms can differ ~1.5x)
for comps in frames:
    for c in comps:
        branched = c["spur_frac"] > 0.15
        if ("loop_or_empty" in c["flags"] or c["area"] > 1.5*A_BIG or
                (branched and c["area"] > 1.2*A_BIG and (c["n_end"] >= 4 or c.get("width_cv", 0) > 0.35))):
            c["flags"].append("cluster")            # two+ worms touching (a single coiled worm keeps its area -> stays a worm, flagged skeleton_branches)
        elif c["area"] < 0.15*A_MED and not c["touches_border"]:
            c["flags"].append("small_fragment")

# ---------------------------------------------------------------- v3: split branched clusters via the skeleton graph
n_split_frames = 0
for comps in frames:
    children = []
    for c in comps:
        sub = c.pop("sub", None)
        if sub is None: continue
        if "cluster" not in c["flags"]: continue
        ox, oy, shp = c.pop("sub_off"); sub = np.unpackbits(sub)[:shp[0]*shp[1]].reshape(shp).astype(bool)
        w = 2*float(np.median(c["width"])) if c["width"] is not None and len(c["width"]) else 24.0   # width profile holds radii
        for p in split_cluster(sub, max(w, 4.0), L_MED):
            res = resample(p + [ox, oy], N_DENSE)
            if res is None: continue
            dense, L = res
            child = dict(area=int(c["area"] / 2), centroid=(float(dense[:, 0].mean()), float(dense[:, 1].mean())),
                         touches_border=c["touches_border"], flags=["from_cluster"], dense=dense, length=float(L),
                         n_end=2, spur_frac=0.0, width=width_profile(sub, dense - [ox, oy]))
            mid = child["width"][5:-5]; child["width_cv"] = float(mid.std() / max(mid.mean(), 1e-6)); child["area_per_len"] = child["area"] / L
            children.append(child)
    if children: comps.extend(children); n_split_frames += 1
print(f"cluster splitting: {n_split_frames} frames got worms out of a cluster", flush=True)
for comps in frames:
    for c in comps: c.pop("sub", None); c.pop("sub_off", None)

# a worm whose translucent middle breaks the mask shows up as two pieces whose ends point at each other -> 'split' (no labels)
SPLIT_D = 350      # px gap between the pieces' midline ends (translucent sections can be long)
SPLIT_ANG = 60     # deg: both end tangents must point into the gap
def pieces_of_one_worm(da, db):
    for ea, ta in ((da[0], da[0]-da[4]), (da[-1], da[-1]-da[-5])):
        for eb, tb in ((db[0], db[0]-db[4]), (db[-1], db[-1]-db[-5])):
            v = eb - ea; n = np.linalg.norm(v)
            if n == 0 or n > SPLIT_D: continue
            ca = np.dot(ta, v) / (np.linalg.norm(ta)*n + 1e-9); cb = np.dot(tb, -v) / (np.linalg.norm(tb)*n + 1e-9)
            if ca > np.cos(np.radians(SPLIT_ANG)) and cb > np.cos(np.radians(SPLIT_ANG)): return True
    return False
for comps in frames:
    ds = [c for c in comps if c["dense"] is not None and "cluster" not in c["flags"] and "small_fragment" not in c["flags"]]
    for i1 in range(len(ds)):
        for i2 in range(i1 + 1, len(ds)):
            if pieces_of_one_worm(ds[i1]["dense"], ds[i2]["dense"]):
                for c in (ds[i1], ds[i2]):
                    if "split" not in c["flags"]: c["flags"].append("split")

# ---------------------------------------------------------------- tracking: link components into worm tracks
tracks = []                          # each: dict(id, last_frame, last_centroid, frames: {frame: comp})
for k, comps in enumerate(frames):
    cand = [c for c in comps if not ({"cluster", "small_fragment", "split"} & set(c["flags"]))]
    pairs = []
    for c in cand:
        for t in tracks:
            gap = k - t["last_frame"]
            if gap > LOST_MAX: continue
            d = np.hypot(*(np.array(c["centroid"]) - t["last_centroid"]))
            size = abs(np.log(c["area"] / t["area"]))          # worms in one video can differ 2x in size
            if not c["touches_border"] and not t["at_border"] and size > 0.7: continue
            gate = GATE if gap == 1 else GATE_LOST
            if d < gate: pairs.append((d + 0.5*gap + 200*size, c, t))
    used_c, used_t = set(), set()
    for d, c, t in sorted(pairs, key=lambda p: p[0]):
        if id(c) in used_c or t["id"] in used_t: continue
        t["frames"][k] = c; t["last_frame"] = k; t["last_centroid"] = np.array(c["centroid"]); t["at_border"] = c["touches_border"]
        if not c["touches_border"]: t["area"] = 0.9*t["area"] + 0.1*c["area"]
        used_c.add(id(c)); used_t.add(t["id"]); c["track"] = t["id"]
    for c in cand:
        if id(c) not in used_c:
            t = dict(id=len(tracks), last_frame=k, last_centroid=np.array(c["centroid"]), frames={k: c},
                     area=float(c["area"]), at_border=c["touches_border"])
            tracks.append(t); c["track"] = t["id"]
# drop tiny tracks and static debris (small blob that never moves), renumber by first appearance
def is_debris(t):
    if len(t["frames"]) < max(5, 0.005*N): return True
    cs = np.array([c["centroid"] for c in t["frames"].values()]); areas_t = np.median([c["area"] for c in t["frames"].values()])
    return bool(cs.std(0).max() < 8 and areas_t < 0.3*A_MED)
for t in tracks:
    if is_debris(t):
        for c in t["frames"].values(): c.pop("track", None); c["flags"].append("small_fragment")
tracks = [t for t in tracks if not is_debris(t)]
tracks.sort(key=lambda t: min(t["frames"]))
for new_id, t in enumerate(tracks):
    t["id"] = new_id
    for c in t["frames"].values(): c["track"] = new_id
K = max(1, len(tracks))
print(f"{len(tracks)} worm track(s): " + ", ".join(f"worm{t['id']+1}: {len(t['frames'])} frames" for t in tracks), flush=True)

# ---------------------------------------------------------------- head/tail orientation, per track
EDGE = 40
def at_edge(p): return p[0] <= EDGE or p[1] <= EDGE or p[0] >= W-1-EDGE or p[1] >= H-1-EDGE

for t in tracks:
    ks = sorted(t["frames"]); prev = None
    for k in ks:
        c = t["frames"][k]; d = c["dense"]
        if d is None: continue
        if prev is not None:
            e0, e1 = at_edge(d[0]), at_edge(d[-1])
            if e0 != e1:
                free = d[-1] if e0 else d[0]
                to_head, to_tail = np.linalg.norm(free-prev[0]), np.linalg.norm(free-prev[-1])
                same, flip = (to_tail, to_head) if e0 else (to_head, to_tail)
            else:
                same = np.linalg.norm(d[0]-prev[0]) + np.linalg.norm(d[-1]-prev[-1])
                flip = np.linalg.norm(d[0]-prev[-1]) + np.linalg.norm(d[-1]-prev[0])
            if flip < same:
                c["dense"] = d[::-1].copy(); d = c["dense"]
                if c["width"] is not None: c["width"] = c["width"][::-1]
        prev = d
    score = 0.0; prev_k = None
    for k in ks:
        d = t["frames"][k]["dense"]
        if d is None: continue
        if prev_k is not None and k - prev_k <= 3:
            a = t["frames"][prev_k]["dense"]
            v = d.mean(0) - a.mean(0); axis = d[0] - d[-1]; n = np.linalg.norm(axis)
            if n > 1: score += np.dot(v, axis/n)
        prev_k = k
    ws = [t["frames"][k]["width"] for k in ks if t["frames"][k]["width"] is not None]
    wh = float(np.nanmean([w[:5].mean() for w in ws])) if ws else np.nan
    wt = float(np.nanmean([w[-5:].mean() for w in ws])) if ws else np.nan
    if score < 0:
        for c in t["frames"].values():
            if c["dense"] is not None: c["dense"] = c["dense"][::-1].copy()
        wh, wt = wt, wh
    L_clean = [t["frames"][k]["length"] for k in ks if not t["frames"][k]["touches_border"] and not t["frames"][k]["flags"]]
    t["L_ref"] = float(np.median(L_clean)) if len(L_clean) >= 10 else L_MED
    for c in t["frames"].values():                       # length outliers relative to THIS worm's body length
        if not c["touches_border"] and not np.isnan(c["length"]) and abs(c["length"]-t["L_ref"])/t["L_ref"] > 0.2:
            c["flags"].append("length_outlier")
    t["orient"] = dict(motion_score=float(score), head_width=wh, tail_width=wt)

# ---------------------------------------------------------------- arc-length labels
def edge_dist(p): return min(p[0], p[1], W-1-p[0], H-1-p[1])
def arc_points(d, touches, L_REF, n=N_DENSE):
    s_arc = np.r_[0, np.cumsum(np.hypot(*np.diff(d, axis=0).T))]; Lvis = s_arc[-1]
    e0, e1 = (at_edge(d[0]), at_edge(d[-1])) if touches else (False, False)
    if touches and not (e0 or e1) and Lvis < 0.92*L_REF:
        if edge_dist(d[0]) < edge_dist(d[-1]): e0 = True
        else: e1 = True
    targets = np.linspace(0, L_REF, n)
    out = np.full((n, 2), np.nan)
    if not e0 and not e1:
        tt = np.linspace(0, Lvis, n)
        return np.c_[np.interp(tt, s_arc, d[:,0]), np.interp(tt, s_arc, d[:,1])], "full"
    if e0 and e1: return out, "both_ends_cut"
    if Lvis > L_REF*1.05: return out, "visible_longer_than_body"
    if e1:
        ok = targets <= Lvis
        out[ok] = np.c_[np.interp(targets[ok], s_arc, d[:,0]), np.interp(targets[ok], s_arc, d[:,1])]
        return out, "tail_cut"
    s_from_tail = L_REF - targets; ok = s_from_tail <= Lvis
    s_rev = s_arc[-1] - s_arc[::-1]; drev = d[::-1]
    out[ok] = np.c_[np.interp(s_from_tail[ok], s_rev, drev[:,0]), np.interp(s_from_tail[ok], s_rev, drev[:,1])]
    return out, "head_cut"

names = ["head", "q1", "mid", "q3", "tail"]
SUB5 = np.linspace(0, N_DENSE-1, N_PTS).round().astype(int)
labels49 = np.full((N, K, N_DENSE, 2), np.nan); status = np.full((N, K), "", dtype=object)
lik = np.zeros((N, K, N_PTS)); curv = np.full((N, K, N_DENSE), np.nan)
for t in tracks:
    j = t["id"]
    for k, c in t["frames"].items():
        d = c["dense"]
        if d is None:
            status[k, j] = "no_midline"; continue
        p49, s = arc_points(d, c["touches_border"], t["L_ref"])
        labels49[k, j] = p49; status[k, j] = s; c["label_status"] = s
        other = [f for f in c["flags"] if f != "touches_border"]
        lik[k, j] = np.where(np.isnan(p49[SUB5, 0]), 0.0, 0.5 if other else 1.0)
        x = gaussian_filter1d(d[:,0], 1.5); y = gaussian_filter1d(d[:,1], 1.5)
        dx, dy = np.gradient(x), np.gradient(y); ddx, ddy = np.gradient(dx), np.gradient(dy)
        curv[k, j] = (dx*ddy - dy*ddx) / np.power(dx*dx + dy*dy, 1.5) * c["length"]

# ---------------------------------------------------------------- DLC csv
stem = os.path.splitext(os.path.basename(VIDEO))[0]
if K == 1:
    cols = pd.MultiIndex.from_product([["auto"], names, ["x", "y", "likelihood"]], names=["scorer", "bodyparts", "coords"])
    data = np.concatenate([np.c_[labels49[:, 0, SUB5[p]], lik[:, 0, p]] for p in range(N_PTS)], axis=1)
    np.save(f"{OUT}/labels49.npy", labels49[:, 0].astype(np.float32)); np.save(f"{OUT}/curvature.npy", curv[:, 0].astype(np.float32))
else:
    indiv = [f"worm{j+1}" for j in range(K)]
    cols = pd.MultiIndex.from_product([["auto"], indiv, names, ["x", "y", "likelihood"]],
                                      names=["scorer", "individuals", "bodyparts", "coords"])
    data = np.concatenate([np.c_[labels49[:, j, SUB5[p]], lik[:, j, p]] for j in range(K) for p in range(N_PTS)], axis=1)
    np.save(f"{OUT}/labels49.npy", labels49.astype(np.float32)); np.save(f"{OUT}/curvature.npy", curv.astype(np.float32))
df = pd.DataFrame(data, columns=cols); df.index.name = "frame"
df.to_csv(f"{OUT}/points5_dlc.csv")
df2 = df.copy(); df2.index = [f"labeled-data/{stem}/img{k:04d}.png" for k in range(N)]
df2.to_csv(f"{OUT}/CollectedData_auto.csv")

# ---------------------------------------------------------------- QC
rows = []
for k, comps in enumerate(frames):
    fl = set()
    if not comps: fl.add("no_worm")
    for c in comps:
        if "small_fragment" in c["flags"]: fl.add("small_fragment"); continue   # debris: its other flags are irrelevant
        fl.update(c["flags"])
        if c["touches_border"]: fl.add("touches_border")
    n_worms = int(sum(1 for s in status[k] if s and s != "no_midline" and s not in ("both_ends_cut", "visible_longer_than_body")))
    big = comps[0] if comps else None
    rows.append(dict(frame=k, flags=";".join(sorted(fl)), n_comp=len(comps), n_worms=n_worms,
                     area=big["area"] if big else 0, length_px=big["length"] if big else np.nan,
                     touches_border=any(c["touches_border"] for c in comps),
                     label_status=";".join(s if s else "-" for s in status[k]) if K > 1 else (status[k, 0] or "no_worm")))
qc = pd.DataFrame(rows); qc.to_csv(f"{OUT}/qc_frames.csv", index=False)
flag_names = ["no_worm","touches_border","cluster","from_cluster","split","skeleton_branches","loop_or_empty","length_outlier","spline_fail","small_fragment"]
track_info = [dict(id=f"worm{t['id']+1}", n_frames=len(t["frames"]), first_frame=int(min(t["frames"])), last_frame=int(max(t["frames"])),
                   body_length_px=t["L_ref"], **t["orient"],
                   label_status_counts={s_: int((status[:, t["id"]] == s_).sum()) for s_ in np.unique(status[:, t["id"]]) if s_})
              for t in tracks]
json.dump(track_info, open(f"{OUT}/tracks.json", "w"), indent=2)
summary = dict(n_frames=N, fps=fps, width=W, height=H, n_tracks=len(tracks), median_length_px=L_MED, median_area_px=A_MED, p90_area_px=A_BIG,
               global_threshold=float(THR_REL),
               frames_with_worm=int((qc["n_comp"] > 0).sum()), frames_labelled=int((qc["n_worms"] > 0).sum()),
               frames_one_worm_labelled=int((qc["n_worms"] == 1).sum()), frames_two_plus_worms_labelled=int((qc["n_worms"] >= 2).sum()),
               clean=int((qc["flags"] == "").sum()),
               flag_counts={f: int(qc["flags"].str.contains(f).sum()) for f in flag_names},
               tracks=track_info, note="index 0 / 'head' = end leading the centroid motion of that track")
json.dump(summary, open(f"{OUT}/summary.json","w"), indent=2, ensure_ascii=False)
print(json.dumps({k_: v for k_, v in summary.items() if k_ != "tracks"}, indent=2, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- overlay video
cap = cv2.VideoCapture(VIDEO)
OW, OH = int(round(W*OVERLAY_SCALE)), int(round(H*OVERLAY_SCALE))
vw = cv2.VideoWriter(f"{OUT}/overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))
colors = [(0,0,255),(0,165,255),(0,255,255),(255,200,0),(255,0,200)]  # head red ... tail magenta
line_col = [(0,255,0),(255,128,0),(0,200,255),(200,0,255)]
k = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    for c in frames[k]:
        d = c["dense"]
        if d is None: continue
        j = c.get("track", None)
        if "cluster" in c["flags"] or "split" in c["flags"] or j is None:
            cv2.polylines(fr, [d.astype(np.int32).reshape(-1,1,2)], False, (0,0,255), 2, cv2.LINE_AA); continue
        cv2.polylines(fr, [d.astype(np.int32).reshape(-1,1,2)], False, line_col[j % 4], 2, cv2.LINE_AA)
        for p in range(N_PTS):
            x, y = labels49[k, j, SUB5[p]]
            if np.isnan(x): continue
            cv2.circle(fr, (int(x),int(y)), 9, colors[p], -1, cv2.LINE_AA)
            cv2.circle(fr, (int(x),int(y)), 9, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(fr, f"H{j+1}", (int(d[0,0])+12, int(d[0,1])-12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
    r = rows[k]; fl = [f for f in r["flags"].split(";") if f and f != "touches_border"]
    txt = f"frame {k}  worms:{r['n_worms']}  " + ("OK" if not fl else ",".join(fl)) + "  " + r["label_status"]
    cv2.putText(fr, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,160,0) if not fl else (0,0,255), 3, cv2.LINE_AA)
    if OVERLAY_SCALE != 1.0: fr = cv2.resize(fr, (OW, OH), interpolation=cv2.INTER_AREA)
    vw.write(fr); k += 1
vw.release(); cap.release()
print("overlay written", flush=True)
