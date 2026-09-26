"""Shared pieces for the crossing benchmark: segmentation identical to worm_pipeline.py,
frame index, ground-truth midlines from auto_labels, instance matching + metrics."""
import os, json, csv, glob
import numpy as np, cv2
from skimage.morphology import remove_small_objects, disk
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "bench", "frames")
DISK7 = disk(7).astype(np.uint8)
N_DENSE = 49

def videos():
    return sorted(os.path.basename(d) for d in glob.glob(f"{FRAMES}/*") if os.path.isdir(d))

def load_index(v):
    return list(csv.DictReader(open(f"{FRAMES}/{v}/index.csv")))

def load_frame(v, f):
    return cv2.imread(f"{FRAMES}/{v}/{int(f):06d}.jpg")

class VideoCtx:
    """Per-video constants: threshold from summary.json, background reference from the bench frames."""
    def __init__(self, v):
        self.v = v
        s = json.load(open(f"{ROOT}/auto_labels/{v}/summary.json"))
        self.thr = s["global_threshold"]; self.L_med = s["median_length_px"]; self.A_med = s["median_area_px"]
        self.A_big = s.get("p90_area_px", 1.3 * self.A_med); self.W, self.H = s["width"], s["height"]
        self.tracks = json.load(open(f"{ROOT}/auto_labels/{v}/tracks.json"))
        lab = np.load(f"{ROOT}/auto_labels/{v}/labels49.npy")
        self.labels = lab if lab.ndim == 4 else lab[:, None]          # (N,K,49,2)
        bgs = []
        for r in load_index(v):
            if r["kind"] == "clean":
                g = cv2.GaussianBlur(cv2.cvtColor(load_frame(v, r["frame"]), cv2.COLOR_BGR2GRAY), (5, 5), 0)
                bgs.append(np.median(g))
        self.bg_ref = float(np.median(bgs)) if bgs else 200.0

    def segment(self, gray):
        g = cv2.GaussianBlur(gray, (5, 5), 0); bg = float(np.median(g))
        m = (g < self.thr + (bg - self.bg_ref)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, DISK7).astype(bool)
        m = ndi.binary_fill_holes(m)
        return remove_small_objects(m, max_size=800)

    def gt_midlines(self, f):
        """Full 49-pt midlines present in frame f (list of (K_index, pts))."""
        out = []
        for k in range(self.labels.shape[1]):
            p = self.labels[int(f), k]
            if not np.isnan(p).any(): out.append((k, p.copy()))
        return out

def components(mask, a_min=1500):
    lab = label(mask); comps = []
    for r in regionprops(lab):
        if r.area < a_min: continue
        comps.append(dict(area=int(r.area), bbox=r.bbox, mask=(lab == r.label), centroid=(r.centroid[1], r.centroid[0])))
    return comps

def resample_polyline(path, n=N_DENSE):
    path = np.asarray(path, float)
    if len(path) < 2: return None
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    if d[-1] == 0: return None
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])]

def arclen(p): return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))

# ---------------------------------------------------------------- metrics
def midline_error(pred, gt):
    """Mean point distance after resampling both to 49 pts, best of the two orientations (px)."""
    a = resample_polyline(pred); b = resample_polyline(gt)
    if a is None or b is None: return np.inf
    e1 = np.mean(np.hypot(*(a - b).T)); e2 = np.mean(np.hypot(*(a - b[::-1]).T))
    return float(min(e1, e2))

def match(preds, gts):
    """Hungarian match on midline error. Returns list of (gt_i, pred_j, err) plus unmatched counts."""
    if not preds or not gts: return [], len(gts), len(preds)
    C = np.array([[midline_error(p, g) for p in preds] for g in gts])
    ri, ci = linear_sum_assignment(C)
    pairs = [(int(i), int(j), float(C[i, j])) for i, j in zip(ri, ci)]
    return pairs, len(gts) - len(pairs), len(preds) - len(pairs)

def score_frame(preds, gts, L, WH=None, border=12):
    """Per-frame summary: recovered = matched with err < 0.05 L; ok = err < 0.10 L.
    Unmatched predictions touching the frame border are not counted as extras (partially visible worms carry no GT)."""
    pairs, miss, extra = match(preds, gts)
    if WH is not None and extra:
        matched = {j for _, j, _ in pairs}
        W, H = WH
        extra = sum(1 for j, p in enumerate(preds) if j not in matched and not
                    (p[:, 0].min() < border or p[:, 1].min() < border or p[:, 0].max() > W - border or p[:, 1].max() > H - border))
    errs = [e / L for _, _, e in pairs]
    return dict(n_gt=len(gts), n_pred=len(preds), n_match=len(pairs), miss=miss, extra=extra,
                recovered=sum(e < 0.05 for e in errs), ok=sum(e < 0.10 for e in errs),
                all_recovered=int(len(gts) > 0 and miss == 0 and extra == 0 and all(e < 0.05 for e in errs)),
                err_mean=float(np.mean(errs)) if errs else np.nan)

def draw_midlines(img, lines, colors=None, thick=2):
    cols = colors or [(0, 200, 0), (255, 128, 0), (0, 200, 255), (200, 0, 255), (0, 255, 255)]
    out = img.copy()
    for i, p in enumerate(lines):
        p = np.asarray(p)
        if p is None or len(p) < 2 or np.isnan(p).any(): continue
        cv2.polylines(out, [p.astype(np.int32).reshape(-1, 1, 2)], False, cols[i % len(cols)], thick, cv2.LINE_AA)
        cv2.circle(out, tuple(p[0].astype(int)), 6, cols[i % len(cols)], -1)
    return out
