"""Midline extraction methods for the crossing benchmark.  Each: f(img_bgr, ctx) -> list of (n,2) midlines (x,y)."""
import numpy as np, cv2, collections
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skan import Skeleton, summarize
from common import components, resample_polyline, arclen, N_DENSE

# ------------------------------------------------------------ helpers
NBR = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def longest_path(sk):
    """Longest endpoint-to-endpoint path of a skeleton (pixel set) -> (n,2) (x,y) or None."""
    pts = set(zip(*np.nonzero(sk)))
    if not pts: return None
    def nb(p): return [(p[0]+dy, p[1]+dx) for dy,dx in NBR if (p[0]+dy, p[1]+dx) in pts]
    ends = [p for p in pts if len(nb(p)) == 1]
    if len(ends) < 2: return None
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
    return np.array(path, float)[:, ::-1]

def comp_width(mask, sk):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return 2 * float(np.median(dist[sk])) if sk.any() else 10.0

def crop(mask):
    ys, xs = np.nonzero(mask); y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    return mask[max(y0-2,0):y1+2, max(x0-2,0):x1+2], (max(x0-2,0), max(y0-2,0))

# ------------------------------------------------------------ M0: current pipeline behaviour
def m0_baseline(img, ctx, **kw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); out = []
    for c in components(ctx.segment(gray)):
        sub, off = crop(c["mask"]); sk = skeletonize(sub)
        path = longest_path(sk)
        if path is None: continue
        spur = sk.sum() - len(path); branched = spur > 0.15*len(path)
        if branched:
            pts = set(zip(*np.nonzero(sk))); n_end = sum(1 for p in pts if sum((p[0]+dy, p[1]+dx) in pts for dy,dx in NBR) == 1)
        else: n_end = 2
        if c["area"] > 1.5*ctx.A_big or (branched and c["area"] > 1.2*ctx.A_big and n_end >= 4): continue   # 'cluster' -> no labels
        out.append(resample_polyline(path + off))
    return out

# ------------------------------------------------------------ M1: skeleton graph, enumerate end-to-end paths, rank by turning
def prune_spurs(sk, spur_len, rounds=3):
    """Remove junction-end branches shorter than spur_len (repeat), re-skeletonize."""
    sk = sk.copy()
    for _ in range(rounds):
        try: S = Skeleton(sk.astype(np.uint8)); df = summarize(S, separator="_")
        except Exception: return sk
        short = [i for i in range(S.n_paths) if int(df.loc[i, "branch_type"]) == 1 and float(df.loc[i, "branch_distance"]) < spur_len]
        if not short: return sk
        for i in short:
            co = S.path_coordinates(i).astype(int)[1:-1]     # keep the two node pixels
            sk[co[:, 0], co[:, 1]] = False
        sk = skeletonize(sk)
    return sk

def turn_angle(a_in, b_out):
    return float(np.degrees(np.arccos(np.clip(np.dot(a_in, b_out), -1, 1))))

def graph_paths(sub, width, L_range=None, max_turn=75, max_paths=4000):
    """All endpoint-to-endpoint simple paths through the pruned skeleton graph, each scored by its worst junction turn."""
    sk = prune_spurs(skeletonize(sub), 1.5*width)
    if sk.sum() < 10: return [], sk
    try: S = Skeleton(sk.astype(np.uint8)); df = summarize(S, separator="_")
    except Exception:
        p = longest_path(sk); return ([dict(co=p, L=arclen(p), turn=0.0, edges=(0,), elen={0: arclen(p)}, term=0)] if p is not None else []), sk
    edges = []
    for i in range(S.n_paths):
        co = S.path_coordinates(i)
        edges.append(dict(id=i, co=co, n0=int(df.loc[i, "node_id_src"]), n1=int(df.loc[i, "node_id_dst"]), L=float(df.loc[i, "branch_distance"])))
    adj = collections.defaultdict(list)
    for e in edges: adj[e["n0"]].append(e); adj[e["n1"]].append(e)
    ends = [n for n, es in adj.items() if len(es) == 1]
    k = max(3, int(width))
    def out_dir(e, node):
        co = e["co"] if e["n0"] == node else e["co"][::-1]
        v = co[min(len(co)-1, k)] - co[0]; return v / (np.linalg.norm(v) + 1e-9)
    def other(e, node): return e["n1"] if e["n0"] == node else e["n0"]
    paths = []
    short = 2.5*width           # junction-junction edges shorter than this are the shared segment of a crossing: look through them
    def dfs(node, e_prev, used, worst, Lsum, d_in):
        """d_in = travel direction of the last LONG edge arriving here (None at a free end)."""
        if len(paths) > max_paths: return
        went = False
        for e in adj[node]:
            if e["id"] in used: continue
            nxt = other(e, node)
            is_short = e["L"] < short and len(adj[nxt]) > 1
            if d_in is not None and not is_short:
                t = turn_angle(d_in, out_dir(e, node))
                if t > max_turn: continue
                w2 = max(worst, t)
            else: w2 = worst
            went = True
            used2 = used + (e["id"],)
            if len(adj[nxt]) == 1:
                paths.append((used2, w2, Lsum + e["L"], 0))
            else:
                if not is_short: paths.append((used2, w2, Lsum + e["L"], 1))   # may terminate here (end buried in another worm)
                d_next = d_in if is_short else -out_dir(e, nxt)
                dfs(nxt, e, used2, w2, Lsum + e["L"], d_next)
        if not went and e_prev is not None:
            paths.append((used, worst, Lsum, 1))
    for n in ends: dfs(n, None, (), 0.0, 0.0, None)
    # materialise unique paths (each found from both ends)
    seen = set(); out = []
    for ids, worst, L, term in paths:
        key = tuple(sorted(ids))
        if key in seen: continue
        seen.add(key)
        if L_range and not (L_range[0] <= L <= L_range[1]): continue
        # ordered coordinates
        co = []; node = None
        first = edges[ids[0]]
        node = first["n0"] if len(adj[first["n0"]]) == 1 else first["n1"]
        for i in ids:
            e = edges[i]; c = e["co"] if e["n0"] == node else e["co"][::-1]; co.append(c); node = other(e, node)
        co = np.vstack(co)[:, ::-1].astype(float)
        out.append(dict(co=co, L=L, turn=worst, edges=key, elen={i: edges[i]["L"] for i in key}, term=term))
    return out, sk

def select_paths(cands, L_med, share_max=0.4):
    """Greedy: best (low turn, plausible length) first; a new path may share at most share_max of its LENGTH with accepted ones."""
    cands = sorted(cands, key=lambda p: (p["turn"] + 100*abs(np.log(p["L"]/L_med)) + 15*p.get("term", 0)))
    acc = []; used = collections.Counter()
    for p in cands:
        sh = sum(p["elen"][e] for e in p["edges"] if used[e] > 0)
        if acc and sh > share_max*p["L"]: continue
        acc.append(p)
        for e in p["edges"]: used[e] += 1
    return acc

def m1_graph(img, ctx, len_filter=True, **kw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); out = []
    for c in components(ctx.segment(gray)):
        sub, off = crop(c["mask"]); sk = skeletonize(sub); w = comp_width(sub, sk)
        rng = (0.45*ctx.L_med, 1.35*ctx.L_med) if len_filter else None
        cands, _ = graph_paths(sub, w, rng)
        if not cands:                              # nothing plausible: fall back to the longest path (single worm / partial)
            p = longest_path(sk)
            if p is not None and (not len_filter or 0.45*ctx.L_med <= arclen(p) <= 1.35*ctx.L_med): out.append(resample_polyline(p + off))
            continue
        for p in select_paths(cands, ctx.L_med):
            r = resample_polyline(p["co"] + off)
            if r is not None: out.append(r)
    return out

def graph_trace(sub, width, ctx=None):
    cands, _ = graph_paths(sub, width, None)
    return [p["co"] for p in select_paths(cands, ctx.L_med if ctx else np.median([p["L"] for p in cands]))] if cands else []

# ------------------------------------------------------------ M2: erosion markers + watershed (for side-by-side worms)
def erosion_split(mask, width, ctx):
    """Erode until the blob splits into >=2 sizeable pieces, then watershed the original mask from those markers."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    for r in np.linspace(0.25*width, 0.75*width, 6):
        er = dist > r
        n, lab = cv2.connectedComponents(er.astype(np.uint8))
        sizes = [(lab == i).sum() for i in range(1, n)]
        big = [i+1 for i, s in enumerate(sizes) if s > 0.05*mask.sum()]
        if len(big) >= 2:
            markers = np.zeros_like(lab);
            for j, i in enumerate(big): markers[lab == i] = j+1
            ws = watershed(-dist, markers, mask=mask)
            return [ws == j+1 for j in range(len(big))]
    return [mask]

def m2_watershed(img, ctx, **kw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); out = []
    for c in components(ctx.segment(gray)):
        sub, off = crop(c["mask"]); sk = skeletonize(sub); w = comp_width(sub, sk)
        pieces = erosion_split(sub, w, ctx) if c["area"] > 1.3*ctx.A_med else [sub]
        for pm in pieces:
            p = longest_path(skeletonize(pm))
            if p is None: continue
            L = arclen(p)
            if not (0.45*ctx.L_med <= L <= 1.35*ctx.L_med): continue
            out.append(resample_polyline(p + off))
    return out

# ------------------------------------------------------------ M3: combo — graph first; over-long survivors go to watershed
def m3_combo(img, ctx, **kw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); out = []
    for c in components(ctx.segment(gray)):
        sub, off = crop(c["mask"]); sk = skeletonize(sub); w = comp_width(sub, sk)
        paths = graph_trace(sub, w, ctx)
        good = [p for p in paths if 0.45*ctx.L_med <= arclen(p) <= 1.35*ctx.L_med]
        too_long = [p for p in paths if arclen(p) > 1.35*ctx.L_med]
        if too_long or (not good and c["area"] > 1.3*ctx.A_med):
            for pm in erosion_split(sub, w, ctx):
                for p in graph_trace(pm, w, ctx):
                    if 0.45*ctx.L_med <= arclen(p) <= 1.35*ctx.L_med: good.append(p)
        for p in good:
            r = resample_polyline(p + off)
            if r is not None: out.append(r)
    # dedupe near-identical midlines from the two stages
    final = []
    for p in good_dedupe(out): final.append(p)
    return final

def good_dedupe(lines, tol=6.0):
    keep = []
    for p in lines:
        if any(min(np.mean(np.hypot(*(p-q).T)), np.mean(np.hypot(*(p-q[::-1]).T))) < tol for q in keep): continue
        keep.append(p)
    return keep

METHODS = {"M0_pipeline": m0_baseline, "M1_graph": m1_graph, "M2_watershed": m2_watershed, "M3_combo": m3_combo}
