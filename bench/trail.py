"""Midline of a single worm whose mask may self-overlap (coil / ring / omega): the longest TRAIL of the
skeleton graph (each branch used at most once, junctions may be revisited) with a turn-angle limit,
preferring lengths near L_ref. Plain longest_path() breaks on loops; this walks around them."""
import numpy as np, collections
from skimage.morphology import skeletonize
from skan import Skeleton

def _branches(sk):
    S = Skeleton(sk.astype(np.uint8)); br = []; ends = []
    for i in range(S.n_paths):
        co = S.path_coordinates(i)[:, ::-1].astype(float)        # (x,y)
        keep = np.r_[True, np.any(np.diff(co, axis=0) != 0, axis=1)]; co = co[keep]
        if len(co) < 2: continue
        br.append(dict(co=co, L=float(np.sum(np.hypot(*np.diff(co, axis=0).T))))); ends.append((co[0], co[-1]))
    # skan ends branches on different pixels of the same junction cluster -> merge end points closer than 3 px
    pts = [p for e in ends for p in e]; cid = list(range(len(pts)))
    def find(i):
        while cid[i] != i: cid[i] = cid[cid[i]]; i = cid[i]
        return i
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.hypot(*(pts[i]-pts[j])) <= 5.0: cid[find(i)] = find(j)
    for k, e in enumerate(br): e["a"], e["b"] = find(2*k), find(2*k+1)
    return br

def _dir(co, at_end, n=6):
    seg = co[-n:] if at_end else co[:n]
    d = seg[-1] - seg[0]; return d/ (np.linalg.norm(d) + 1e-9)

def longest_trail(mask, L_ref=None, max_turn=80, max_states=20000, spur=None):
    sk = skeletonize(mask)
    if spur is None:
        import cv2; dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5); spur = 3.0*float(np.median(dist[sk])) if sk.any() else 8
    from methods import prune_spurs
    if spur > 0: sk = prune_spurs(sk, spur)
    if sk.sum() < 5: return None
    try: br = _branches(sk)
    except Exception: return None
    if not br: return None
    if len(br) == 1: return br[0]["co"]
    adj = collections.defaultdict(list)
    for i, e in enumerate(br): adj[e["a"]].append(i); adj[e["b"]].append(i)
    nodes = list(adj); best = [None, -1.0]; cnt = [0]
    def score(L): return L if L_ref is None else L - 0.5*abs(L - L_ref)
    def dfs(node, used, coords, L, indir):
        cnt[0] += 1
        if cnt[0] > max_states: return
        s = score(L)
        if s > best[1]: best[0], best[1] = coords, s
        for i in adj[node]:
            if i in used: continue
            e = br[i]; fwd = e["a"] == node
            co = e["co"] if fwd else e["co"][::-1]
            if e["a"] == e["b"] and not fwd: continue
            if indir is not None:
                d0 = _dir(co, False)
                if np.degrees(np.arccos(np.clip(np.dot(indir, d0), -1, 1))) > max_turn: continue
            nxt = e["b"] if fwd else e["a"]
            dfs(nxt, used | {i}, coords + [co], L + e["L"], _dir(co, True))
    starts = [n for n in nodes if len(adj[n]) == 1] or nodes
    for n in starts: dfs(n, frozenset(), [], 0.0, None)
    if best[0] is None: return None
    return np.vstack(best[0])
