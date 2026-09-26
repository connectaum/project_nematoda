"""Midline (49-point centreline) of a worm instance mask.

`mask_midline(mask, L_ref)` = longest TRAIL of the skeleton graph (each branch used at most once, junctions may
be revisited, turn limit 80 deg, lengths near L_ref preferred). Unlike a plain longest path it walks around
self-loops (coiled / omega / ring worms). Pure numpy + scikit-image + skan, no torch.
"""
import collections
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton

N_DENSE = 49


def crop(mask, pad=2):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + 1 + pad, max(xs.min() - pad, 0), xs.max() + 1 + pad
    return mask[y0:y1, x0:x1], (x0, y0)


def arclen(p):
    return float(np.sum(np.hypot(*np.diff(np.asarray(p, float), axis=0).T)))


def resample_polyline(path, n=N_DENSE):
    path = np.asarray(path, float)
    if len(path) < 2:
        return None
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    if d[-1] == 0:
        return None
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])]


def _branches(sk):
    S = Skeleton(sk.astype(np.uint8))
    br, ends = [], []
    for i in range(S.n_paths):
        co = S.path_coordinates(i)[:, ::-1].astype(float)          # (x, y)
        keep = np.r_[True, np.any(np.diff(co, axis=0) != 0, axis=1)]
        co = co[keep]
        if len(co) < 2:
            continue
        br.append(dict(co=co, L=float(np.sum(np.hypot(*np.diff(co, axis=0).T)))))
        ends.append((co[0], co[-1]))
    # skan ends branches on different pixels of the same junction cluster -> merge end points closer than 5 px
    pts = [p for e in ends for p in e]
    cid = list(range(len(pts)))

    def find(i):
        while cid[i] != i:
            cid[i] = cid[cid[i]]
            i = cid[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(*(pts[i] - pts[j])) <= 5.0:
                cid[find(i)] = find(j)
    for k, e in enumerate(br):
        e["a"], e["b"] = find(2 * k), find(2 * k + 1)
    return br


def _dir(co, at_end, n=6):
    seg = co[-n:] if at_end else co[:n]
    d = seg[-1] - seg[0]
    return d / (np.linalg.norm(d) + 1e-9)


def longest_trail(mask, L_ref=None, max_turn=80, max_states=20000):
    """Ordered (n,2) (x,y) polyline through the skeleton of `mask`, or None."""
    sk = skeletonize(mask)
    if sk.sum() < 5:
        return None
    try:
        br = _branches(sk)
    except Exception:
        return None
    if not br:
        return None
    if len(br) == 1:
        return br[0]["co"]
    adj = collections.defaultdict(list)
    for i, e in enumerate(br):
        adj[e["a"]].append(i)
        adj[e["b"]].append(i)
    best = [None, -1.0]
    cnt = [0]

    def score(L):
        return L if L_ref is None else L - 0.5 * abs(L - L_ref)

    def dfs(node, used, coords, L, indir):
        cnt[0] += 1
        if cnt[0] > max_states:
            return
        s = score(L)
        if s > best[1]:
            best[0], best[1] = coords, s
        for i in adj[node]:
            if i in used:
                continue
            e = br[i]
            fwd = e["a"] == node
            if e["a"] == e["b"] and not fwd:
                continue
            co = e["co"] if fwd else e["co"][::-1]
            if indir is not None:
                d0 = _dir(co, False)
                if np.degrees(np.arccos(np.clip(np.dot(indir, d0), -1, 1))) > max_turn:
                    continue
            nxt = e["b"] if fwd else e["a"]
            dfs(nxt, used | {i}, coords + [co], L + e["L"], _dir(co, True))

    starts = [n for n in adj if len(adj[n]) == 1] or list(adj)
    for n in starts:
        dfs(n, frozenset(), [], 0.0, None)
    if best[0] is None:
        return None
    return np.vstack(best[0])


def mask_midline(mask, L_ref=None, min_area=0):
    """49-point midline (x,y) of one instance mask in full-frame coordinates, or None."""
    if mask is None or mask.sum() <= max(min_area, 4):
        return None
    sub, off = crop(mask)
    p = longest_trail(sub, L_ref=L_ref)
    if p is None or len(p) < 2:
        return None
    return resample_polyline(p + np.array(off, float))


def orient_like(p, ref):
    """Flip p (49,2) so that it matches the head/tail orientation of ref (49,2)."""
    if ref is None or np.isnan(ref).any():
        return p
    e1 = np.mean(np.hypot(*(p - ref).T))
    e2 = np.mean(np.hypot(*(p[::-1] - ref).T))
    return p if e1 <= e2 else p[::-1]
