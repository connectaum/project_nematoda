"""Split a 'cluster' component (two worms touching / crossing) into individual midlines via the skeleton graph.

Method (bench M1, 2026-09-05): skeletonize -> prune short spurs -> skan branch graph -> enumerate every simple path
from a free end, measuring the turn at each junction *through* short junction-junction edges (the shared segment of an
X crossing has an intermediate direction and must not be judged on its own) -> a path may also terminate at a junction
(worm end buried in the other body) -> rank by (max turn + length prior + termination penalty) -> greedy selection where a
new path may share at most 40 % of its length with accepted ones.

    split_cluster(mask, width, L_ref) -> list of (n,2) float arrays (x, y) in mask coordinates

Benchmark: solves ~40 % of X crossings and ~20 % of tip contacts fully (both worms < 5 % L error); cannot split worms
lying side by side or two-worm rings. Requires `skan`; without it split_cluster returns [].
"""
import collections, numpy as np
from skimage.morphology import skeletonize
try:
    from skan import Skeleton, summarize
    HAVE_SKAN = True
except Exception:
    HAVE_SKAN = False

NBR = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _arclen(p): return float(np.sum(np.hypot(*np.diff(np.asarray(p, float), axis=0).T)))

def _summ(sk):
    S = Skeleton(sk.astype(np.uint8))
    try: df = summarize(S, separator="_")
    except TypeError: df = summarize(S)
    return S, df

def prune_spurs(sk, spur_len, rounds=3):
    """Remove junction-end branches shorter than spur_len (repeat), re-skeletonize."""
    sk = sk.copy()
    for _ in range(rounds):
        try: S, df = _summ(sk)
        except Exception: return sk
        short = [i for i in range(S.n_paths) if int(df.loc[i, "branch_type"]) == 1 and float(df.loc[i, "branch_distance"]) < spur_len]
        if not short: return sk
        for i in short:
            co = S.path_coordinates(i).astype(int)[1:-1]
            sk[co[:, 0], co[:, 1]] = False
        sk = skeletonize(sk)
    return sk

def graph_paths(mask, width, max_turn=75, max_paths=4000):
    """Candidate worm paths through the pruned skeleton graph of one component."""
    sk = prune_spurs(skeletonize(mask), 1.5*width)
    if sk.sum() < 10: return []
    try: S, df = _summ(sk)
    except Exception: return []
    edges = [dict(id=i, co=S.path_coordinates(i), n0=int(df.loc[i, "node_id_src"]), n1=int(df.loc[i, "node_id_dst"]),
                  L=float(df.loc[i, "branch_distance"])) for i in range(S.n_paths)]
    adj = collections.defaultdict(list)
    for e in edges: adj[e["n0"]].append(e); adj[e["n1"]].append(e)
    ends = [n for n, es in adj.items() if len(es) == 1]
    k = max(3, int(width))
    def out_dir(e, node):
        co = e["co"] if e["n0"] == node else e["co"][::-1]
        v = co[min(len(co)-1, k)] - co[0]; return v / (np.linalg.norm(v) + 1e-9)
    def other(e, node): return e["n1"] if e["n0"] == node else e["n0"]
    def turn(a, b): return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    paths = []; short = 2.5*width
    def dfs(node, e_prev, used, worst, Lsum, d_in):
        if len(paths) > max_paths: return
        went = False
        for e in adj[node]:
            if e["id"] in used: continue
            nxt = other(e, node); is_short = e["L"] < short and len(adj[nxt]) > 1
            if d_in is not None and not is_short:
                t = turn(d_in, out_dir(e, node))
                if t > max_turn: continue
                w2 = max(worst, t)
            else: w2 = worst
            went = True; used2 = used + (e["id"],)
            if len(adj[nxt]) == 1: paths.append((used2, w2, Lsum + e["L"], 0))
            else:
                if not is_short: paths.append((used2, w2, Lsum + e["L"], 1))
                dfs(nxt, e, used2, w2, Lsum + e["L"], d_in if is_short else -out_dir(e, nxt))
        if not went and e_prev is not None: paths.append((used, worst, Lsum, 1))
    for n in ends: dfs(n, None, (), 0.0, 0.0, None)
    seen = set(); out = []
    for ids, worst, L, term in paths:
        key = tuple(sorted(ids))
        if key in seen: continue
        seen.add(key)
        first = edges[ids[0]]; node = first["n0"] if len(adj[first["n0"]]) == 1 else first["n1"]; co = []
        for i in ids:
            e = edges[i]; c = e["co"] if e["n0"] == node else e["co"][::-1]; co.append(c); node = other(e, node)
        out.append(dict(co=np.vstack(co)[:, ::-1].astype(float), L=L, turn=worst, edges=key, elen={i: edges[i]["L"] for i in key}, term=term))
    return out

def select_paths(cands, L_ref, share_max=0.4):
    cands = sorted(cands, key=lambda p: p["turn"] + 100*abs(np.log(p["L"]/L_ref)) + 15*p["term"])
    acc = []; used = collections.Counter()
    for p in cands:
        sh = sum(p["elen"][e] for e in p["edges"] if used[e] > 0)
        if acc and sh > share_max*p["L"]: continue
        acc.append(p)
        for e in p["edges"]: used[e] += 1
    return acc

def split_cluster(mask, width, L_ref, L_range=(0.6, 1.3), min_worms=2):
    """Midlines of the individual worms in a cluster mask, or [] if the split is not convincing
    (fewer than min_worms plausible-length paths)."""
    if not HAVE_SKAN: return []
    cands = [p for p in graph_paths(mask, width) if L_range[0]*L_ref <= p["L"] <= L_range[1]*L_ref]
    if not cands: return []
    sel = select_paths(cands, L_ref)
    if len(sel) < min_worms: return []
    out = []
    for p in sel:                                   # drop repeated consecutive pixels (junction joins) — splines need strictly increasing arc length
        co = p["co"]; keep = np.r_[True, np.any(np.diff(co, axis=0) != 0, axis=1)]
        out.append(co[keep])
    return out
