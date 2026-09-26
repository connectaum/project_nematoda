"""Export contact episodes of one processed video for the wormsam2 tool (SAM2 on a workstation).

  python episode_export.py <video.mp4> <auto_labels_dir> <out_zip> [--margin 60] [--min-frames 5]

Contact is decided PER TRACK PAIR, not from blob area (a single big worm must not look like a contact):
a frame is 'contact' when the midlines of two tracks come closer than ~1.5 worm widths, or the frame is flagged
'cluster' (worms merged into one component). Episodes = runs of contact frames (gaps <= 10) of >= min_frames.

For every track taking part in an episode, the SAM2 prompt is its cleanest labelled frame: label_status 'full'
and the midline at least ~3 widths away from every other track — searched first in `margin` frames before the
episode, then after it, then anywhere in the track (SAM2 propagates both ways). The midlines of the other
tracks on that frame are exported too (negative points). Clips cover [earliest prompt-5, latest of end/prompt+5],
prompts farther than --max-ext frames from the episode are dropped.

Writes episodes_<video>.zip = episodes.json + epNN.mp4. Prints the number of episodes (0 -> no zip written).
"""
import argparse
import csv
import json
import os
import sys
import zipfile

import cv2
import numpy as np
from scipy.spatial.distance import cdist

FORMAT = 2


def load_status(qc_path, K):
    rows = list(csv.DictReader(open(qc_path)))
    N = len(rows)
    flags = [r["flags"] for r in rows]
    status = np.full((N, K), "", dtype=object)
    for i, r in enumerate(rows):
        s = r["label_status"]
        parts = s.split(";") if K > 1 else [s]
        for j, p in enumerate(parts[:K]):
            status[i, j] = "" if p in ("-", "no_worm") else p
    return flags, status


def pair_distances(lab):
    """min midline-to-midline distance for every frame and track pair: (N, K, K), NaN where a track is absent."""
    N, K = lab.shape[:2]
    D = np.full((N, K, K), np.nan, np.float32)
    present = ~np.isnan(lab[:, :, 0, 0])
    for f in range(N):
        js = np.where(present[f])[0]
        pts = {j: lab[f, j][~np.isnan(lab[f, j]).any(1)] for j in js}     # labels may hold NaN points (cut ends)
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                if len(pts[js[a]]) and len(pts[js[b]]):
                    d = cdist(pts[js[a]], pts[js[b]]).min()
                    D[f, js[a], js[b]] = D[f, js[b], js[a]] = d
    return D, present


def absorbed_gaps(lab, present, D, d_near, max_gap):
    """Frames where a track vanished while it was next to another track — i.e. it was merged into that blob.
    A gap in track j (<= max_gap frames, labelled on both sides) counts when the last/first label of j around the
    gap lies within d_near of another track's midline."""
    N, K = present.shape
    out = np.zeros(N, bool)
    for j in range(K):
        fr = np.where(present[:, j])[0]
        for f0, f1 in zip(fr[:-1], fr[1:]):
            if 1 < f1 - f0 <= max_gap:
                near = False
                for f in (f0, f1):
                    d = D[f, j]
                    d = d[~np.isnan(d)]
                    near |= bool(len(d) and d.min() < d_near)
                if near:
                    out[f0 + 1:f1] = True
    return out


def runs(mask, gap, min_len):
    eps, cur = [], None
    for i in np.where(mask)[0]:
        if cur is None:
            cur = [i, i]
        elif i - cur[1] > gap:
            eps.append(cur)
            cur = [i, i]
        else:
            cur[1] = i
    if cur:
        eps.append(cur)
    return [(int(a), int(b)) for a, b in eps if b - a + 1 >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("labels")
    ap.add_argument("out_zip")
    ap.add_argument("--margin", type=int, default=60, help="frames before/after an episode to look for a clean prompt first")
    ap.add_argument("--max-ext", type=int, default=600, help="max distance of a prompt frame from the episode")
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--gap", type=int, default=10)
    args = ap.parse_args()
    L = args.labels
    summ = json.load(open(os.path.join(L, "summary.json")))
    lab = np.load(os.path.join(L, "labels49.npy")).astype(np.float32)
    if lab.ndim == 3:
        lab = lab[:, None]
    N, K = lab.shape[:2]
    tracks = json.load(open(os.path.join(L, "tracks.json")))
    ids = [t.get("id", f"worm{j + 1}") for j, t in enumerate(tracks)] + [f"worm{j + 1}" for j in range(len(tracks), K)]
    flags, status = load_status(os.path.join(L, "qc_frames.csv"), K)
    A_med, L_med = float(summ["median_area_px"]), float(summ["median_length_px"])
    width = float(np.clip(A_med / max(L_med, 1.0), 4.0, 0.15 * L_med))     # rough body width in px
    D_CONTACT, D_CLEAN = 1.5 * width, 3.0 * width
    D, present = pair_distances(lab)

    close = np.nan_to_num(D, nan=np.inf).min(axis=(1, 2)) < D_CONTACT        # some pair touches
    cluster = np.array(["cluster" in f for f in flags[:N]] + [False] * max(0, N - len(flags)))
    absorbed = absorbed_gaps(lab, present, D, D_CLEAN, max_gap=400)          # track lost while next to another worm
    contact = close | cluster | absorbed
    eps = runs(contact, args.gap, args.min_frames)
    merged = []                                   # episodes closer than 2*margin share their clip -> one episode
    for a, b in eps:
        if merged and a - merged[-1][1] <= 2 * args.margin:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    eps = merged
    if not eps:
        print("episodes: 0")
        return 0

    cap = cv2.VideoCapture(args.video)
    fps = float(summ.get("fps") or cap.get(5) or 20.0)
    W, H = int(summ["width"]), int(summ["height"])
    thr = float(summ.get("global_threshold") or 0)
    exp_area = {}
    for j in range(K):
        Lj = (tracks[j].get("body_length_px") if j < len(tracks) else None) or L_med
        exp_area[j] = float(Lj) * width
    blob_cache = {}

    def blob_ok(f, j):
        """Image check: the dark component under track j's midline must have about ONE worm's area — the labels
        know nothing about worms the pipeline never labelled (border, fragments), the image does."""
        if thr <= 0:
            return True
        key = (f, j)
        if key in blob_cache:
            return blob_cache[key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        res = True
        if ok:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            bw = (g < thr).astype(np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n, cc = cv2.connectedComponents(bw)
            pts = lab[f, j][~np.isnan(lab[f, j]).any(1)].round().astype(int)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < g.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < g.shape[0])]
            labs = cc[pts[:, 1], pts[:, 0]]
            nz = labs[labs > 0]
            if len(nz) == 0:
                res = False
            else:
                lab_id = np.bincount(nz).argmax()
                inside = float((labs == lab_id).mean())          # a chimeric midline crosses background / other blobs
                area = float((cc == lab_id).sum())
                res = inside >= 0.95 and 0.5 * exp_area[j] <= area <= 1.4 * exp_area[j]
        blob_cache[key] = res
        return res

    def min_dist_to_others(f, j):
        d = D[f, j]
        d = d[~np.isnan(d)]
        return float(d.min()) if len(d) else np.inf

    def label_clean(f, j):
        return present[f, j] and status[f, j] == "full" and min_dist_to_others(f, j) >= D_CLEAN

    def find_prompt(j, a, b, max_checks=40):
        """Cleanest labelled frame: label-clean AND image-clean, nearest before the episode, then after, then the
        farthest-from-others within max_ext. At most max_checks frames are read from the video."""
        checks = 0
        for f in list(range(a, max(-1, a - args.margin - 1), -1)) + list(range(b, min(N, b + args.margin + 1))):
            if label_clean(f, j):
                checks += 1
                if blob_ok(f, j):
                    return f
                if checks >= max_checks:
                    return None
        cands = [(min_dist_to_others(f, j), f) for f in range(max(0, a - args.max_ext), min(N, b + args.max_ext + 1))
                 if present[f, j] and status[f, j] == "full" and min_dist_to_others(f, j) >= D_CLEAN]
        for _, f in sorted(cands, reverse=True)[:max_checks - checks]:
            if blob_ok(f, j):
                return f
        return None

    meta = dict(format=FORMAT, video=os.path.splitext(os.path.basename(args.video))[0], fps=fps, width=W, height=H,
                n_frames=int(N), median_length_px=L_med, median_area_px=A_med, width_px=width, n_tracks=int(K), episodes=[])
    tmpdir = args.out_zip + ".tmp"
    os.makedirs(tmpdir, exist_ok=True)
    for e, (a, b) in enumerate(eps, 1):
        lo, hi = max(0, a - args.margin), min(N - 1, b + args.margin)
        # participants: tracks labelled near the episode, or hidden inside a cluster there
        part = [j for j in range(K) if present[lo:hi + 1, j].any()]
        trs = []
        for j in part:
            pf = find_prompt(j, a, b)
            others = []
            if pf is not None:
                others = [dict(track=int(j2), id=ids[j2], midline=lab[pf, j2].round(1).tolist())
                          for j2 in range(K) if j2 != j and present[pf, j2]]
            trs.append(dict(track=int(j), id=ids[j], prompt_frame=pf,
                            midline=lab[pf, j].round(1).tolist() if pf is not None else None,
                            others=others,
                            clean_dist_px=round(min_dist_to_others(pf, j), 1) if pf is not None and np.isfinite(min_dist_to_others(pf, j)) else None))
        prompted = [t for t in trs if t["prompt_frame"] is not None]
        c0 = max(0, min([a] + [t["prompt_frame"] for t in prompted]) - 5)
        c1 = min(N - 1, max([b] + [t["prompt_frame"] for t in prompted]) + 5)
        clip = f"ep{e:02d}.mp4"
        vw = cv2.VideoWriter(os.path.join(tmpdir, clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        cap.set(cv2.CAP_PROP_POS_FRAMES, c0)
        for f in range(c0, c1 + 1):
            ok, im = cap.read()
            if not ok:
                c1 = f - 1
                break
            vw.write(im)
        vw.release()
        unprompted = [t["id"] for t in trs if t["prompt_frame"] is None]
        note = (f"{len(prompted)} of {len(trs)} tracks have a clean prompt"
                + (f"; no clean frame for {', '.join(unprompted)} — click them in wormsam2" if unprompted else ""))
        meta["episodes"].append(dict(id=e, start=int(a), end=int(b), frames=int(b - a + 1), seconds=round((b - a + 1) / fps, 1),
                                     clip=clip, clip_start=int(c0), clip_end=int(c1), tracks=trs,
                                     needs_manual_prompt=bool(unprompted), note=note))
    cap.release()
    json.dump(meta, open(os.path.join(tmpdir, "episodes.json"), "w"), indent=1)
    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted(os.listdir(tmpdir)):
            z.write(os.path.join(tmpdir, f), f)
            os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    n_manual = sum(ep["needs_manual_prompt"] for ep in meta["episodes"])
    print(f"episodes: {len(meta['episodes'])} (need manual prompt: {n_manual}) -> {args.out_zip} "
          f"{os.path.getsize(args.out_zip) // 1_000_000} MB; width≈{width:.0f}px contact<{D_CONTACT:.0f}px clean>={D_CLEAN:.0f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
