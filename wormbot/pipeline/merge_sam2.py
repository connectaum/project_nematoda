"""Merge a wormsam2 result archive into a video's auto_labels and refresh the QC tables.

  python merge_sam2.py <auto_labels_dir> <sam2_result.zip>

* labels49.npy   : SAM2 midlines (plausible length only) are written into the frames/tracks of the contact episodes;
                   frames skipped by the stride are linearly interpolated between neighbouring SAM2 frames (gap <= 4);
                   manual objects (track = -1) become new tracks appended after the existing ones.
                   The previous file is kept as labels49_before_sam2.npy.
* qc_frames.csv  : flag 'from_sam2' on touched frames, label_status 'sam2' / 'sam2_interp' per track.
* tracks.json / summary.json : counts updated, summary['sam2'] = what was merged.
Prints a one-line summary. Re-run worm_kinematics.py afterwards.
"""
import csv
import json
import os
import sys
import tempfile
import zipfile

import numpy as np


def orient_like(p, ref):
    if ref is None or np.isnan(ref).any():
        return p
    e1 = np.mean(np.hypot(*(p - ref).T))
    e2 = np.mean(np.hypot(*(p[::-1] - ref).T))
    return p if e1 <= e2 else p[::-1]


def main():
    L, zpath = sys.argv[1], sys.argv[2]
    tmp = tempfile.mkdtemp(prefix="sam2merge_")
    with zipfile.ZipFile(zpath) as z:
        z.extract("result.json", tmp)
        z.extract("labels_sam2.npz", tmp)
    res = json.load(open(os.path.join(tmp, "result.json")))
    d = np.load(os.path.join(tmp, "labels_sam2.npz"))
    frame, track, obj, ep = d["frame"], d["track"], d["obj"], d["episode"]
    mid, plaus = d["midline"], d["plausible"]

    lab_path = os.path.join(L, "labels49.npy")
    lab = np.load(lab_path)
    if lab.ndim == 3:
        lab = lab[:, None]
    lab = lab.astype(np.float32)
    N, K = lab.shape[:2]
    if not os.path.exists(os.path.join(L, "labels49_before_sam2.npy")):
        np.save(os.path.join(L, "labels49_before_sam2.npy"), lab)
    tracks = json.load(open(os.path.join(L, "tracks.json")))
    summ = json.load(open(os.path.join(L, "summary.json")))

    # manual objects -> new track columns
    manual = sorted({(int(e), int(o)) for e, o, t in zip(ep, obj, track) if t < 0})
    new_cols = {}
    for e, o in manual:
        new_cols[(e, o)] = K + len(new_cols)
    if new_cols:
        lab = np.concatenate([lab, np.full((N, len(new_cols), 49, 2), np.nan, np.float32)], axis=1)
    K2 = lab.shape[1]
    touched = {}                     # (frame, col) -> 'sam2' | 'sam2_interp'
    per_track = {}
    for i in range(len(frame)):
        if not plaus[i] or np.isnan(mid[i]).any():
            continue
        col = int(track[i]) if track[i] >= 0 else new_cols[(int(ep[i]), int(obj[i]))]
        f = int(frame[i])
        if not (0 <= f < N):
            continue
        per_track.setdefault(col, []).append((f, mid[i].astype(np.float32)))
    n_set = n_interp = 0
    for col, items in per_track.items():
        items.sort(key=lambda t: t[0])
        # orientation: continuity with the nearest existing label of this track, then chain
        prev = None
        for f, m in items:
            ref = prev
            if ref is None:
                before = np.where(~np.isnan(lab[:f, col, 0, 0]))[0]
                if len(before):
                    ref = lab[before[-1], col]
            m = orient_like(m, ref)
            lab[f, col] = m
            touched[(f, col)] = "sam2"
            prev = m
            n_set += 1
        for (f1, m1), (f2, m2) in zip(items[:-1], items[1:]):
            if 1 < f2 - f1 <= 4:
                a, b = lab[f1, col], lab[f2, col]
                for f in range(f1 + 1, f2):
                    if (f, col) in touched:
                        continue
                    w = (f - f1) / (f2 - f1)
                    lab[f, col] = (1 - w) * a + w * b
                    touched[(f, col)] = "sam2_interp"
                    n_interp += 1
    np.save(lab_path, lab if K2 > 1 else lab[:, 0])

    # qc_frames.csv
    qc_path = os.path.join(L, "qc_frames.csv")
    rows = list(csv.DictReader(open(qc_path)))
    fields = list(rows[0].keys())
    frames_touched = {}
    for (f, col), kind in touched.items():
        frames_touched.setdefault(f, {})[col] = kind
    for r in rows:
        f = int(r["frame"])
        if f not in frames_touched:
            if K2 > K:                                   # pad status for new tracks
                r["label_status"] = ";".join((r["label_status"].split(";") if K > 1 else [r["label_status"]]) + ["-"] * (K2 - K))
            continue
        st = r["label_status"].split(";") if K > 1 else [r["label_status"]]
        st = (st + ["-"] * K2)[:K2]
        for col, kind in frames_touched[f].items():
            st[col] = kind
        r["label_status"] = ";".join(st) if K2 > 1 else st[0]
        fl = set(x for x in r["flags"].split(";") if x)
        fl.add("from_sam2")
        r["flags"] = ";".join(sorted(fl))
        r["n_worms"] = str(sum(1 for s in st if s and s not in ("-", "no_midline", "both_ends_cut", "visible_longer_than_body")))
    with open(qc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # tracks.json / summary.json
    for (e, o), col in new_cols.items():
        tracks.append(dict(id=f"sam2_{col + 1}", n_frames=0, first_frame=None, last_frame=None, body_length_px=None,
                           origin=f"wormsam2 manual object {o} in episode {e}"))
    for col in range(K2):
        fr = np.where(~np.isnan(lab[:, col, 0, 0]))[0]
        if col < len(tracks):
            tracks[col]["n_frames"] = int(len(fr))
            if len(fr):
                tracks[col]["first_frame"], tracks[col]["last_frame"] = int(fr[0]), int(fr[-1])
            tracks[col]["sam2_frames"] = int(sum(1 for (f, c) in touched if c == col))
    json.dump(tracks, open(os.path.join(L, "tracks.json"), "w"), indent=2)
    summ["n_tracks"] = K2
    summ["tracks"] = tracks
    summ.setdefault("flag_counts", {})["from_sam2"] = len(frames_touched)
    summ["frames_labelled"] = int(np.any(~np.isnan(lab[:, :, 0, 0]), axis=1).sum())
    summ["sam2"] = dict(result=os.path.basename(zpath), model=res.get("model"), device=res.get("device"), stride=res.get("stride"),
                        episodes=[dict(id=x["id"], start=x["start"], end=x["end"], plausible_fraction=x.get("plausible_fraction"))
                                  for x in res.get("episodes", [])],
                        frames_set=n_set, frames_interpolated=n_interp, new_tracks=len(new_cols))
    json.dump(summ, open(os.path.join(L, "summary.json"), "w"), indent=2, ensure_ascii=False)
    print(f"merged: {n_set} SAM2 midlines + {n_interp} interpolated into {len(frames_touched)} frames, "
          f"{len(new_cols)} new tracks; tracks now {K2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
