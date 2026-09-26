"""Re-render the overlay video of a processed video from its (possibly SAM2-updated) auto_labels.

  python render_overlay.py <video.mp4> <auto_labels_dir> <out.mp4> [--scale 0.5]

Same look as worm_pipeline's overlay (5 coloured points per worm, per-track midline colour, status line),
plus: midlines that came from SAM2 are drawn thicker in white-outlined colour and the status line says SAM2.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 200, 0), (255, 0, 200)]   # head red ... tail magenta
LINE_COL = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (200, 0, 255), (255, 255, 0), (0, 128, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=float(os.environ.get("OVERLAY_SCALE", "1.0")))
    a = ap.parse_args()
    L = a.labels
    lab = np.load(os.path.join(L, "labels49.npy"))
    if lab.ndim == 3:
        lab = lab[:, None]
    N, K = lab.shape[:2]
    sub5 = np.linspace(0, 48, 5).round().astype(int)
    rows = list(csv.DictReader(open(os.path.join(L, "qc_frames.csv"))))
    summ = json.load(open(os.path.join(L, "summary.json")))
    cap = cv2.VideoCapture(a.video)
    fps = float(summ.get("fps") or cap.get(5) or 20.0)
    W, H = int(cap.get(3)), int(cap.get(4))
    OW, OH = int(round(W * a.scale)), int(round(H * a.scale))
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok or k >= N:
            break
        r = rows[k] if k < len(rows) else {"flags": "", "n_worms": "?", "label_status": ""}
        st = r["label_status"].split(";") if K > 1 else [r["label_status"]]
        st = (st + [""] * K)[:K]
        for j in range(K):
            d = lab[k, j]
            if np.isnan(d).any():
                continue
            from_sam2 = st[j].startswith("sam2")
            from_dtc = st[j].startswith("dtc")
            pts = d.astype(np.int32).reshape(-1, 1, 2)
            if from_sam2:
                cv2.polylines(fr, [pts], False, (255, 255, 255), 5, cv2.LINE_AA)
            elif from_dtc:
                cv2.polylines(fr, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)       # black outline = DeepTangle
            cv2.polylines(fr, [pts], False, LINE_COL[j % len(LINE_COL)], 3 if (from_sam2 or from_dtc) else 2, cv2.LINE_AA)
            for p in range(5):
                x, y = d[sub5[p]]
                cv2.circle(fr, (int(x), int(y)), 9, COLORS[p], -1, cv2.LINE_AA)
                cv2.circle(fr, (int(x), int(y)), 9, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(fr, f"H{j + 1}" + (" S" if from_sam2 else (" D" if from_dtc else "")), (int(d[0, 0]) + 12, int(d[0, 1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        fl = [f for f in r["flags"].split(";") if f and f != "touches_border"]
        sam2_here = "from_sam2" in fl
        dtc_here = "from_dtc" in fl
        fl = [f for f in fl if f not in ("from_sam2", "from_dtc")]
        txt = f"frame {k}  worms:{r['n_worms']}  " + ("SAM2" if sam2_here else ("DTC" if dtc_here else ("OK" if not fl else ",".join(fl)))) + "  " + r["label_status"]
        col = (255, 200, 0) if sam2_here else ((200, 120, 0) if dtc_here else ((0, 160, 0) if not fl else (0, 0, 255)))
        cv2.putText(fr, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3, cv2.LINE_AA)
        if a.scale != 1.0:
            fr = cv2.resize(fr, (OW, OH), interpolation=cv2.INTER_AREA)
        vw.write(fr)
        k += 1
    vw.release()
    cap.release()
    print(f"overlay written: {a.out} ({k} frames)")


if __name__ == "__main__":
    main()
