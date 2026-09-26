"""Extract benchmark frames (clean worms + cluster episodes) from one video as JPEG.
Usage: python3 extract_frames.py <video_name> <video_path>
Writes bench/frames/<video>/<frame:06d>.jpg and bench/frames/<video>/index.csv (frame,kind,episode)"""
import sys, csv, os, cv2, numpy as np
name, path = sys.argv[1], sys.argv[2]
out = f'bench/frames/{name}'; os.makedirs(out, exist_ok=True)
rows = list(csv.DictReader(open(f'auto_labels/{name}/qc_frames.csv')))
cl = [int(r['frame']) for r in rows if 'cluster' in r['flags']]
good = [int(r['frame']) for r in rows if 'cluster' not in r['flags'] and 'full' in r['label_status'] and 'split' not in r['flags']]
ep = []
for f in cl:
    if ep and f == ep[-1][1] + 1: ep[-1][1] = f
    else: ep.append([f, f])
want = {}
def pick(lst, k):
    if len(lst) <= k: return lst
    return [lst[int(i)] for i in np.linspace(0, len(lst) - 1, k)]
for f in pick(good, 40): want[f] = ('clean', -1)
n = len(rows)
if len(ep) == 1 and ep[0][1] - ep[0][0] > 2000:      # whole-video tangle (ga17)
    for f in pick(list(range(ep[0][0], ep[0][1] + 1)), 40): want[f] = ('cluster', 0)
else:
    per = max(3, 60 // max(1, len(ep)))
    for i, (a, b) in enumerate(ep):
        for f in pick(list(range(a, b + 1)), per): want[f] = ('cluster', i)
        for f in (a - 2, a - 1, b + 1, b + 2):
            if 0 <= f < n and f not in want: want[f] = ('context', i)
cap = cv2.VideoCapture(path); idx = 0; got = 0; last = max(want)
while idx <= last:
    ok, fr = cap.read()
    if not ok: break
    if idx in want:
        cv2.imwrite(f'{out}/{idx:06d}.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 95]); got += 1
    idx += 1
with open(f'{out}/index.csv', 'w') as fh:
    fh.write('frame,kind,episode\n')
    for f in sorted(want): fh.write(f'{f},{want[f][0]},{want[f][1]}\n')
print(name, 'episodes', len(ep), 'wanted', len(want), 'written', got)
