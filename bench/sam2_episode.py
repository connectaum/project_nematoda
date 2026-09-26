"""SAM2 propagation through a full-rate cluster episode of a video.
  PYTORCH_ENABLE_MPS_FALLBACK=1 python sam2_episode.py <video> <f_start> <f_end> [stride] [model]
Prompts = segmentation components of the FIRST frame (must be a clean frame with separate worms).
Outputs in results/sam2ep/<video>_<f_start>-<f_end>/: masks.npz (packbits, T x K x H x W), stats.csv, overlay.mp4, sheet.jpg"""
import sys, os, time, shutil, warnings, numpy as np, cv2, pandas as pd, torch; warnings.filterwarnings("ignore")
from sam2.sam2_video_predictor import SAM2VideoPredictor
from skimage.morphology import skeletonize
from common import *
from methods import longest_path, crop

v, f0, f1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
REV = f0 > f1
if REV: f0, f1 = f1, f0
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 2
mt = sys.argv[5] if len(sys.argv) > 5 else "facebook/sam2.1-hiera-small"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ctx = VideoCtx(v)
OUT = os.path.join(ROOT, "bench", "results", "sam2ep", f"{v}_{f0}-{f1}_s{stride}{'_rev' if REV else ''}_{mt.split('-')[-1]}"); os.makedirs(OUT, exist_ok=True)
TMP = os.path.join(ROOT, "bench", "_sam2_tmp"); shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
vp = [p for p in os.listdir(f"{ROOT}/videos") if p.lower().startswith(v.lower() + ".")][0]
cap = cv2.VideoCapture(f"{ROOT}/videos/{vp}"); cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
frames = []; f = f0
while f <= f1:
    ok, img = cap.read()
    if not ok: break
    if (f - f0) % stride == 0:
        cv2.imwrite(f"{TMP}/{len(frames):05d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95]); frames.append(f)
    f += 1
cap.release()
if REV:   # propagate backwards in time: reverse the frame order on disk
    for i in range(len(frames)): os.rename(f"{TMP}/{i:05d}.jpg", f"{TMP}/r{i:05d}.jpg")
    for i in range(len(frames)): os.rename(f"{TMP}/r{i:05d}.jpg", f"{TMP}/{len(frames)-1-i:05d}.jpg")
    frames = frames[::-1]
T = len(frames); print(v, T, "frames", "reverse" if REV else "", flush=True)
img0 = cv2.imread(f"{TMP}/00000.jpg")
comps = components(ctx.segment(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)), a_min=int(0.4*ctx.A_med))
print("prompt objects:", [(c["area"], [round(x) for x in c["centroid"]]) for c in comps], "A_med", ctx.A_med, flush=True)
predictor = SAM2VideoPredictor.from_pretrained(mt, device=dev)
t0 = time.time(); masks = np.zeros((T, len(comps), ctx.H, ctx.W), bool)
with torch.inference_mode():
    st = predictor.init_state(video_path=TMP, offload_video_to_cpu=True)
    for k, c in enumerate(comps): predictor.add_new_mask(st, frame_idx=0, obj_id=k+1, mask=torch.from_numpy(c["mask"]))
    for fi, oids, logits in predictor.propagate_in_video(st):
        for i, o in enumerate(oids): masks[fi, o-1] = (logits[i, 0] > 0).cpu().numpy()
print(f"propagated in {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"{OUT}/masks.npz", masks=np.packbits(masks, axis=-1), frames=np.array(frames))

def midline(m):
    if m.sum() < 0.15*ctx.A_med: return None
    sub, off = crop(m); p = longest_path(skeletonize(sub))
    return None if p is None else resample_polyline(p + off)

rows = []; cols = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (200, 0, 255)]
vw = cv2.VideoWriter(f"{OUT}/overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (ctx.W, ctx.H)); sheet = []
for i, f in enumerate(frames):
    img = cv2.imread(f"{TMP}/{i:05d}.jpg"); ov = img.copy(); r = dict(frame=f)
    seg = ctx.segment(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for k in range(len(comps)):
        m = masks[i, k]; a = int(m.sum()); p = midline(m); L = arclen(p)/ctx.L_med if p is not None else 0
        r[f"area{k}"] = round(a/ctx.A_med, 2); r[f"L{k}"] = round(L, 2)
        r[f"inseg{k}"] = round(float((m & seg).sum()/max(a, 1)), 2)
        ov[m] = (0.5*ov[m] + 0.5*np.array(cols[k % 4])).astype(np.uint8)
        if p is not None: cv2.polylines(ov, [p.astype(np.int32)], False, cols[k % 4], 2)
    inter = 0
    for a_ in range(len(comps)):
        for b_ in range(a_+1, len(comps)): inter += int((masks[i, a_] & masks[i, b_]).sum())
    r["overlap"] = round(inter/ctx.A_med, 2)
    r["n_plausible"] = sum(0.6 <= r[f"L{k}"] <= 1.3 for k in range(len(comps)))
    rows.append(r); cv2.putText(ov, f"{v} f{f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); vw.write(ov)
    if i % max(1, T // 12) == 0 and len(sheet) < 12: sheet.append(cv2.resize(ov, (640, 360)))
vw.release(); shutil.rmtree(TMP, ignore_errors=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/stats.csv", index=False)
while len(sheet) % 3: sheet.append(np.zeros_like(sheet[0]))
cv2.imwrite(f"{OUT}/sheet.jpg", np.vstack([np.hstack(sheet[i:i+3]) for i in range(0, len(sheet), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
K = len(comps)
print(f"{v} {f0}-{f1}: {T} frames, {K} objs; frames with all objs plausible-length: {(df.n_plausible==K).mean():.2f}; "
      f"mean area/A_med per obj: {[round(df[f'area{k}'].mean(),2) for k in range(K)]}; mean overlap {df.overlap.mean():.2f}; "
      f"mask inside segmentation: {[round(df[f'inseg{k}'].mean(),2) for k in range(K)]}", flush=True)
print(df.iloc[::max(1, T//15)].to_string(index=False))
