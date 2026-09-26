"""Process an episodes archive produced by wormbot (episode_export.py) with SAM2 and write a result archive
that the bot can merge back into the video's labels.

episodes_<video>.zip
    episodes.json          see FORMAT below
    ep01.mp4, ep02.mp4 …   clips (original fps, full resolution) covering each contact episode + margins

sam2_result_<video>.zip
    result.json            what was done (model, device, stride, per-episode objects and their track ids)
    labels_sam2.npz        frame (M,), track (M,) [-1 = manual object], obj (M,), midline (M,49,2), area (M,), plausible (M,)
    ep01_overlay.mp4 …     QC overlays (masks + midlines), ep01_sheet.jpg contact sheets
"""
import json
import os
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np

from .midline import mask_midline, arclen, orient_like

FORMAT = 1
COLS = [(0, 0, 255), (255, 80, 0), (0, 200, 0), (200, 0, 255), (0, 220, 255), (255, 0, 160), (120, 255, 0), (255, 200, 0)]   # BGR: red, blue, green, magenta, yellow…


def read_clip(path, stride=1):
    cap = cv2.VideoCapture(path)
    frames, idx, i = [], [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(im)
            idx.append(i)
        i += 1
    cap.release()
    return frames, idx


def midline_points(midline, n=7, labels_neg=None):
    """n well-spread positive points along a 49-pt midline, tips excluded."""
    m = np.asarray(midline, float)
    m = m[~np.isnan(m).any(1)]                      # partial labels (cut ends) hold NaN points
    if len(m) < 12:
        return []
    ii = np.linspace(4, len(m) - 5, n).round().astype(int)
    return m[ii].tolist()


def build_prompts(ep, clip_idx, stride):
    """Auto prompts from the bot's clean midlines: positive points on the worm, negative points on the others."""
    prompts = []
    tr = ep.get("tracks", [])
    for k, t in enumerate(tr):
        if t.get("midline") is None:
            continue
        f_local = int(t["prompt_frame"]) - int(ep["clip_start"])
        # nearest decoded frame at or before the prompt frame
        j = max(0, int(np.searchsorted(clip_idx, f_local, side="right") - 1))
        pos = midline_points(t["midline"])
        if not pos:
            continue
        neg = []
        if t.get("others"):                                   # format 2: the other worms on this very frame
            for o in t["others"]:
                neg += midline_points(o["midline"], n=5)
        else:                                                 # format 1: other prompts on (almost) the same frame
            for k2, t2 in enumerate(tr):
                if k2 != k and t2.get("midline") is not None and abs(int(t2["prompt_frame"]) - int(t["prompt_frame"])) <= 3 * stride:
                    neg += midline_points(t2["midline"], n=4)
        prompts.append(dict(obj_id=k + 1, frame=j, points=pos + neg, labels=[1] * len(pos) + [0] * len(neg),
                            track=int(t["track"]), id=t.get("id", f"track{t['track']}")))
    return prompts


def process_archive(zip_path, out_zip=None, model="auto", stride=2, gui=False, only=None, log=print, auto=False):
    from .sam2run import propagate
    work = tempfile.mkdtemp(prefix="wormsam2_ep_")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        meta = json.load(open(os.path.join(work, "episodes.json")))
        assert meta.get("format", 1) in (1, 2), "unknown episodes.json format"
        video = meta["video"]
        L_med, A_med = float(meta["median_length_px"]), float(meta["median_area_px"])
        out_zip = out_zip or os.path.join(os.path.dirname(os.path.abspath(zip_path)), f"sam2_result_{video}.zip")
        outdir = tempfile.mkdtemp(prefix="wormsam2_out_")
        rows = dict(frame=[], track=[], obj=[], episode=[], midline=[], area=[], plausible=[])
        res_eps = []
        t_all = time.time()
        model_name = device = None
        for ep in meta["episodes"]:
            if only and ep["id"] not in only:
                continue
            clip = os.path.join(work, ep["clip"])
            frames, cidx = read_clip(clip, stride)
            if not frames:
                log(f"episode {ep['id']}: empty clip, skipped")
                continue
            prompts = build_prompts(ep, cidx, stride)
            need_gui = gui or ep.get("needs_manual_prompt") or not prompts
            if need_gui and auto and prompts:
                need_gui = False
                log(f"episode {ep['id']}: {ep.get('note', '')} — --auto, continuing with the automatic prompts only")
            if need_gui and auto:
                log(f"episode {ep['id']}: no automatic prompts, skipped (--auto)")
                continue
            if need_gui:
                from .gui import ask_prompts
                extra = ask_prompts(frames, cidx, ep, existing=prompts, fps=meta.get("fps", 20) / stride)
                if extra is None:
                    log(f"episode {ep['id']}: skipped by user")
                    continue
                prompts = extra
            if not prompts:
                log(f"episode {ep['id']}: no prompts, skipped")
                continue
            log(f"episode {ep['id']} ({ep['start']}–{ep['end']}): {len(frames)} frames, {len(prompts)} objects…")
            t0 = time.time()
            masks, obj_ids, model_name, device = propagate(frames, prompts, model=model,
                                                           progress=lambda d, n: log(f"  {d}/{n} frames") if d % 50 == 0 else None)
            log(f"  done in {time.time() - t0:.0f}s on {device} ({model_name})")
            warn = []
            for k, p in enumerate(prompts):                       # a prompt that grabbed two worms shows as a 2x mask
                ratio = float(masks[p["frame"], k].sum()) / max(A_med, 1.0)
                p["prompt_area_ratio"] = round(ratio, 2)
                if ratio > 1.6:
                    warn.append(f"{p.get('id', p['obj_id'])} mask at its prompt frame is {ratio:.1f}x a worm — probably two worms; re-run with --gui and add negative points")
                elif ratio < 0.3:
                    warn.append(f"{p.get('id', p['obj_id'])} mask at its prompt frame is only {ratio:.1f}x a worm — prompt missed; re-run with --gui")
            for w in warn:
                log("  WARNING: " + w)
            # midlines, orientation continuity, QC overlay
            H, W = frames[0].shape[:2]
            vw = cv2.VideoWriter(os.path.join(outdir, f"ep{ep['id']:02d}_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(2, int(round(meta.get("fps", 20) / stride))), (W, H))
            sheet = []
            prev = {p["obj_id"]: (np.asarray(next((t["midline"] for t in ep.get("tracks", []) if t.get("track") == p.get("track", -2)), None), float)
                                  if p.get("track", -1) >= 0 else None) for p in prompts}
            n_ok = 0
            for i, im in enumerate(frames):
                f_abs = int(ep["clip_start"]) + cidx[i]
                ov = im.copy()
                for k, p in enumerate(prompts):
                    m = masks[i, k]
                    a = int(m.sum())
                    ml = mask_midline(m, L_ref=L_med, min_area=0.15 * A_med) if a > 0.15 * A_med else None
                    pl = ml is not None and 0.6 * L_med <= arclen(ml) <= 1.3 * L_med
                    if ml is not None:
                        ref = prev.get(p["obj_id"])
                        ml = orient_like(ml, ref if ref is not None and ref.shape == (49, 2) else None)
                        prev[p["obj_id"]] = ml
                    rows["frame"].append(f_abs)
                    rows["track"].append(int(p.get("track", -1)))
                    rows["obj"].append(int(p["obj_id"]))
                    rows["episode"].append(int(ep["id"]))
                    rows["midline"].append(ml if ml is not None else np.full((49, 2), np.nan))
                    rows["area"].append(a / A_med)
                    rows["plausible"].append(bool(pl))
                    n_ok += bool(pl)
                    c = COLS[k % len(COLS)]
                    ov[m] = (0.5 * ov[m] + 0.5 * np.array(c)).astype(np.uint8)
                    if ml is not None:
                        cv2.polylines(ov, [ml.astype(np.int32)], False, c, 2)
                        cv2.circle(ov, tuple(ml[0].astype(int)), 6, c, -1)      # head
                cv2.putText(ov, f"{video} f{f_abs} ep{ep['id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                vw.write(ov)
                if i % max(1, len(frames) // 12) == 0 and len(sheet) < 12:
                    sheet.append(cv2.resize(ov, (640, int(640 * H / W))))
            vw.release()
            while len(sheet) % 3:
                sheet.append(np.zeros_like(sheet[0]))
            cv2.imwrite(os.path.join(outdir, f"ep{ep['id']:02d}_sheet.jpg"),
                        np.vstack([np.hstack(sheet[j:j + 3]) for j in range(0, len(sheet), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            frac = n_ok / max(1, len(frames) * len(prompts))
            res_eps.append(dict(id=ep["id"], start=ep["start"], end=ep["end"], frames=len(frames), stride=stride,
                                objects=[dict(obj=p["obj_id"], track=p.get("track", -1), id=p.get("id", "manual"), prompt_area_ratio=p.get("prompt_area_ratio")) for p in prompts],
                                warnings=warn,
                                plausible_fraction=round(frac, 3), seconds=round(time.time() - t0, 1)))
            log(f"  plausible-length midlines: {frac:.0%} of object-frames")
        np.savez_compressed(os.path.join(outdir, "labels_sam2.npz"),
                            frame=np.array(rows["frame"], np.int32), track=np.array(rows["track"], np.int32),
                            obj=np.array(rows["obj"], np.int32), episode=np.array(rows["episode"], np.int32),
                            midline=np.array(rows["midline"], np.float32).reshape(-1, 49, 2),
                            area=np.array(rows["area"], np.float32), plausible=np.array(rows["plausible"], bool))
        json.dump(dict(format=FORMAT, video=video, model=model_name, device=device, stride=stride,
                       total_seconds=round(time.time() - t_all, 1), episodes=res_eps),
                  open(os.path.join(outdir, "result.json"), "w"), indent=1)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(outdir)):
                z.write(os.path.join(outdir, f), f)
        shutil.rmtree(outdir, ignore_errors=True)
        log(f"result: {out_zip}")
        return out_zip
    finally:
        shutil.rmtree(work, ignore_errors=True)
