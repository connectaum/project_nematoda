"""Thin wrapper around the SAM2 video predictor: device choice, model choice, prompts -> per-frame masks."""
import os
import shutil
import tempfile

import cv2
import numpy as np

MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def pick_model(name, device):
    if name and name != "auto":
        return MODELS.get(name, name)
    return MODELS["small"] if device in ("cuda", "mps") else MODELS["tiny"]


_PRED = {}


def load_predictor(model="auto", device=None):
    import torch  # noqa: F401  (import here so that `wormsam2 --help` works without torch)
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    device = device or pick_device()
    name = pick_model(model, device)
    key = (name, device)
    if key not in _PRED:
        _PRED[key] = SAM2VideoPredictor.from_pretrained(name, device=device)
    return _PRED[key], name, device


def write_frames(images, tmpdir):
    """SAM2 wants a directory of JPEGs named by index."""
    os.makedirs(tmpdir, exist_ok=True)
    for i, im in enumerate(images):
        cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])


def propagate(images, prompts, model="auto", device=None, progress=None):
    """
    images  : list of BGR frames (consecutive in time, any stride)
    prompts : list of dict(obj_id=int, frame=int (index into images), points=[[x,y],...] | None,
                           labels=[1/0,...] | None, mask=bool HxW | None)
    returns : masks bool (T, n_obj, H, W) ordered by prompts' obj_id order, plus the obj_id list
    """
    import torch
    predictor, name, device = load_predictor(model, device)
    T = len(images)
    H, W = images[0].shape[:2]
    obj_ids = [p["obj_id"] for p in prompts]
    masks = np.zeros((T, len(obj_ids), H, W), bool)
    tmp = tempfile.mkdtemp(prefix="wormsam2_")
    try:
        write_frames(images, tmp)
        # Objects prompted on DIFFERENT frames are propagated in separate passes (one shared state with
        # several prompt frames crashes on MPS and is fragile elsewhere); objects sharing a prompt frame go together.
        groups = {}
        for p in prompts:
            groups.setdefault(int(p["frame"]), []).append(p)
        done = 0
        total = 2 * T * len(groups)
        with torch.inference_mode():
            for f0, grp in sorted(groups.items()):
                st = predictor.init_state(video_path=tmp, offload_video_to_cpu=True,
                                          offload_state_to_cpu=(device == "cpu"))
                for p in grp:
                    if p.get("mask") is not None:
                        predictor.add_new_mask(st, frame_idx=f0, obj_id=int(p["obj_id"]),
                                               mask=torch.from_numpy(np.asarray(p["mask"], bool)))
                    else:
                        pts = np.asarray(p["points"], np.float32).reshape(-1, 2)
                        lab = np.asarray(p.get("labels") or [1] * len(pts), np.int32)
                        predictor.add_new_points_or_box(st, frame_idx=f0, obj_id=int(p["obj_id"]), points=pts, labels=lab)
                for rev in (False, True):
                    if rev and f0 == 0:
                        break
                    for fi, oids, logits in predictor.propagate_in_video(st, start_frame_idx=f0, reverse=rev):
                        for i, o in enumerate(oids):
                            masks[fi, obj_ids.index(int(o))] = (logits[i, 0] > 0).cpu().numpy()
                        done += 1
                        if progress:
                            progress(min(done, total), total)
                predictor.reset_state(st)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return masks, obj_ids, name, device
