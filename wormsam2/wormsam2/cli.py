"""wormsam2 command line.

  wormsam2 run episodes_IMG_8370.zip [--out sam2_result_IMG_8370.zip] [--model auto|tiny|small|base] [--stride 2] [--gui | --auto] [--episode 1 --episode 3]
  wormsam2 selftest            download the model, run a tiny propagation, print device and speed
  wormsam2 info                show torch / device / models
"""
import argparse
import os
import sys
import time


def cmd_info(args):
    import torch
    from .sam2run import pick_device, pick_model
    dev = pick_device()
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  device: {dev}")
    if dev == "cuda":
        print("  gpu:", torch.cuda.get_device_name(0))
    print("  default model:", pick_model("auto", dev))
    print("  models cached in:", os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))


def cmd_selftest(args):
    import numpy as np
    from .sam2run import propagate
    rng = np.random.default_rng(0)
    T, H, W = 6, 360, 640
    frames = []
    for t in range(T):
        im = np.full((H, W, 3), 200, np.uint8)
        x = 150 + 8 * t
        import cv2
        cv2.ellipse(im, (x, 180), (90, 12), 20, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(im, (450, 200 - 5 * t), (80, 12), -30, 0, 360, (40, 40, 40), -1)
        im = np.clip(im.astype(int) + rng.integers(-8, 8, im.shape), 0, 255).astype(np.uint8)
        frames.append(im)
    t0 = time.time()
    masks, ids, name, dev = propagate(frames, [dict(obj_id=1, frame=0, points=[[150, 180], [120, 170], [180, 190]]),
                                              dict(obj_id=2, frame=0, points=[[450, 200], [420, 215], [480, 185]])], model=args.model)
    dt = (time.time() - t0) / T
    areas = masks.reshape(T, 2, -1).sum(-1)
    ok = (areas > 500).all()
    print(f"model {name} on {dev}: {dt:.2f} s/frame at {W}x{H} (first run includes warm-up); "
          f"objects tracked in all frames: {'yes' if ok else 'NO'}")
    print("selftest OK" if ok else "selftest FAILED")
    return 0 if ok else 1


def cmd_run(args):
    from .episodes import process_archive
    only = set(args.episode) if args.episode else None
    out = process_archive(args.zip, out_zip=args.out, model=args.model, stride=args.stride, gui=args.gui, only=only, auto=args.auto)
    print("\nDone. Send this file back to the bot as a reply to its episodes message:\n  " + out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wormsam2", description="SAM2 propagation for stuck/coiled nematodes (wormbot episodes)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process an episodes_<video>.zip from the bot")
    r.add_argument("zip")
    r.add_argument("--out", default=None, help="output zip (default: sam2_result_<video>.zip next to the input)")
    r.add_argument("--model", default="auto", help="auto | tiny | small | base | large  (auto = small on GPU/MPS, tiny on CPU)")
    r.add_argument("--stride", type=int, default=2, help="process every N-th frame (worms in contact barely move; 2–3 is fine)")
    r.add_argument("--gui", action="store_true", help="open the click-prompt editor for every episode (otherwise only when some worm has no clean prompt)")
    r.add_argument("--auto", action="store_true", help="never open the editor: use only the automatic prompts, skip episodes without any")
    r.add_argument("--episode", type=int, action="append", help="only these episode ids (repeatable)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("selftest", help="download the model and check that propagation works on this machine")
    s.add_argument("--model", default="auto")
    s.set_defaults(fn=cmd_selftest)
    i = sub.add_parser("info", help="torch / device / model info")
    i.set_defaults(fn=cmd_info)
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
