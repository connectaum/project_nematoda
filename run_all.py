"""
One-shot driver: every video in a folder -> auto labels -> (one) DeepLabCut project.

  python run_all.py --videos videos/ --project nematoda-auto-2026-08-22 --npoints 5 --nframes 40
  python run_all.py --videos videos/ --project nematoda-ma-2026-08-24 --multi --individuals 3   # multi-animal project

Re-running is safe: videos already in the project are skipped; new videos are added
(labels + frames + an entry in config.yaml). After that open the project in DLC:
  python -m deeplabcut   ->  Load project  ->  config.yaml  ->  "Label frames" shows the auto labels.
"""
import argparse, subprocess, sys
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--project", required=True)
ap.add_argument("--labels-root", default="auto_labels")
ap.add_argument("--npoints", default="5")
ap.add_argument("--nframes", default="40")
ap.add_argument("--ext", default=".mp4,.avi,.mov")
ap.add_argument("--multi", action="store_true", help="build / extend a multi-animal (maDLC) project")
ap.add_argument("--individuals", default="3")
ap.add_argument("--overlay-scale", default="0.5", help="overlay.mp4 scale for videos longer than 1000 frames")
a = ap.parse_args()

here = Path(__file__).parent
cfg = Path(a.project) / "config.yaml"
done = set()
if cfg.exists():
    done = {Path(v).stem for v in yaml.safe_load(cfg.read_text())["video_sets"]}

vids = sorted(p for p in Path(a.videos).iterdir() if p.suffix.lower() in a.ext.split(","))
for v in vids:
    if v.stem in done:
        print(f"skip {v.name} (already in project)"); continue
    out = Path(a.labels_root) / v.stem
    if not (out / "labels49.npy").exists():
        print(f"== labelling {v.name}")
        import cv2, os
        n = int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT))
        env = dict(os.environ, OVERLAY_SCALE=a.overlay_scale if n > 1000 else "1.0")
        subprocess.run([sys.executable, "-W", "ignore", str(here / "worm_pipeline.py"), str(v), str(out)], check=True, env=env)
    print(f"== adding {v.name} to {a.project}")
    subprocess.run([sys.executable, "-W", "ignore", str(here / "make_dlc_project.py"),
                    "--project", a.project, "--video", str(v), "--labels", str(out),
                    "--npoints", a.npoints, "--nframes", a.nframes]
                   + (["--multi", "--individuals", a.individuals] if a.multi else []), check=True)
print("done. Next: open config.yaml in DeepLabCut (see README_DLC.md)")
