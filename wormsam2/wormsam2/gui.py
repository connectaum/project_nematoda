"""Minimal cross-platform prompt editor (matplotlib): click on worms to tell SAM2 what to track.

Keys / mouse
  ← →            previous / next frame (prompts are placed on the frame shown)
  left click     positive point for the CURRENT worm
  right click    negative point for the current worm (something that is NOT this worm)
  n              start a new worm
  d              delete the last point
  x              delete the current worm
  Enter          done — run SAM2 with these prompts
  Esc            skip this episode
Auto prompts from the bot (worms that were seen clean before the episode) are shown and kept.
"""
import numpy as np

COLORS = ["#00c8ff", "#ff7800", "#00dc00", "#c800ff", "#ffff00", "#ff0078", "#78ff00", "#ffc800"]


def ask_prompts(frames, clip_idx, ep, existing=None, fps=10):
    import importlib
    import sys
    import matplotlib
    # pick an interactive backend that actually imports (matplotlib.use() alone is lazy and does not check)
    cands = ["macosx", "qtagg", "tkagg"] if sys.platform == "darwin" else ["tkagg", "qtagg"]
    for name in cands:
        try:
            importlib.import_module(f"matplotlib.backends.backend_{name}")
            matplotlib.use(name)
            break
        except Exception:
            continue
    else:
        raise SystemExit("wormsam2: no GUI backend for the click editor. Install one: "
                         "macOS/Linux -> a Python with Tk (brew install python-tk@3.11) or `uv pip install PyQt6` "
                         "into wormsam2/.venv; Windows -> python.org Python includes Tk. "
                         "Or run with --auto to skip the editor.")
    import matplotlib.pyplot as plt

    state = dict(i=0, objs=[], cur=None, done=False, skip=False)
    for p in (existing or []):
        pts = np.asarray(p["points"], float)
        lab = list(p.get("labels") or [1] * len(pts))
        state["objs"].append(dict(frame=p["frame"], pos=[tuple(q) for q, l in zip(pts.tolist(), lab) if l == 1],
                                  neg=[tuple(q) for q, l in zip(pts.tolist(), lab) if l == 0],
                                  track=p.get("track", -1), id=p.get("id", "auto"), auto=True))
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f"wormsam2 — {ep.get('id', '?')}: frames {ep.get('start')}–{ep.get('end')}")
    img = ax.imshow(frames[0][:, :, ::-1])
    ax.set_axis_off()
    title = ax.set_title("")
    artists = []

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        img.set_data(frames[state["i"]][:, :, ::-1])
        f_abs = int(ep.get("clip_start", 0)) + clip_idx[state["i"]]
        for k, o in enumerate(state["objs"]):
            c = COLORS[k % len(COLORS)]
            vis = o["frame"] == state["i"]
            alpha = 1.0 if vis else 0.25
            if o["pos"]:
                a = ax.scatter(*zip(*o["pos"]), s=60, c=c, marker="o", alpha=alpha, edgecolors="k")
                artists.append(a)
            if o["neg"]:
                a = ax.scatter(*zip(*o["neg"]), s=60, c=c, marker="x", alpha=alpha)
                artists.append(a)
            if o["pos"]:
                a = ax.annotate(f"{k + 1}{'*' if o.get('auto') else ''}", o["pos"][0], color=c, fontsize=12, weight="bold")
                artists.append(a)
        cur = state["cur"]
        title.set_text(f"frame {f_abs}  ({state['i'] + 1}/{len(frames)})   worms: {len(state['objs'])}   "
                       f"current: {'-' if cur is None else cur + 1}   [n new worm, click = point, Enter run, Esc skip]")
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        if state["cur"] is None or state["objs"][state["cur"]]["frame"] != state["i"]:
            state["objs"].append(dict(frame=state["i"], pos=[], neg=[], track=-1, id=f"manual{len(state['objs']) + 1}", auto=False))
            state["cur"] = len(state["objs"]) - 1
        o = state["objs"][state["cur"]]
        (o["pos"] if ev.button == 1 else o["neg"]).append((float(ev.xdata), float(ev.ydata)))
        redraw()

    def on_key(ev):
        k = ev.key
        if k == "right":
            state["i"] = min(len(frames) - 1, state["i"] + 1)
        elif k == "left":
            state["i"] = max(0, state["i"] - 1)
        elif k == "n":
            state["objs"].append(dict(frame=state["i"], pos=[], neg=[], track=-1, id=f"manual{len(state['objs']) + 1}", auto=False))
            state["cur"] = len(state["objs"]) - 1
        elif k == "d" and state["cur"] is not None:
            o = state["objs"][state["cur"]]
            if o["neg"]:
                o["neg"].pop()
            elif o["pos"]:
                o["pos"].pop()
        elif k == "x" and state["cur"] is not None:
            state["objs"].pop(state["cur"])
            state["cur"] = len(state["objs"]) - 1 if state["objs"] else None
        elif k == "enter":
            state["done"] = True
            plt.close(fig)
            return
        elif k == "escape":
            state["skip"] = True
            plt.close(fig)
            return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show(block=True)
    if state["skip"] or not state["done"] and not state["objs"]:
        return None
    prompts = []
    for k, o in enumerate(state["objs"]):
        if not o["pos"]:
            continue
        prompts.append(dict(obj_id=k + 1, frame=int(o["frame"]), points=o["pos"] + o["neg"],
                            labels=[1] * len(o["pos"]) + [0] * len(o["neg"]), track=int(o.get("track", -1)), id=o.get("id", "manual")))
    return prompts
