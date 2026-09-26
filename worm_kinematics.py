"""
worm_kinematics.py — per-track locomotion features from worm_pipeline outputs.

  python worm_kinematics.py <auto_labels_root> [video1 video2 ...]

Reads, for every video dir under <auto_labels_root> (or only the ones listed):
  labels49.npy   (N,49,2) or (N,K,49,2), NaN = point hidden
  summary.json   fps, per-track body length
Writes into <auto_labels_root>/../analysis/:
  kin_tracks.csv    one row per worm track  (features over the whole track)
  kin_windows.csv   one row per 20-s window (step 10 s) — material for clustering

Frame validity is two-level:
  core  = central half of the body visible (points 12..36) -> speed, pauses, frequency,
          bending activity, reversals (survives head_cut / tail_cut frames at the border)
  full  = all 49 points visible -> body length, wavelength, amplitude, curvature, coiling

Features (all size-normalized unless noted):
  L_px            median body length, px (NOT comparable across optical setups)
  speed_bl_s      median centroid speed while moving, body lengths / s
  frac_move / frac_pause / frac_thrash
                  moving = centroid speed >= 0.04 bl/s; pause = slow AND not bending;
                  thrash = slow but bending in place (nictation-like)
  freq_hz         dominant undulation frequency while moving (Welch PSD, mid-body curvature)
  freq_zc_hz      same from zero-crossing rate (sanity check)
  wavelength_bl   spatial wavelength of the body wave / L (zero crossings of kappa(s))
  amp_bl          half peak-to-peak lateral amplitude / L (median over moving frames)
  curv_rms        RMS dimensionless curvature |kappa|*L
  frac_coil       fraction of full-valid frames with max |kappa|*L > 12 (deep bend / omega)
  bend_act        median d(kappa*L)/dt at mid-body, 1/s
  rev_per_min     reversal starts per minute of core-valid time (sustained >= 0.4 s backward)
  frac_backward   fraction of moving time going backward
  core_s / full_s seconds of core- / full-valid frames (quality guard)
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

PAUSE_V = 0.04      # body lengths / s
BEND_ACT_LOW = 0.8  # (kappa*L)/s
COIL_K = 12.0       # |kappa|*L threshold for a deep bend / omega / coil
REV_MIN_S = 0.5
MIN_CORE_S = 3.0    # minimum seconds of core-valid frames for a track/window row
WIN_S, STEP_S = 20.0, 10.0
CORE = slice(12, 37)   # central half of the body
EDGE_I = 4             # curvature indices ignored at the very tips


def arc_len(pts):
    d = np.diff(pts, axis=-2)
    return np.hypot(d[..., 0], d[..., 1]).sum(axis=-1)


def curvature(pts, L):
    """pts (T,49,2) full frames only, L (T,) -> dimensionless curvature (T,49)."""
    x = gaussian_filter1d(pts[..., 0], 1.5, axis=-1)
    y = gaussian_filter1d(pts[..., 1], 1.5, axis=-1)
    dx, dy = np.gradient(x, axis=-1), np.gradient(y, axis=-1)
    ddx, ddy = np.gradient(dx, axis=-1), np.gradient(dy, axis=-1)
    k = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    return k * L[:, None]


def segments(mask, min_len=1):
    m = np.asarray(mask, bool)
    if not m.any(): return []
    d = np.diff(m.astype(int)); starts = list(np.where(d == 1)[0] + 1); stops = list(np.where(d == -1)[0] + 1)
    if m[0]: starts = [0] + starts
    if m[-1]: stops = stops + [len(m)]
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def dominant_freq(sig_segs, fps):
    segs = [s for s in sig_segs if len(s) >= int(3 * fps)]
    if not segs: return np.nan
    npseg = min(max(len(s) for s in segs), int(8 * fps))
    tot, f = None, None
    for s in segs:
        if len(s) < npseg: continue
        f, p = welch(s - s.mean(), fs=fps, nperseg=npseg)
        tot = p if tot is None else tot + p
    if tot is None:
        s = max(segs, key=len)
        f, tot = welch(s - s.mean(), fs=fps, nperseg=len(s))
    band = (f >= 0.05) & (f <= 0.8 * fps / 2)
    if not band.any() or not np.isfinite(tot[band]).any(): return np.nan
    return float(f[band][np.nanargmax(tot[band])])


def zc_rate(sig_segs, fps):
    n, T = 0, 0
    for s in sig_segs:
        if len(s) < fps: continue
        s = s - np.nanmean(s)
        n += int((np.diff(np.signbit(s)) != 0).sum()); T += len(s)
    return float(n / 2 / (T / fps)) if T else np.nan


def track_features(P, fps, L_ref=None):
    """P (T,49,2) with NaN for hidden points -> dict of features (or None)."""
    core = np.isfinite(P[:, CORE]).all(axis=(1, 2))
    full = np.isfinite(P).all(axis=(1, 2))
    if core.sum() < MIN_CORE_S * fps: return None

    L_full = np.where(full, arc_len(np.nan_to_num(P)), np.nan)
    L = float(np.nanmedian(L_full[full])) if full.sum() >= 2 * fps else (L_ref or np.nan)
    if not np.isfinite(L) or L < 50: return None

    # centroid of the central half -> speed in bl/s, smoothed ~0.5 s
    cent = np.nanmean(P[:, CORE], axis=1)
    cent[~core] = np.nan
    dc = np.diff(cent, axis=0)
    ok = core[1:] & core[:-1]
    v = np.full(len(P), np.nan)
    v[1:][ok] = np.hypot(dc[ok, 0], dc[ok, 1]) * fps / L
    vs = pd.Series(v).interpolate(limit=int(fps)).rolling(int(fps / 2) or 1, min_periods=1, center=True).median().to_numpy()

    # mid-body dimensionless curvature on core frames (local 5-pt estimate around index 24)
    Km = np.full(len(P), np.nan)
    seg = P[core][:, 18:31]
    Km[core] = curvature(seg, np.full(seg.shape[0], L))[:, 6]
    dk = np.full(len(P), np.nan)
    dkv = np.abs(np.diff(Km)) * fps
    dk[1:][ok] = dkv[ok]
    dks = pd.Series(dk).interpolate(limit=int(fps)).rolling(int(fps) or 1, min_periods=1, center=True).mean().to_numpy()

    moving = core & (vs >= PAUSE_V)
    paused = core & (vs < PAUSE_V) & (dks < BEND_ACT_LOW)
    thrash = core & (vs < PAUSE_V) & (dks >= BEND_ACT_LOW)
    ncore = int(core.sum())

    f = dict(L_px=round(L, 1), core_s=round(ncore / fps, 1), full_s=round(int(full.sum()) / fps, 1),
             frac_move=round(float(moving.sum() / ncore), 3),
             frac_pause=round(float(paused.sum() / ncore), 3),
             frac_thrash=round(float(thrash.sum() / ncore), 3),
             speed_bl_s=round(float(np.nanmedian(vs[moving])), 3) if moving.any() else 0.0,
             bend_act=round(float(np.nanmedian(dks[core])), 3))

    segs = [Km[a:b] for a, b in segments(moving & np.isfinite(Km), int(2 * fps))]
    f["freq_hz"] = round(dominant_freq(segs, fps), 3) if segs else np.nan
    segs_sm = [gaussian_filter1d(s, max(fps / 10, 1)) for s in segs]
    f["freq_zc_hz"] = round(zc_rate(segs_sm, fps), 3) if segs_sm else np.nan

    # full-shape features
    lam, amp = [], []
    Kfull = np.full((len(P), 49), np.nan)
    if full.any():
        Kfull[full] = curvature(P[full], L_full[full])
    Kb = Kfull[:, EDGE_I:-EDGE_I]
    use = full & moving
    if use.sum() < fps: use = full
    for i in np.where(use)[0]:
        row = Kb[i]
        if not np.isfinite(row).all(): continue
        # zero crossings with sub-index interpolation -> continuous wavelength estimate
        sgn = np.diff(np.signbit(row)) != 0
        zi = np.where(sgn)[0]
        if len(zi) >= 2:
            pos = (zi + row[zi] / (row[zi] - row[zi + 1]) + EDGE_I) / 48.0   # body fraction
            lam.append(2.0 * float(np.mean(np.diff(pos))))
        pts = P[i] - P[i].mean(0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        perp = pts @ vt[1]
        amp.append((perp.max() - perp.min()) / 2 / L)
    f["wavelength_bl"] = round(float(np.median(lam)), 3) if lam else np.nan
    f["amp_bl"] = round(float(np.median(amp)), 3) if amp else np.nan
    f["curv_rms"] = round(float(np.sqrt(np.nanmean(Kb[use] ** 2))), 3) if use.any() and np.isfinite(Kb[use]).any() else np.nan
    if full.any():
        with np.errstate(all="ignore"):
            kmax = np.nanmax(np.abs(Kb[full]), axis=1)
        f["frac_coil"] = round(float(np.nansum(kmax > COIL_K) / max(int(full.sum()), 1)), 3)
        f["kmax_p90"] = round(float(np.nanpercentile(kmax, 90)), 2)
    else:
        f["frac_coil"] = np.nan; f["kmax_p90"] = np.nan

    # reversals: centroid velocity projected on head-ward mid-body tangent
    tang = P[:, 16, :] - P[:, 32, :]
    tn = np.linalg.norm(tang, axis=1)
    tang = tang / np.where(tn > 1e-6, tn, np.nan)[:, None]
    vel = np.full_like(cent, np.nan); vel[1:][ok] = dc[ok] * fps / L
    velm = uniform_filter1d(np.nan_to_num(vel), int(fps / 2) or 1, axis=0)
    signed = (velm * tang).sum(axis=1)
    back = moving & (signed < -PAUSE_V)
    fwd = moving & (signed > PAUSE_V)
    rev_runs = segments(back, int(REV_MIN_S * fps))
    f["rev_per_min"] = round(len(rev_runs) / (ncore / fps / 60), 2)
    f["frac_backward"] = round(float(back.sum() / max(moving.sum(), 1)), 3)
    f["frac_forward"] = round(float(fwd.sum() / max(moving.sum(), 1)), 3)
    return f


def load_video(vdir):
    lab = np.load(os.path.join(vdir, "labels49.npy"))
    if lab.ndim == 3: lab = lab[:, None]
    s = json.load(open(os.path.join(vdir, "summary.json")))
    return lab, float(s["fps"]), s


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "auto_labels"
    vids = sys.argv[2:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    outdir = os.path.join(os.path.dirname(os.path.abspath(root)), "analysis")
    os.makedirs(outdir, exist_ok=True)
    trows, wrows = [], []
    for v in vids:
        vdir = os.path.join(root, v)
        if not os.path.exists(os.path.join(vdir, "labels49.npy")):
            print(f"{v}: no labels49.npy, skip"); continue
        lab, fps, s = load_video(vdir)
        N, K = lab.shape[:2]
        Lrefs = {t["id"]: t.get("body_length_px") for t in s.get("tracks", [])}
        for j in range(K):
            P = lab[:, j].astype(float)
            f = track_features(P, fps, Lrefs.get(f"worm{j+1}"))
            if f is None: continue
            trows.append(dict(video=v, track=f"worm{j+1}", fps=round(fps, 2), **f))
            w, st = int(WIN_S * fps), int(STEP_S * fps)
            for a in range(0, max(N - w, 0) + 1, st):
                fw = track_features(P[a:a + w], fps, Lrefs.get(f"worm{j+1}"))
                if fw is None or fw["core_s"] < 0.6 * WIN_S: continue
                wrows.append(dict(video=v, track=f"worm{j+1}", t0_s=round(a / fps, 1), **fw))
        print(f"{v}: {sum(1 for r in trows if r['video']==v)} track(s), "
              f"{sum(1 for r in wrows if r['video']==v)} windows", flush=True)
    pd.DataFrame(trows).to_csv(os.path.join(outdir, "kin_tracks.csv"), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(outdir, "kin_windows.csv"), index=False)
    print(f"wrote {len(trows)} tracks, {len(wrows)} windows -> {outdir}/kin_tracks.csv")


if __name__ == "__main__":
    main()
