"""
kin_analysis.py — do the videos fall apart into locomotion groups?

  python kin_analysis.py analysis/

Input:  analysis/kin_tracks.csv, analysis/kin_windows.csv (from worm_kinematics.py)
Output: analysis/kin_summary_by_video.csv
        analysis/fig_features_by_video.png   dot plot: key features per video (track medians)
        analysis/fig_pca_windows.png         PCA of 20-s windows, colored by cluster, labeled by video
        analysis/fig_dendrogram.png          hierarchical clustering of tracks
Cluster count: silhouette over k=2..4 on the window PCA scores.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

# ---- reference palette (light mode) ----
SURFACE = "#fcfcfb"; PAGE = "#f9f9f7"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3: all-pairs safe on scatter
MARKERS = ["o", "s", "^"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "svg.fonttype": "none",
})

FEATS = ["speed_bl_s", "freq_hz", "wavelength_bl", "amp_bl", "curv_rms",
         "frac_coil", "frac_move", "frac_thrash", "bend_act", "rev_per_min"]
FEAT_RU = {"speed_bl_s": "скорость, длин тела/с", "freq_hz": "частота ундуляций, Гц",
           "wavelength_bl": "длина волны / L", "amp_bl": "амплитуда / L",
           "curv_rms": "кривизна RMS (κ·L)", "frac_coil": "доля кадров с глубоким изгибом",
           "frac_move": "доля времени в движении", "frac_thrash": "изгибы на месте (доля)",
           "bend_act": "активность изгибов, 1/с", "rev_per_min": "реверсы, 1/мин",
           "L_px": "длина тела, px"}


def zscore(X):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd == 0] = 1
    return (X - mu) / sd


def silhouette(X, labels):
    from scipy.spatial.distance import cdist
    u = np.unique(labels)
    if len(u) < 2: return -1
    D = cdist(X, X)
    s = []
    for i in range(len(X)):
        same = labels == labels[i]; same[i] = False
        if not same.any(): continue
        a = D[i][same].mean()
        b = min(D[i][labels == c].mean() for c in u if c != labels[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def main():
    adir = sys.argv[1] if len(sys.argv) > 1 else "analysis"
    tr = pd.read_csv(os.path.join(adir, "kin_tracks.csv"))
    wi = pd.read_csv(os.path.join(adir, "kin_windows.csv"))
    tr["setup"] = np.where(tr["video"].str.startswith("IMG"), "IMG (720p)", "ga (1080p)")

    # ---------- per-video summary (weight tracks by observed time) ----------
    rows = []
    for v, g in tr.groupby("video"):
        w = g["core_s"].to_numpy()
        r = {"video": v, "n_tracks": len(g), "obs_s": round(w.sum(), 1),
             "L_px": round(float(np.average(g["L_px"], weights=w)), 0)}
        for f in FEATS:
            vals, ww = g[f].to_numpy(), w.copy()
            m = np.isfinite(vals)
            r[f] = round(float(np.average(vals[m], weights=ww[m])), 3) if m.any() else np.nan
        rows.append(r)
    sm = pd.DataFrame(rows).sort_values("video")
    sm.to_csv(os.path.join(adir, "kin_summary_by_video.csv"), index=False)
    print(sm.to_string(index=False))

    # ---------- fig 1: dot plot of key features per video ----------
    show = ["L_px", "speed_bl_s", "freq_hz", "wavelength_bl", "amp_bl",
            "curv_rms", "frac_coil", "frac_thrash", "rev_per_min"]
    vids = sm["video"].tolist()
    fig, axes = plt.subplots(1, len(show), figsize=(2.1 * len(show), 0.55 * len(vids) + 1.6),
                             sharey=True)
    y = np.arange(len(vids))[::-1]
    for ax, f in zip(axes, show):
        # per-track dots (small, muted) + per-video weighted mean (large)
        for yi, v in zip(y, vids):
            g = tr[tr["video"] == v]
            ax.plot(g[f], np.full(len(g), yi), "o", ms=4, color=MUTED, alpha=0.55, zorder=2)
        ax.plot(sm[f], y, "o", ms=8, color=CAT[0], zorder=3)
        ax.set_title(FEAT_RU.get(f, f), fontsize=9, color=INK2, pad=8)
        ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=8)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(vids, fontsize=9)
    fig.suptitle("Локомоция по видео: точки — отдельные черви (треки), крупная точка — среднее по видео",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(adir, "fig_features_by_video.png"), dpi=160)
    plt.close(fig)

    # ---------- windows -> PCA + clusters ----------
    Wd = wi.dropna(subset=["freq_hz", "wavelength_bl", "amp_bl"]).copy()
    used = [f for f in FEATS if Wd[f].notna().mean() > 0.7]
    X = zscore(Wd[used].fillna(Wd[used].median()).to_numpy())
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    expl = (S ** 2 / (S ** 2).sum())[:2]

    best_k, best_s, best_lab = 1, -1, np.zeros(len(X), int)
    Z = linkage(X, "ward")
    for k in (2, 3):
        lab = fcluster(Z, k, "maxclust") - 1
        s = silhouette(X, lab)
        print(f"k={k}: silhouette {s:.3f}")
        if s > best_s: best_k, best_s, best_lab = k, s, lab
    Wd["cluster"] = best_lab

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for c in range(best_k):
        m = best_lab == c
        ax.scatter(pcs[m, 0], pcs[m, 1], s=42, c=CAT[c], marker=MARKERS[c],
                   edgecolors=SURFACE, linewidths=1.2, zorder=3,
                   label=f"кластер {c + 1} (n={m.sum()})")
    # label each video at its window centroid
    for v, g in Wd.groupby("video"):
        i = g.index
        loc = np.searchsorted(Wd.index, i)  # positions in pcs
        pos = np.array([np.median(pcs[Wd.index.get_indexer(i), 0]),
                        np.median(pcs[Wd.index.get_indexer(i), 1])])
        ax.annotate(v, pos, fontsize=9, color=INK, fontweight="bold",
                    xytext=(0, 9), textcoords="offset points", ha="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec=GRID, alpha=0.85))
    ax.set_xlabel(f"PC1 ({expl[0]:.0%} дисперсии)"); ax.set_ylabel(f"PC2 ({expl[1]:.0%})")
    ax.set_title("Окна по 20 с в пространстве признаков (PCA): цвет — кластер, подпись — видео",
                 loc="left", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="best")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "fig_pca_windows.png"), dpi=160)
    plt.close(fig)

    # loadings printout for interpretation
    load = pd.DataFrame(Vt[:2].T, index=used, columns=["PC1", "PC2"]).round(2)
    print("\nPCA loadings:\n", load.sort_values("PC1").to_string())
    Wd.to_csv(os.path.join(adir, "kin_windows_clustered.csv"), index=False)

    # ---------- dendrogram over tracks ----------
    Td = tr.dropna(subset=["freq_hz"]).copy()
    used_t = [f for f in FEATS if Td[f].notna().mean() > 0.7]
    Xt = zscore(Td[used_t].fillna(Td[used_t].median()).to_numpy())
    Zt = linkage(Xt, "ward")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(Td) + 1.5))
    dendrogram(Zt, labels=(Td["video"] + " / " + Td["track"]).tolist(),
               orientation="left", ax=ax, color_threshold=0, above_threshold_color=BASE)
    ax.set_title("Иерархическая кластеризация треков (Ward, z-признаки)", loc="left", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(False)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "fig_dendrogram.png"), dpi=160)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
