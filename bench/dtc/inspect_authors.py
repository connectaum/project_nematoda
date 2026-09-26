import h5py, numpy as np, cv2, time
rng = np.random.default_rng(1); t0 = time.time()
h = h5py.File('/work/authors/Final_mixed_train_data_wt_hand_ann.hdf5', 'r'); X, Y = h['x_train'], h['y_train']; ids = list(X.keys())
ours = h5py.File('/work/dtc_train.hdf5', 'r')
tiles = []; nlab = []; lens = []
def bright(x, pts):  # mean pixel value at label points, (x,y) vs (y,x)
    p = np.round(pts).astype(int); ok = (p >= 0).all(1) & (p < 512).all(1); p = p[ok]
    return x[p[:, 1], p[:, 0]].mean(), x[p[:, 0], p[:, 1]].mean()
for n, i in enumerate(rng.choice(len(ids), 12, replace=False)):
    k = ids[i]; x = np.asarray(X[k], np.float32); y = np.asarray(Y[k], np.float32)
    nlab.append(len(y)); L = np.hypot(*np.diff(y[:, 1], axis=1).T).sum(0) if len(y) else []; lens += list(L)
    bxy = [bright(x[f], y[:, j].reshape(-1, 2)) for f, j in ((4, 0), (5, 1), (6, 2), (0, 1), (10, 1))] if len(y) else []
    print(k, x.dtype, x.shape, 'x range %.3f..%.3f mean %.4f frac>0.05 %.3f' % (x.min(), x.max(), x.mean(), (x > 0.05).mean()), 'y', y.shape, 'y range %.1f..%.1f' % (y.min(), y.max()) if len(y) else '', 'L', np.round(L, 0)[:6], 'bright(xy,yx) f4/5/6/0/10', np.round(bxy, 2).tolist(), flush=True)
    if n < 8:
        t = cv2.cvtColor((np.clip(x[5], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for j in range(len(y)):
            p = np.round(y[j, 1]).astype(np.int32).reshape(-1, 1, 2); cv2.polylines(t, [p], False, (0, 255, 0), 1); cv2.circle(t, tuple(p[0, 0]), 3, (255, 0, 0), -1)
        cv2.putText(t, f'authors {k} n={len(y)}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); tiles.append(t)
print('n labels/clip', nlab, 'mean L px', np.mean(lens), 'L pct', np.percentile(lens, [5, 50, 95]), time.time() - t0)
# ours: 4 tiles padded to 512x512
for i in rng.choice(ours['x'].shape[0], 4, replace=False):
    x = ours['x'][i][5]; y = ours['y'][i]; t = np.zeros((512, 512, 3), np.uint8); t[:x.shape[0], :x.shape[1]] = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    for j in range(6):
        if y[j, 0, 0, 0] < 0: continue
        p = np.round(y[j, 1]).astype(np.int32).reshape(-1, 1, 2); cv2.polylines(t, [p], False, (0, 255, 0), 1); cv2.circle(t, tuple(p[0, 0]), 3, (255, 0, 0), -1)
    cv2.putText(t, f'ours {ours["meta_video"][i].decode()} x mean {x.mean():.1f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); tiles.append(t)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]; cv2.imwrite('/work/authors/sheet.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
