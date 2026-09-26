import numpy as np, trail, collections
from common import *; from methods import longest_path, crop
from skimage.morphology import skeletonize
d="results/sam2ep/IMG_8370_550-1420_s3_small"; ctx=VideoCtx("IMG_8370")
z=np.load(f"{d}/masks.npz"); M=np.unpackbits(z["masks"],axis=-1)[...,:ctx.W].astype(bool)
shown=0
for t in range(0,291,7):
    m=M[t,1]; sub,off=crop(m)
    p=longest_path(skeletonize(sub)); Ls=arclen(p)/ctx.L_med
    q=trail.longest_trail(sub,L_ref=ctx.L_med,spur=0); Lt=arclen(q)/ctx.L_med if q is not None else 0
    if Lt < Ls-0.05 and shown<3:
        shown+=1; br=trail._branches(skeletonize(sub))
        deg=collections.Counter([b["a"] for b in br]+[b["b"] for b in br])
        print(t,"skel %.2f trail %.2f"%(Ls,Lt),"branches",len(br),"lens",[round(b["L"]) for b in br],"deg",dict(deg))
        print("   ends", [(tuple(b["co"][0].astype(int)),tuple(b["co"][-1].astype(int))) for b in br])
