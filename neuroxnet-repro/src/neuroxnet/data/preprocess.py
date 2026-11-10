"""Brain windowing, in-plane resize, slice selection and normalisation."""
import numpy as np
import torch, torch.nn.functional as F
from .slice_select import select_window

def brain_window(vol_hu, wl=40.0, ww=80.0):
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return (np.clip(vol_hu, lo, hi) - lo) / (hi - lo)          # -> [0,1]

def resize_inplane(vol, h=256, w=256):
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None]     # 1,1,S,H,W
    t = F.interpolate(t, size=(vol.shape[0], h, w), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()

def preprocess_volume(vol_hu, cfg):
    """HU volume -> normalised window [128,256,256] float32 and the pad mask.
    Order: window -> rescale to [0,1] -> slice selection -> in-plane resize -> (x - mean)/std. Padded slices keep the background value (0 - mean)/std, as stated in the manuscript."""
    d = cfg["data"]
    v = brain_window(vol_hu, d["window_level"], d["window_width"])
    v, mask = select_window(v, d["slices"])
    v = resize_inplane(v, d["height"], d["width"])
    v = (v - d["normalisation"]["mean"]) / d["normalisation"]["std"]
    return v.astype(np.float32), mask

def training_set_statistics(windowed_volumes):
    """Population mean and std of [0,1] windowed voxels over the training studies (logged by 01_preprocess.py)."""
    s = n = ss = 0.0
    for v in windowed_volumes:
        s += float(v.sum()); ss += float((v.astype(np.float64) ** 2).sum()); n += v.size
    mean = s / n
    std = float(np.sqrt(ss / n - mean ** 2))
    return mean, std
