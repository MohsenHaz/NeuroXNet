"""Deterministic mid-volume slice window (Section 3.2 of the manuscript)."""
import numpy as np

def select_window(volume, n_slices=128):
    """Select `n_slices` consecutive axial slices centred on floor(S/2).
    If S < n_slices the volume is zero-padded symmetrically (extra slice goes to the end when the deficit is odd).
    Returns (window [n_slices,H,W], pad_mask [n_slices] bool: True for real slices)."""
    S = volume.shape[0]
    if S >= n_slices:
        c = S // 2
        start = c - n_slices // 2
        start = max(0, min(start, S - n_slices))
        win = volume[start:start + n_slices]
        mask = np.ones(n_slices, bool)
    else:
        deficit = n_slices - S
        before = deficit // 2
        after = deficit - before
        pad = [(before, after)] + [(0, 0)] * (volume.ndim - 1)
        win = np.pad(volume, pad, mode="constant", constant_values=0)
        mask = np.zeros(n_slices, bool); mask[before:before + S] = True
    return win, mask
