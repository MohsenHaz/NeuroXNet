"""DICOM reading: primary axial non-contrast series -> HU volume ordered cranio-caudally."""
import numpy as np, pydicom, pathlib

def _is_axial(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return True
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    return abs(np.cross(r, c)[2]) > 0.9

def read_series(series_dir):
    """Return (volume_hu [S,H,W], slice_thickness_mm, spacing). Slices sorted by ImagePositionPatient z, ascending (caudal -> cranial)."""
    files = sorted(pathlib.Path(series_dir).glob("*.dcm")) or sorted(pathlib.Path(series_dir).iterdir())
    dss = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData") or not _is_axial(ds):
            continue
        dss.append(ds)
    if not dss:
        raise ValueError(f"no axial DICOM slices in {series_dir}")
    dss.sort(key=lambda d: float(d.ImagePositionPatient[2]) if hasattr(d, "ImagePositionPatient") else float(d.InstanceNumber))
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) +
                    float(getattr(d, "RescaleIntercept", 0)) for d in dss])
    thick = float(getattr(dss[0], "SliceThickness", 0) or 0)
    spacing = [float(x) for x in getattr(dss[0], "PixelSpacing", [1.0, 1.0])]
    return vol, thick, spacing

def select_primary_series(study_dir):
    """Pick the primary axial non-contrast head series: the series with the most axial slices whose description does not indicate reformat or contrast."""
    best, best_n = None, -1
    for sd in sorted(pathlib.Path(study_dir).iterdir()):
        if not sd.is_dir():
            continue
        n = 0
        for f in sd.iterdir():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                continue
            desc = str(getattr(ds, "SeriesDescription", "")).lower()
            if any(k in desc for k in ("cor", "sag", "reformat", "mpr", "contrast", "c+", "bone")):
                n = -1; break
            if _is_axial(ds):
                n += 1
        if n > best_n:
            best, best_n = sd, n
    return best
