"""Load and preprocess a nodule-centred 3D CT ROI in Z,Y,X order."""
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.ndimage import zoom
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

def load_3d_image(path: Path):
    """Return: arr (D,H,W) float32, spacing_zyx (sz,sy,sx) or None."""
    ext = path.suffix.lower()
    if ext == ".gz":
        ext = "".join(path.suffixes[-2:]).lower()  # .nii.gz

    if ext in [".nii", ".nii.gz"]:
        if nib is None:
            raise RuntimeError("Need nibabel for NIfTI: pip install nibabel")
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)  # (X,Y,Z)
        zooms_xyz = img.header.get_zooms()[:3]  # (sx,sy,sz)
        arr = np.transpose(arr, (2, 1, 0))  # (Z,Y,X) = (D,H,W)
        spacing_zyx = (float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0]))
        return arr.astype(np.float32), spacing_zyx

    if ext == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr, None

    if ext in [".mha", ".mhd"]:
        if sitk is None:
            raise RuntimeError("Need SimpleITK for MHA/MHD: pip install SimpleITK")
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # (D,H,W)
        sp_xyz = img.GetSpacing()  # (sx,sy,sz)
        spacing_zyx = (float(sp_xyz[2]), float(sp_xyz[1]), float(sp_xyz[0]))
        return arr.astype(np.float32), spacing_zyx

    raise ValueError(f"Unsupported volume format: {path.name}")

def clip_and_normalize_hu(arr: np.ndarray, hu_min=-1000.0, hu_max=400.0) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=hu_max, neginf=hu_min).astype(np.float32)
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min + 1e-8)  # -> [0,1]
    return arr.astype(np.float32)

def resample_to_spacing(arr: np.ndarray,
                        spacing_zyx: Tuple[float, float, float],
                        out_spacing_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        order: int = 1) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    oz, oy, ox = out_spacing_zyx
    zf = (sz / oz, sy / oy, sx / ox)
    out = zoom(arr, zf, order=order)
    return out.astype(np.float32)

def center_crop_or_pad(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    td, th, tw = target_shape
    d, h, w = arr.shape

    pd0 = max(0, (td - d) // 2);
    pd1 = max(0, td - d - pd0)
    ph0 = max(0, (th - h) // 2);
    ph1 = max(0, th - h - ph0)
    pw0 = max(0, (tw - w) // 2);
    pw1 = max(0, tw - w - pw0)
    if pd0 or pd1 or ph0 or ph1 or pw0 or pw1:
        arr = np.pad(arr, ((pd0, pd1), (ph0, ph1), (pw0, pw1)),
                     mode="constant", constant_values=0.0)

    d, h, w = arr.shape
    sd = (d - td) // 2
    sh = (h - th) // 2
    sw = (w - tw) // 2
    arr = arr[sd:sd + td, sh:sh + th, sw:sw + tw]
    return arr.astype(np.float32)


def preprocess_roi(path, vol_size=48, iso_spacing=1.0, use_spacing=True,
                   hu_min=-1000., hu_max=400., interpolation_order=3):
    """Preserve the provided nodule-centred ROI; no lesion detection is performed.

    NIfTI input orientation follows the header voxel axes and the original script.
    An externally provided ROI must already be centred on the target lesion.
    """
    vol, spacing = load_3d_image(Path(path))
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}: {path}")
    vol = clip_and_normalize_hu(vol, hu_min=hu_min, hu_max=hu_max)
    if use_spacing and spacing is not None:
        vol = resample_to_spacing(vol, spacing, (iso_spacing,)*3, order=interpolation_order)
    return center_crop_or_pad(vol, (vol_size,)*3)
