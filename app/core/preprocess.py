from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .config import CENTER_CROP_FRACTION, EXPECTED_SEQUENCES, INPUT_SIZE


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _as_path(uploaded_file) -> Path:
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    name = getattr(uploaded_file, "name", None)
    if name:
        return Path(name)
    raise TypeError(f"Unsupported upload object: {type(uploaded_file)!r}")


def _visible_files(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.is_file() and not p.name.startswith(".")]


def expand_uploads(uploaded_files, work_dir: Path) -> list[Path]:
    if uploaded_files is None:
        return []
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    expanded: list[Path] = []
    for item in uploaded_files:
        path = _as_path(item)
        if path.is_dir():
            expanded.extend(_visible_files(path.rglob("*")))
            continue
        if path.suffix.lower() == ".zip":
            extract_dir = work_dir / f"zip_{len(expanded)}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as zip_ref:
                zip_ref.extractall(extract_dir)
            expanded.extend(_visible_files(extract_dir.rglob("*")))
            continue
        if path.is_file() and not path.name.startswith("."):
            expanded.append(path)
    return expanded


def _metadata_text(path: Path) -> str:
    try:
        import pydicom

        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return ""
    fields = [
        getattr(ds, "SeriesDescription", ""),
        getattr(ds, "ProtocolName", ""),
        getattr(ds, "SequenceName", ""),
    ]
    return " ".join(str(field).upper() for field in fields)


def infer_sequence(path: Path) -> str | None:
    text = " ".join(part.upper() for part in path.parts)
    text = f"{text} {_metadata_text(path)}"

    if "ADC" in text:
        return "AX_DIFFUSION_ADC"
    if "CALC_BVAL" in text or "BVAL" in text or "B1500" in text or "B-1500" in text:
        return "AX_DIFFUSION_CALC_BVAL"
    if "AX_T2" in text or "T2" in text:
        return "AX_T2"
    return None


def group_sequence_files(paths: list[Path]) -> dict[str, list[Path]]:
    grouped = {sequence: [] for sequence in EXPECTED_SEQUENCES}
    for path in paths:
        sequence = infer_sequence(path)
        if sequence in grouped:
            grouped[sequence].append(path)
    return grouped


def _slice_sort_key(path: Path):
    try:
        import pydicom

        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        image_position = getattr(ds, "ImagePositionPatient", None)
        if image_position is not None and len(image_position) >= 3:
            return (0, float(image_position[2]), path.name)
        instance_number = getattr(ds, "InstanceNumber", None)
        if instance_number is not None:
            return (1, int(instance_number), path.name)
    except Exception:
        pass
    return (2, path.name)


def sort_slice_files(files: list[Path]) -> list[Path]:
    return sorted(files, key=_slice_sort_key)


def _read_dicom_image(path: Path) -> np.ndarray:
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pydicom is required for DICOM input. Install the app requirements first."
        ) from exc

    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return image * slope + intercept


def _read_pil_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def read_medical_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = _read_pil_image(path)
    elif path.suffix.lower() == ".dcm":
        image = _read_dicom_image(path)
    else:
        try:
            image = _read_dicom_image(path)
        except ModuleNotFoundError:
            raise
        except Exception:
            image = _read_pil_image(path)

    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].mean(axis=-1)
    elif image.ndim == 3:
        image = image[0]
    return image.astype(np.float32)


def normalize_image(image: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(image, [q_low, q_high])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lo, hi)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def center_crop(image: np.ndarray, fraction: float = CENTER_CROP_FRACTION) -> np.ndarray:
    height, width = image.shape[:2]
    new_height = max(1, int(height * fraction))
    new_width = max(1, int(width * fraction))
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    return image[top : top + new_height, left : left + new_width]


def resize_nn(image: np.ndarray, size: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> np.ndarray:
    target_height, target_width = size
    y_idx = np.linspace(0, image.shape[0] - 1, target_height).astype(int)
    x_idx = np.linspace(0, image.shape[1] - 1, target_width).astype(int)
    return image[np.ix_(y_idx, x_idx)].astype(np.float32)


def preprocess_channel(path: Path) -> np.ndarray:
    image = read_medical_image(path)
    image = normalize_image(center_crop(image))
    return resize_nn(image)


def build_triplet_from_files(t2_path: Path, adc_path: Path, bval_path: Path) -> np.ndarray:
    channels = [preprocess_channel(path) for path in (t2_path, adc_path, bval_path)]
    return np.stack(channels, axis=0).astype(np.float32)


def build_patient_triplets(uploaded_files) -> tuple[np.ndarray, list[dict[str, str | int]]]:
    with tempfile.TemporaryDirectory(prefix="pirads_upload_") as tmp:
        paths = expand_uploads(uploaded_files, Path(tmp))
        grouped = group_sequence_files(paths)

        missing = [sequence for sequence, files in grouped.items() if not files]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                "Could not find all required sequences. Missing: "
                + missing_text
                + ". Use folder/file names or DICOM metadata containing AX_T2, ADC, and BVAL/B1500."
            )

        sorted_groups = {sequence: sort_slice_files(files) for sequence, files in grouped.items()}
        slice_count = min(len(sorted_groups[sequence]) for sequence in EXPECTED_SEQUENCES)
        if slice_count < 1:
            raise ValueError("No matched patient slices were found.")

        triplets = []
        slice_info = []
        for idx in range(slice_count):
            t2_path = sorted_groups["AX_T2"][idx]
            adc_path = sorted_groups["AX_DIFFUSION_ADC"][idx]
            bval_path = sorted_groups["AX_DIFFUSION_CALC_BVAL"][idx]
            triplets.append(build_triplet_from_files(t2_path, adc_path, bval_path))
            slice_info.append(
                {
                    "slice": idx + 1,
                    "t2_file": t2_path.name,
                    "adc_file": adc_path.name,
                    "bval_file": bval_path.name,
                }
            )

    return np.stack(triplets, axis=0).astype(np.float32), slice_info
