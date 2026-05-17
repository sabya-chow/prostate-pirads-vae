from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "app" / "sample_uploads"
WORK_DIR = OUT_DIR / "synthetic_patient_demo"
ZIP_PATH = OUT_DIR / "synthetic_patient_demo.zip"

SEQUENCES = (
    "AX_T2",
    "AX_DIFFUSION_ADC",
    "AX_DIFFUSION_CALC_BVAL",
)


def normalize(image: np.ndarray) -> np.ndarray:
    image = image - image.min()
    denom = image.max() or 1.0
    return (255 * image / denom).astype(np.uint8)


def make_slice(sequence: str, slice_idx: int, size: int = 192) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]

    prostate = np.exp(-((x / 0.62) ** 2 + (y / 0.42) ** 2) * 2.4)
    center_zone = np.exp(-((x / 0.28) ** 2 + (y / 0.22) ** 2) * 3.0)

    lesion_x = -0.22 + 0.04 * slice_idx
    lesion_y = 0.10
    lesion = np.exp(-(((x - lesion_x) / 0.13) ** 2 + ((y - lesion_y) / 0.09) ** 2) * 5.0)

    rng = np.random.default_rng(2026 + slice_idx)
    noise = rng.normal(0, 0.03, size=(size, size))

    if sequence == "AX_T2":
        image = 0.20 + 0.72 * prostate - 0.28 * lesion + 0.10 * center_zone + noise
    elif sequence == "AX_DIFFUSION_ADC":
        image = 0.18 + 0.58 * prostate - 0.42 * lesion + 0.08 * center_zone + noise
    else:
        image = 0.12 + 0.45 * prostate + 0.55 * lesion + 0.04 * center_zone + noise

    image *= (prostate > 0.05)
    return normalize(image)


def main() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    for sequence in SEQUENCES:
        seq_dir = WORK_DIR / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)
        for slice_idx in range(1, 6):
            image = make_slice(sequence, slice_idx)
            Image.fromarray(image, mode="L").save(seq_dir / f"slice_{slice_idx:03d}.png")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(WORK_DIR.rglob("*.png")):
            zf.write(file_path, file_path.relative_to(OUT_DIR))

    print(f"Created {ZIP_PATH}")


if __name__ == "__main__":
    main()
