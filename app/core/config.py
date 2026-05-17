from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_ROOT / "model"
MODEL_PATH = MODEL_DIR / "prostate_pirads_vae.pt"

EXPECTED_SEQUENCES = (
    "AX_T2",
    "AX_DIFFUSION_ADC",
    "AX_DIFFUSION_CALC_BVAL",
)

INPUT_SIZE = 128
CENTER_CROP_FRACTION = 0.88

PIRADS_CLASS_NAMES = {
    0: "PI-RADS 1",
    1: "PI-RADS 2",
    2: "PI-RADS 3",
    3: "PI-RADS 4",
    4: "PI-RADS 5",
}

PIRADS_DESCRIPTIONS = {
    1: ("Very low", "clinically significant cancer is highly unlikely to be present"),
    2: ("Low", "clinically significant cancer is unlikely to be present"),
    3: ("Intermediate", "clinically significant cancer is equivocal"),
    4: ("High", "clinically significant cancer is likely to be present"),
    5: ("Very high", "clinically significant cancer is highly likely to be present"),
}
