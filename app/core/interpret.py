from __future__ import annotations

from .config import PIRADS_DESCRIPTIONS


def build_interpretation(
    pirads: int,
    confidence: float,
    slice_count: int,
    selected_slice: int,
    expected_score: float,
) -> str:
    level, description = PIRADS_DESCRIPTIONS[pirads]
    clinically_significant_flag = (
        "This falls above the PI-RADS > 2 research threshold used for clinically significant risk."
        if pirads > 2
        else "This falls at or below the PI-RADS > 2 research threshold used for clinically significant risk."
    )

    return (
        f"### PI-RADS {pirads}: {level}\n"
        f"The model-selected slice was {selected_slice} of {slice_count}. "
        f"Its expected PI-RADS score was {expected_score:.2f}, with {confidence:.1%} probability "
        f"assigned to PI-RADS {pirads}.\n\n"
        f"Interpretation: the model estimates that {description}. {clinically_significant_flag}\n\n"
        "Research/demo output only. This is not a diagnosis and must be reviewed by a qualified radiologist."
    )
