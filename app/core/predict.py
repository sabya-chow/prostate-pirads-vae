from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_PATH, PIRADS_CLASS_NAMES
from .interpret import build_interpretation
from .model import build_model_from_checkpoint
from .preprocess import build_patient_triplets


class ProstatePIRADSPredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = PIRADS_CLASS_NAMES.copy()

    def _load_model(self):
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Run the updated notebook training cells to create prostate_pirads_vae.pt."
            )

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names.update(checkpoint.get("class_names", {}))
        self.model = build_model_from_checkpoint(checkpoint).to(self.device)
        self.model.eval()

    def _predict_probabilities(self, triplets: np.ndarray, batch_size: int = 16) -> np.ndarray:
        self._load_model()
        assert self.model is not None

        all_probs = []
        with torch.no_grad():
            for start in range(0, len(triplets), batch_size):
                batch = torch.from_numpy(triplets[start : start + batch_size]).float().to(self.device)
                logits = self.model.predict_logits(batch)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def predict(self, uploaded_files) -> dict:
        triplets, slice_info = build_patient_triplets(uploaded_files)
        probabilities = self._predict_probabilities(triplets)

        pirads_values = np.arange(1, 6, dtype=np.float32)
        expected_scores = probabilities @ pirads_values
        selected_idx = int(np.argmax(expected_scores))
        selected_probs = probabilities[selected_idx]

        patient_pirads = int(np.clip(np.rint(expected_scores[selected_idx]), 1, 5))
        confidence = float(selected_probs[patient_pirads - 1])
        selected_slice = int(slice_info[selected_idx]["slice"])

        class_df = pd.DataFrame(
            {
                "category": [self.class_names[i] for i in range(5)],
                "probability": [round(float(value), 4) for value in selected_probs],
            }
        )

        slice_rows = []
        for idx, info in enumerate(slice_info):
            predicted_class = int(np.argmax(probabilities[idx])) + 1
            slice_rows.append(
                {
                    "slice": info["slice"],
                    "predicted_pirads": predicted_class,
                    "expected_score": round(float(expected_scores[idx]), 3),
                    "confidence": round(float(probabilities[idx, predicted_class - 1]), 4),
                    "t2_file": info["t2_file"],
                    "adc_file": info["adc_file"],
                    "bval_file": info["bval_file"],
                }
            )
        slice_df = pd.DataFrame(slice_rows).sort_values("expected_score", ascending=False)

        summary = (
            f"## Predicted PI-RADS {patient_pirads}\n"
            f"Selected slice: {selected_slice} / {len(slice_info)}  \n"
            f"Model confidence for PI-RADS {patient_pirads}: {confidence:.1%}"
        )

        return {
            "pirads": patient_pirads,
            "summary": summary,
            "interpretation": build_interpretation(
                pirads=patient_pirads,
                confidence=confidence,
                slice_count=len(slice_info),
                selected_slice=selected_slice,
                expected_score=float(expected_scores[selected_idx]),
            ),
            "class_probabilities": class_df,
            "slice_summary": slice_df,
        }
