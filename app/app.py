from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from core.predict import ProstatePIRADSPredictor


predictor = ProstatePIRADSPredictor()


def analyze_patient(files):
    if not files:
        return (
            "## No patient images uploaded",
            "Upload a patient ZIP or DICOM files containing T2, ADC, and calculated BVAL/B1500 sequences.",
            pd.DataFrame(),
            pd.DataFrame(),
        )

    try:
        result = predictor.predict(files)
    except Exception as exc:
        return (
            "## Unable to analyze patient",
            str(exc),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    return (
        result["summary"],
        result["interpretation"],
        result["class_probabilities"],
        result["slice_summary"],
    )


with gr.Blocks(title="Prostate MRI PI-RADS Predictor") as demo:
    gr.Markdown("# Prostate MRI PI-RADS Predictor")
    with gr.Row():
        patient_files = gr.File(
            label="Patient DICOM files or ZIP",
            file_count="multiple",
            file_types=[".dcm", ".zip", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        )
        with gr.Column():
            analyze_button = gr.Button("Analyze Patient", variant="primary")
            prediction = gr.Markdown()
            interpretation = gr.Markdown()

    class_probabilities = gr.Dataframe(label="Selected Slice Probabilities", interactive=False)
    slice_summary = gr.Dataframe(label="Slice-Level Summary", interactive=False)

    analyze_button.click(
        fn=analyze_patient,
        inputs=[patient_files],
        outputs=[prediction, interpretation, class_probabilities, slice_summary],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7960")),
    )
