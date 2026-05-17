# Prostate MRI PI-RADS Predictor

Research prototype for predicting patient-level PI-RADS from prostate MRI slices using a physics-aware variational autoencoder.

Built for **Uncommon Hacks 2026**.

## Team

- Sabyasachi Chowdhury — sabyasachic@uchicago.edu
- Sameeksha Kini — sameekshakini@uchicago.edu
- Jenitha Patel — jenithapatel@uchicago.edu
- Akshaj Chandwani — akshajchandwani@uchicago.edu

The app expects three aligned MRI channels per slice:

- `AX_T2`
- `AX_DIFFUSION_ADC`
- `AX_DIFFUSION_CALC_BVAL` / `B1500`

It preprocesses the channels into a 3-channel tensor, predicts slice-level PI-RADS probabilities, and reports the patient-level score from the highest-risk slice.

## Project Layout

```text
.
├── app/
│   ├── app.py                  # Gradio entry point
│   ├── core/                   # preprocessing, model loading, prediction logic
│   ├── model/                  # trained checkpoint and training history
│   └── sample_uploads/         # upload format notes
├── data/
│   └── README.md               # data access notes; raw NYU fastMRI data excluded
├── notebooks/
│   └── prostrate_fastMRI_VAE.ipynb
├── src/
│   └── prostate_pirads_model.py
├── requirements.txt
└── .gitignore
```

## Run The Gradio App

From this project folder:

```bash
pip install -r requirements.txt
python app/app.py
```

Then open the local Gradio URL printed in the terminal.

## Upload Format

Upload either a ZIP file or multiple files where folders, filenames, or DICOM metadata contain sequence names:

```text
patient_001/
  AX_T2/
  AX_DIFFUSION_ADC/
  AX_DIFFUSION_CALC_BVAL/
```

PNG/JPG/TIFF files can be used for a quick demo if their filenames include `T2`, `ADC`, and `BVAL` or `B1500`.

## Model Summary

The model is a convolutional VAE with a deterministic PI-RADS classifier:

```text
3-channel MRI slice
  -> CNN encoder
  -> latent distribution: mu, logvar
  -> sampled z for image reconstruction
  -> mu for stable PI-RADS classification
```

The loss combines:

- image reconstruction loss
- KL regularization on the latent distribution
- PI-RADS classification loss
- a small ADC/B1500 consistency penalty

## Data Source And Disclosure

This project uses the **NYU fastMRI prostate MRI dataset**:

```text
https://fastmri.med.nyu.edu/
```

The fastMRI prostate data contains prostate MRI exams acquired on 3T scanners, including axial T2-weighted and diffusion-weighted imaging. Access requires accepting the NYU fastMRI Dataset Sharing Agreement.

Important disclosure:

- Raw fastMRI data is **not included** in this repository.
- Dataset files, download links, and derived patient-level data should not be redistributed through this repository.
- Anyone who wants to reproduce training should apply for fastMRI access individually through NYU and follow the fastMRI Dataset Sharing Agreement.
- This project is for research/educational hackathon use only.

Suggested acknowledgement:

```text
Data used in this project were obtained from the NYU fastMRI Initiative database (https://fastmri.med.nyu.edu/).
```

## Disclaimer

This is a research and educational prototype only. It is not a diagnostic medical device and must not be used for clinical decision-making.
