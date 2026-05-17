# Prostate MRI PI-RADS Predictor

Built for **Uncommon Hacks 2026**.

This project is a research prototype that predicts a patient-level **PI-RADS risk score** from prostate MRI slices. The goal is to support earlier, more consistent prostate cancer risk assessment from MRI, while keeping the model interpretable enough to explain what it is seeing.

## Team

- Sabyasachi Chowdhury — sabyasachic@uchicago.edu
- Sameeksha Kini — sameekshakini@uchicago.edu
- Jenitha Patel — jenithapatel@uchicago.edu
- Akshaj Chandwani — akshajchandwani@uchicago.edu

## Problem Statement

Every year, many men are sent for prostate biopsy after elevated PSA screening or suspicious MRI findings. The clinical problem is that biopsy is invasive, expensive, stressful, and often unnecessary. In the VERDICT MRI study, Singh et al. (2022) reported that multiparametric MRI can produce a high false-positive burden, meaning many suspicious-looking cases do not turn out to be clinically significant cancer.

Our project addresses that gap with an AI-assisted MRI risk scoring workflow:

```text
prostate MRI images -> slice-level risk estimates -> patient-level PI-RADS prediction
```

Instead of relying only on visual image appearance, the model learns compact MRI representations from multiple sequences: T2, ADC, and calculated high-b-value diffusion imaging. These sequences capture complementary biological signals:

- **T2-weighted MRI** shows prostate anatomy and gland structure.
- **ADC** reflects water diffusion restriction, often linked to dense or suspicious tissue.
- **BVAL/B1500 diffusion imaging** highlights areas with strong diffusion-weighted signal.

The model then predicts PI-RADS 1-5 probabilities for each slice and reports the highest-risk patient-level estimate. This makes the tool a triage-style research prototype: it is not replacing a radiologist, but it shows how AI could help prioritize suspicious exams and reduce unnecessary follow-up.

## Why This Matters

Current prostate MRI interpretation depends on expert radiology review. That creates three practical challenges:

- **False positives** can lead to unnecessary biopsies.
- **Reader variability** can make risk scoring inconsistent across settings.
- **Access bottlenecks** can delay review where expert prostate MRI readers are limited.

Our prototype explores whether a physics-aware variational model can learn a stable representation of prostate MRI and convert it into PI-RADS risk probabilities.

## What The App Does

The app expects three aligned MRI channels per slice:

```text
AX_T2
AX_DIFFUSION_ADC
AX_DIFFUSION_CALC_BVAL / B1500
```

It preprocesses the channels into a 3-channel image tensor, predicts slice-level PI-RADS probabilities, and reports the patient-level score from the highest-risk slice.

Output includes:

- predicted patient PI-RADS score
- model confidence
- selected highest-risk slice
- class probability table
- slice-level summary table

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
