# Sample Uploads

The Gradio app expects a ZIP or files with sequence names in folders, filenames, or DICOM metadata:

```text
patient_001/
  AX_T2/
  AX_DIFFUSION_ADC/
  AX_DIFFUSION_CALC_BVAL/
```

PNG/JPG/TIFF files can be used for a quick single-patient demo if their filenames include T2, ADC, and BVAL or B1500.

## Synthetic Demo ZIP

This folder may include:

```text
synthetic_patient_demo.zip
```

It contains fake PNG images with the expected folder names. It is only for testing that the app upload flow works.

It is **not real MRI data**, **not from NYU fastMRI**, and **not clinically meaningful**.

To regenerate it:

```bash
python scripts/create_synthetic_sample.py
```
