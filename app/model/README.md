# Model Checkpoint

This folder contains the trained app checkpoint:

```text
prostate_pirads_vae.pt
```

The checkpoint is loaded by `app/core/predict.py` through `app/core/config.py`.

To regenerate it, run:

```text
notebooks/prostrate_fastMRI_VAE.ipynb
```

The notebook uses the NYU prostate MRI labels and raw image folders, which are not included in this GitHub-ready project folder.
