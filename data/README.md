# Data

Raw NYU fastMRI data is not included in this GitHub-ready project folder.

The project used the NYU fastMRI prostate MRI dataset:

```text
https://fastmri.med.nyu.edu/
```

Access requires applying through NYU fastMRI and accepting the Dataset Sharing Agreement.

Expected local raw-data folders, if retraining:

```text
data/raw/DICOMS/
data/raw/PROSTATE_MRI/
```

The app itself does not need the training dataset. It only needs uploaded patient files and the checkpoint in `app/model/`.
