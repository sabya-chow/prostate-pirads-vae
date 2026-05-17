# GitHub Upload Checklist

Before pushing this repository publicly:

- Confirm no raw fastMRI DICOM/H5/image files are included.
- Confirm no fastMRI download links or access credentials are included.
- Confirm `data/labels/*.csv` is not committed unless redistribution is allowed for your use case.
- Confirm the model checkpoint does not contain patient-identifiable data.
- Keep the NYU fastMRI acknowledgement in `README.md`.
- Keep the clinical disclaimer in `README.md` and `DATA_DISCLOSURE.md`.

Useful check:

```bash
find . -type f | sort
```

Look for files ending in:

```text
.dcm
.h5
.nii
.nii.gz
```

These should not be committed.
