The three executed notebooks cover source inspection, preprocessing and model training/evaluation for the Hugging Face dataset recorded in the active model metadata.

Run from the repository root using Python 3.12 and the installed project requirements:

```bash
python -m scripts.download_data
python -m src.train
python -m scripts.build_notebooks
```

The notebook builder uses the current Python executable and workspace-local kernel configuration under `.tmp`.
It executes every cell, captures output, and fails if a cell fails. Notebook 03 loads the measured training artifacts;
set `RUN_TRAINING = True` in that notebook to retrain. Open notebooks with the repository root as the working directory.
Global EDA numbers are source integrity checks; feature analysis is restricted to training rows.
