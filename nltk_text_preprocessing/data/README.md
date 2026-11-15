# Data Manifest

This file lists each data file that the project reads or writes.

## Input and output files

| File | Used by (module.function) | Source (from notebook) | Status | Action required |
|---|---|---|---|---|
| `raw/IMDB_Dataset.csv` | `src/data_loader.load_dataset()` | Hidden Colab cell "Download the IMDB Movie Review dataset". The notebook links to the Stanford Large Movie Review Dataset page. | ⚠️ PLACEHOLDER (20 made-up rows) | Replace with the real 50,000-row file. Then delete `raw/IMDB_Dataset.PLACEHOLDER.md`. |
| `raw/IMDB_Dataset.PLACEHOLDER.md` | `src/data_loader.is_placeholder_data()` | Created for this package | Marker file | Delete it after you replace the CSV. |
| `processed/IMDB_Dataset_preprocessed.csv` | `src/pipeline.save_processed()` | Created by `python main.py` | Generated (git-ignored) | No action. |

## Required format of `IMDB_Dataset.csv`

| Column | Type | Values |
|---|---|---|
| `review` | text | The full review. It can contain HTML such as `<br />`. |
| `sentiment` | text | `positive` or `negative` |

The real file has 50,000 rows: 25,000 `positive` and 25,000 `negative`.

## How to get the real file

The notebook downloads the CSV in a hidden cell. The PDF does not show the code or the URL of that cell.

1. Open the IISc / TalentSprint course page for this notebook.
2. Download `IMDB_Dataset.csv` from the course material.
3. Copy the file to `data/raw/IMDB_Dataset.csv`.
4. Delete `data/raw/IMDB_Dataset.PLACEHOLDER.md`.

**NOTE:** The notebook cites the original source: <http://ai.stanford.edu/~amaas/data/sentiment/>. That page gives the reviews as many small text files, not as one CSV. A CSV version with the same name (`IMDB_Dataset.csv`, "IMDB Dataset of 50K Movie Reviews") is also available on Kaggle.

**CAUTION:** The real file is about 63 MB. GitHub accepts files up to 100 MB, but large files make the repository slow to clone. To keep the file out of Git, remove the `#` in front of `data/raw/IMDB_Dataset.csv` in `.gitignore`.
