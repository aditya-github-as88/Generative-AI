# Data Manifest

This file lists each data file that the project reads or writes.

## Input and output files

| File | Used by (module.function) | Source | Status | Action required |
|---|---|---|---|---|
| `raw/all-data.csv` | `src/data_loader.load_dataset()` | Sentiment Analysis for Financial News (Kaggle) | Real data, 500 rows | No action. |
| `processed/all-data_preprocessed.csv` | `src/pipeline.save_processed()` | Created by `python main.py` | Generated (git-ignored) | No action. |

## Format of `all-data.csv`

The file has **no header row**. Each line has 2 columns in this order:

| Position | Name in the code | Type | Values |
|---|---|---|---|
| 1 | `sentiment` | text | `positive`, `neutral` or `negative` |
| 2 | `review` | text | One sentence from a financial news article |

Example line:

```
neutral,"According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing ."
```

**NOTE:** The file is encoded as **Latin-1**, not UTF-8. `config/settings.py` sets `DATASET_ENCODING = "latin-1"`. If you read the file as UTF-8, you get a `UnicodeDecodeError`.

**NOTE:** The code names the columns `review` and `sentiment` after it reads the file. This keeps the rest of the pipeline the same for any sentiment dataset.

The current file has 500 rows: 394 `positive`, 102 `neutral` and 4 `negative`.

## Source

Sentiment Analysis for Financial News (Financial PhraseBank), Kaggle: <https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news>

The full dataset has about 4,800 sentences. To use it, download `all-data.csv` from Kaggle and replace `data/raw/all-data.csv`. To test quickly, run `python main.py --sample 500`.
