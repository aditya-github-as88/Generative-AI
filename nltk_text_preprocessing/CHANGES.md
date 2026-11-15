# CHANGES

This file lists each difference between the original notebook and this package.

- **Source:** `IISC_2_Text_Preprocessing_using_NLTK_M1.ipynb` (Colab PDF export, 9 pages).
- **Format of each entry:** *File → What changed → Why*.

## 1. Notebook-only code that we converted

| # | File | What changed | Why |
|---|---|---|---|
| 1 | `utils/setup_resources.py` | The six `nltk.download(...)` lines became `ensure_nltk_resources()`. The function downloads only missing resources. | A package must not download data each time a module is imported. |
| 2 | `data/raw/IMDB_Dataset.csv` | The hidden cell "Download the IMDB Movie Review dataset" became a manual step in `data/README.md`. A placeholder CSV is in its place. | The PDF shows "Show code" but not the code or the URL. We do not invent a URL. |
| 3 | `src/data_loader.py` | `pd.read_csv('IMDB_Dataset.csv')` now reads `data/raw/IMDB_Dataset.csv` through `config/settings.py`. | Colab used the current folder. A package needs a fixed path from `PROJECT_ROOT`. |
| 4 | `main.py` | `df.head(10)`, `df.describe()`, `df.head()` and other bare expressions became `print(...)`. | A bare expression shows output only in a notebook. |
| 5 | `src/eda.py` | `sns.barplot(...)` now saves the chart to `outputs/figures/sentiment_distribution.png`. A flag (`SHOW_PLOTS`, `--show-plots`) opens a window as well. | A script has no inline plot area. |
| 6 | `src/pipeline.py` | The four `df['review'] = df['review'].apply(...)` cells became `preprocess_reviews()`. It returns a new DataFrame. | No hidden global state. The input DataFrame stays unchanged. |
| 7 | `src/stopword_removal.py` | The global `updated_stopword_list` became `get_updated_stopword_list()` (cached). `remove_stopwords()` has a new optional parameter `stopword_list`. | Function parameters replace global variables. The default gives the notebook behavior. |
| 8 | `src/pipeline.py` | The cleaned data is saved to `data/processed/IMDB_Dataset_preprocessed.csv`. | In the notebook the result stayed in memory. A file lets the next tutorial use it. |
| 9 | `main.py` | The cell-by-cell flow became 5 stages with `argparse` (`--step`, `--data`, `--sample`, `--show-plots`). | One command runs the tutorial. One stage can run alone for a demo. |

## 2. Changes for speed or safety (same output)

| # | File | What changed | Why |
|---|---|---|---|
| 10 | `src/normalization.py` | `simple_stemmer()` uses one shared `PorterStemmer`. The notebook created a new stemmer for each review. | The output is the same. The run is faster on 50,000 reviews. |
| 11 | `src/stopword_removal.py` | `remove_stopwords()` checks tokens against a `set`, not a `list`. | The output is the same. A set lookup is faster. |
| 12 | `src/text_cleaning.py` | `remove_special_characters()` uses a pre-compiled regular expression. | The output is the same. |
| 13 | `src/text_cleaning.py` | `warnings.filterwarnings('ignore')` (all warnings) became a filter for `MarkupResemblesLocatorWarning` only. | To hide all warnings can hide real problems. BeautifulSoup gives only this warning on short reviews. |

## 3. Code that we removed

| # | Notebook code | Why |
|---|---|---|
| 14 | `import numpy as np` | The notebook does not use `np`. |
| 15 | `import string` | The notebook does not use `string`. |
| 16 | `nltk.porter.PorterStemmer()` inside `simple_stemmer()` | The notebook imported `PorterStemmer` but did not use the import. The package uses the import. It is the same class. |
| 17 | `df['sentiment'].value_counts().values` as a separate cell | `main.py` prints it inside the EDA stage. Nothing is lost. |

## 4. Behavior that we kept on purpose (not a defect in the code)

| # | Behavior | Where it is explained |
|---|---|---|
| 18 | Stemming runs **before** stop-word removal. So stems such as `thi`, `wa`, `ha` stay in the text, because they are not in the stop-word list. The notebook output shows this too (`"thought thi wa wonder way"`). | `knowledge.md` section 10 |
| 19 | Special-character removal runs **before** stop-word removal. So `didn't` becomes `didnt`. The negation stays in the text, but it is not the word `not`. | `knowledge.md` sections 4 and 9 |
| 20 | Lemmatization, tokenization (alone) and POS tagging are demos only. The notebook does not apply them to the dataset. | `knowledge.md` sections 5, 7, 8 |

## 5. Reconstruction from the PDF

| # | Item | Result |
|---|---|---|
| 21 | Code text | All code cells were readable. No line was reconstructed. |
| 22 | Stop-word list outputs | The PDF cut the printed lists at the page edge. This does not affect code. |
| 23 | Hidden download cell | Not visible. See entry 2. |

## 6. Assumptions

| # | Assumption |
|---|---|
| 24 | `PROJECT_NAME` = `nltk_text_preprocessing`. `TOPIC` = "Text preprocessing with NLTK". `PYTHON_VERSION` = 3.10. `LLM_PROVIDER` = none. `RUN_TARGET` = local VS Code. |
| 25 | Versions are floors (`>=`), because the notebook does not show package versions. The package was tested with nltk 3.10.4, pandas 3.0.5, seaborn 0.13.2, matplotlib 3.11.2, beautifulsoup4 4.15.0. |
| 26 | The notebook uses no API, so the package has no `database/`, `utils/llm_client.py` or `utils/tools.py`. |

## Dataset change: IMDB -> Sentiment Analysis for Financial News

The project now uses `data/raw/all-data.csv` (Sentiment Analysis for Financial News) instead of the IMDB movie review dataset. The rows above describe the original IMDB version.

| # | File | Change | Why |
|---|---|---|---|
| 1 | `config/settings.py` | `DATASET_FILENAME` is `all-data.csv`. New settings `DATASET_ENCODING` (`latin-1`), `DATASET_HAS_HEADER` (`False`) and `CSV_COLUMN_ORDER`. The output file is `all-data_preprocessed.csv`. | The new file has no header row and is not UTF-8. |
| 2 | `src/data_loader.py` | Reads the CSV with `header=None`, explicit column names and Latin-1 encoding, then reorders the columns to `review, sentiment`. | Keeps the internal column names, so the rest of the pipeline is unchanged. |
| 3 | `src/eda.py` | Chart title is "Sentiment distribution". | The new dataset has 3 classes, not 2. |
| 4 | `README.md`, `data/README.md`, `knowledge.md`, `.env.example`, `.gitignore`, `pyproject.toml`, `main.py` | IMDB references now point to `all-data.csv`. | Documentation matches the data. |
