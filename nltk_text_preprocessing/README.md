# NLTK Text Preprocessing (IMDB Movie Reviews)

This project cleans movie reviews with the Natural Language Toolkit (`nltk`). It removes HTML, special characters and stop words, and it reduces words to stems.

You learn the standard text preprocessing steps that come before a language model or a sentiment classifier.

The code comes from the notebook *Text Preprocessing using NLTK* (Advanced Certification Programme in Agentic and Generative AI, IISc and TalentSprint).

---

## 1. Prerequisites

| Item | Version / detail |
|---|---|
| Python | 3.10 or later |
| Git | Any recent version |
| VS Code | With the **Python** extension (Microsoft) |
| VS Code extension (optional) | **Markdown Preview Mermaid Support**, to see the diagrams in `knowledge.md` |
| Internet connection | Needed one time, to install packages and NLTK data |
| API key | Not needed. This project uses no paid service. |

---

## 2. Project Structure

```
nltk_text_preprocessing/
├── README.md                  # This file
├── knowledge.md               # Theory behind each step
├── CHANGES.md                 # Differences from the notebook
├── requirements.txt           # Third-party packages
├── pyproject.toml             # Makes the project installable
├── .env.example               # Template for optional settings
├── .gitignore                 # Files that Git ignores
├── main.py                    # Entry point: runs all stages
├── config/
│   ├── __init__.py
│   └── settings.py            # Paths, constants, flags
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Shared log format
│   └── setup_resources.py     # Downloads NLTK data files
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Reads the CSV
│   ├── eda.py                 # Summary and bar chart
│   ├── text_cleaning.py       # HTML, brackets, special characters
│   ├── normalization.py       # Lemmatization and stemming
│   ├── tokenization.py        # Tokens and POS tags
│   ├── stopword_removal.py    # Stop words (keeps negations)
│   └── pipeline.py            # Demos and the full cleaning pipeline
├── data/
│   ├── README.md              # Data manifest
│   ├── raw/                   # Input CSV (placeholder now)
│   └── processed/             # Cleaned CSV (generated)
├── outputs/
│   └── figures/               # Charts (generated)
└── tests/
    ├── __init__.py
    └── test_smoke.py          # Smoke tests
```

| Folder / File | Purpose | Example content |
|---|---|---|
| `config/` | Keeps all settings in one place | `DATASET_PATH`, `SPECIAL_CHAR_PATTERN` |
| `utils/` | Keeps helpers that are not specific to text cleaning | `get_logger()`, `ensure_nltk_resources()` |
| `src/` | Keeps the text-processing logic | `strip_html()`, `simple_stemmer()` |
| `data/raw/` | Keeps the input data | `IMDB_Dataset.csv` |
| `data/processed/` | Keeps the cleaned data | `IMDB_Dataset_preprocessed.csv` |
| `outputs/figures/` | Keeps the charts | `sentiment_distribution.png` |
| `tests/` | Keeps the tests | `test_smoke.py` |
| `main.py` | Runs the stages in order | `python main.py` |

**NOTE:** This project has no `database/` folder and no LLM client. The notebook uses no database and no API.

---

## 3. Setup Procedure

Do these steps one time.

### 3.1 Get the code

1. Open a terminal.
2. Clone the repository:
   ```bash
   git clone <your-repository-url>
   ```
3. Go into the project folder:
   ```bash
   cd nltk_text_preprocessing
   ```
4. Open the folder in VS Code:
   ```bash
   code .
   ```

### 3.2 Create and activate a virtual environment

1. Create the virtual environment:

   | Operating system | Command |
   |---|---|
   | Windows | `python -m venv .venv` |
   | macOS / Linux | `python3 -m venv .venv` |

2. Activate the virtual environment:

   | Operating system | Command |
   |---|---|
   | Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
   | Windows (Command Prompt) | `.venv\Scripts\activate.bat` |
   | macOS / Linux | `source .venv/bin/activate` |

3. Make sure that the prompt starts with `(.venv)`.

**CAUTION:** On Windows, PowerShell can block the activation script. If this occurs, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

### 3.3 Select the interpreter in VS Code

1. Push `Ctrl+Shift+P` (macOS: `Cmd+Shift+P`).
2. Type `Python: Select Interpreter`.
3. Select the interpreter in `.venv`.

### 3.4 Install the packages

1. Install the third-party packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the project in editable mode:
   ```bash
   pip install -e .
   ```

### 3.5 Download the NLTK data files

1. Run this command:
   ```bash
   python -m utils.setup_resources
   ```
2. Make sure that the last line says `All NLTK resources are ready.`

**NOTE:** `main.py` also does this step automatically. The step downloads only the missing files.

### 3.6 Make the settings file (optional)

1. Copy the template:

   | Operating system | Command |
   |---|---|
   | Windows | `copy .env.example .env` |
   | macOS / Linux | `cp .env.example .env` |

2. Change the values in `.env` if necessary. All values are optional.

### 3.7 Replace the placeholder data

**CAUTION:** `data/raw/IMDB_Dataset.csv` contains 20 made-up reviews. Results from this file are not real results.

1. Read `data/README.md`.
2. Put the real `IMDB_Dataset.csv` in `data/raw/`.
3. Delete `data/raw/IMDB_Dataset.PLACEHOLDER.md`.

---

## 4. Run Procedure

### 4.1 Run the full pipeline

```bash
python main.py
```

**CAUTION:** With the real dataset (50,000 reviews), the stemming step takes 2 to 3 minutes. To test quickly, use `--sample`.

### 4.2 Run one stage

| Command | What it does |
|---|---|
| `python main.py --step load` | Reads the CSV and shows the first 10 rows |
| `python main.py --step eda` | Shows the summary and saves the bar chart |
| `python main.py --step demo` | Runs each technique on the notebook example sentences |
| `python main.py --step preprocess` | Cleans every review and saves the result |
| `python main.py --sample 2000` | Uses only the first 2,000 reviews |
| `python main.py --show-plots` | Opens the chart in a window as well |

### 4.3 Run one module alone

Each `src` module has a small demo:

```bash
python -m src.text_cleaning
python -m src.normalization
python -m src.tokenization
python -m src.stopword_removal
```

### 4.4 Run the tests

```bash
pytest
```

The expected result is `22 passed`.

### 4.5 Expected output (short sample)

```text
[2] Load the dataset
WARNING | src.data_loader | PLACEHOLDER DATA: 'IMDB_Dataset.csv' contains 20 made-up reviews. ...
(20, 2)
...
--- simple_stemmer ---
he like lemon banana and oranges. he goe to market.
...
--- remove_stopwords ---
movie not great
...
[5] Preprocess every review
                                              review sentiment
0  thi film wa pleasant surpris act wa strong sto...  positive
1  didnt enjoy thi movi plot wa slow joke flat en...  negative
...
SUMMARY
1. setup: all NLTK resources are ready
2. load: read 20 reviews from IMDB_Dataset.csv
3. eda: class counts printed, chart saved to .../outputs/figures/sentiment_distribution.png
4. demo: ran 10 technique examples from the notebook
5. preprocess: cleaned 20 reviews, saved to .../data/processed/IMDB_Dataset_preprocessed.csv
```

---

## 5. Execution Flow

This list shows what occurs when you run `python main.py`.

1. Python starts `main.py`.
2. `main.py` imports `config/settings.py`. `settings.py` builds all paths and reads `.env`, if it exists.
3. `main.py` imports `utils/logger.py`. Each module uses `get_logger()` to write log lines.
4. **Stage 1 (setup):** `main.py` calls `utils/setup_resources.ensure_nltk_resources()`. The function downloads the missing NLTK data files.
5. **Stage 2 (load):** `main.py` calls `src/data_loader.load_dataset()`.
   1. `load_dataset()` calls `config/settings.validate_settings()`. This function checks that the CSV exists.
   2. `load_dataset()` reads `data/raw/IMDB_Dataset.csv` with `pandas`.
   3. `load_dataset()` calls `is_placeholder_data()`. If the marker file exists, it writes a WARNING.
   4. `main.py` calls `get_review()` to show review number 5.
6. **Stage 3 (eda):** `main.py` calls `src/eda.py`.
   1. `describe_dataset()` returns the summary table.
   2. `sentiment_counts()` counts the positive and negative reviews.
   3. `plot_sentiment_distribution()` writes `outputs/figures/sentiment_distribution.png`.
7. **Stage 4 (demo):** `main.py` calls `src/pipeline.run_demos()`. This function calls each technique one time on the notebook example sentences:
   - `src/text_cleaning.py`: `strip_html()`, `remove_between_square_brackets()`, `remove_special_characters()`
   - `src/normalization.py`: `lemmatize_word()`, `simple_stemmer()`
   - `src/tokenization.py`: `tokenize()`, `pos_tag_tokens()`
   - `src/stopword_removal.py`: `get_stopword_list()`, `get_updated_stopword_list()`, `remove_stopwords()`
8. **Stage 5 (preprocess):** `main.py` calls `src/pipeline.preprocess_reviews()`. This function applies 4 steps to each review, in this order:
   1. `src/text_cleaning.denoise_text()`
   2. `src/text_cleaning.remove_special_characters()`
   3. `src/normalization.simple_stemmer()`
   4. `src/stopword_removal.remove_stopwords()`. This function calls `src/tokenization.tokenize()`.
9. `main.py` calls `src/pipeline.save_processed()`. This function writes `data/processed/IMDB_Dataset_preprocessed.csv`.
10. `main.py` prints a numbered summary.

`tests/test_smoke.py` runs only when you run `pytest`. It imports each module and checks the notebook outputs.

---

## 6. File Reference

| File | Main functions | Called by | Calls | Reads / Writes |
|---|---|---|---|---|
| `main.py` | `main()`, `parse_args()`, `stage_load()`, `stage_eda()`, `stage_demo()`, `stage_preprocess()` | You (command line), `tests/test_smoke.py` | All `src` modules, `utils/setup_resources.py` | Prints to the terminal |
| `config/settings.py` | `validate_settings()` | All modules | Nothing in the project | Reads `.env` |
| `utils/logger.py` | `get_logger()` | All modules with logs | `config/settings.py` | Writes log lines to the terminal |
| `utils/setup_resources.py` | `ensure_nltk_resources()` | `main.py`, tests | `nltk.download()` | Writes NLTK data files |
| `src/data_loader.py` | `load_dataset()`, `is_placeholder_data()`, `get_review()` | `main.py`, tests | `config/settings.validate_settings()` | Reads `data/raw/IMDB_Dataset.csv` |
| `src/eda.py` | `describe_dataset()`, `sentiment_counts()`, `plot_sentiment_distribution()` | `main.py`, tests | `seaborn`, `matplotlib` | Writes `outputs/figures/sentiment_distribution.png` |
| `src/text_cleaning.py` | `strip_html()`, `remove_between_square_brackets()`, `denoise_text()`, `remove_special_characters()` | `src/pipeline.py` | `BeautifulSoup`, `re` | Nothing |
| `src/normalization.py` | `lemmatize_word()`, `simple_stemmer()` | `src/pipeline.py` | `WordNetLemmatizer`, `PorterStemmer` | Reads NLTK `wordnet` data |
| `src/tokenization.py` | `tokenize()`, `pos_tag_tokens()` | `src/pipeline.py`, `src/stopword_removal.py` | `word_tokenize`, `nltk.pos_tag` | Reads NLTK `punkt_tab` and tagger data |
| `src/stopword_removal.py` | `get_stopword_list()`, `build_updated_stopword_list()`, `get_updated_stopword_list()`, `remove_stopwords()` | `src/pipeline.py` | `src/tokenization.tokenize()` | Reads NLTK `stopwords` data |
| `src/pipeline.py` | `run_demos()`, `preprocess_reviews()`, `save_processed()` | `main.py` | All other `src` modules | Writes `data/processed/IMDB_Dataset_preprocessed.csv` |
| `tests/test_smoke.py` | 22 tests | `pytest` | All modules | Writes temporary files only |

---

## 7. Configuration Reference

All settings are in `config/settings.py`. Some settings also come from `.env`.

| Name | Default | Purpose | Safe to change? |
|---|---|---|---|
| `PROJECT_ROOT` | Folder of the project | Base for all paths | No |
| `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR` | `data/`, `data/raw/`, `data/processed/` | Data folders | No |
| `OUTPUT_DIR`, `FIGURES_DIR` | `outputs/`, `outputs/figures/` | Output folders | No |
| `DATASET_FILENAME` | `IMDB_Dataset.csv` | Name of the input file | Yes, if your file has a different name |
| `DATASET_PATH` | `data/raw/IMDB_Dataset.csv` | Full path of the input file | Use `--data` instead |
| `PLACEHOLDER_MARKER_PATH` | `data/raw/IMDB_Dataset.PLACEHOLDER.md` | Marker that shows placeholder data | No |
| `PROCESSED_DATASET_PATH` | `data/processed/IMDB_Dataset_preprocessed.csv` | Output CSV | Yes |
| `SENTIMENT_PLOT_PATH` | `outputs/figures/sentiment_distribution.png` | Output chart | Yes |
| `DATASET_SOURCE_URL` | Stanford dataset page | Reference only | Yes |
| `TEXT_COLUMN` | `review` | Column with the text | Only if your CSV is different |
| `LABEL_COLUMN` | `sentiment` | Column with the label | Only if your CSV is different |
| `PREVIEW_ROWS` | `10` | Rows shown by the load stage | Yes |
| `SAMPLE_REVIEW_INDEX` | `5` | Review shown by the load stage | Yes |
| `NLTK_RESOURCES` | 6 names | NLTK data files to download | No |
| `NLTK_DATA_DIR` | `NLTK_DATA` in `.env`, or NLTK default | Folder for NLTK data | Yes |
| `SPECIAL_CHAR_PATTERN` | `[^a-zA-Z0-9\s]` | Characters to remove | Yes, for experiments |
| `STOPWORD_LANGUAGE` | `english` | Stop-word list language | Yes |
| `NEGATION_WORD` | `not` | Stop word to keep | Yes, for experiments |
| `NEGATION_SUFFIX` | `n't` | Stop words with this ending stay | Yes, for experiments |
| `DEMO_*` (8 values) | Notebook example sentences | Inputs for the demo stage | Yes |
| `SHOW_PLOTS` | `false` (`.env`) | Open the chart in a window | Yes |
| `SAMPLE_SIZE` | empty = all rows (`.env`) | Use only the first N reviews | Yes |
| `LOG_LEVEL` | `INFO` (`.env`) | Amount of log output | Yes |

---

## 8. Data Status

| File | Status | Action |
|---|---|---|
| `data/raw/IMDB_Dataset.csv` | ⚠️ PLACEHOLDER (20 made-up rows) | Replace it with the real 50,000-row file. Then delete `data/raw/IMDB_Dataset.PLACEHOLDER.md`. See `data/README.md`. |

---

## 9. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `ModuleNotFoundError: No module named 'nltk'` (or `pandas`, `bs4`) | The packages are not installed in the active environment. | Activate `.venv`. Then run `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'config'` (or `src`) | Python runs from a different folder. | Run commands from the project folder. Run `pip install -e .`. |
| VS Code shows yellow lines under imports | VS Code uses the wrong interpreter. | Select the `.venv` interpreter (section 3.3). |
| `LookupError: Resource punkt_tab not found` (or `stopwords`, `wordnet`) | The NLTK data files are missing. | Run `python -m utils.setup_resources`. |
| `Missing NLTK resources: ...` | The download failed. | Check the internet connection. Then run `python -m utils.setup_resources` again. |
| `ERROR: Dataset not found` | The CSV is not in `data/raw/`. | Put `IMDB_Dataset.csv` in `data/raw/`. |
| `WARNING ... PLACEHOLDER DATA` | The placeholder marker file exists. | Replace the CSV. Then delete `data/raw/IMDB_Dataset.PLACEHOLDER.md`. |
| `ValueError: ... must have the columns` | The CSV has different column names. | Rename the columns to `review` and `sentiment`. |
| The run stops for minutes at `simple_stemmer` | Stemming 50,000 reviews takes time. | Wait 2 to 3 minutes, or use `--sample 2000`. |
| `pip` shows a version conflict | Old packages are in the environment. | Delete `.venv`. Do sections 3.2 and 3.4 again. |
| No chart window opens | `SHOW_PLOTS` is `false` by default. | Use `--show-plots`. The PNG file is always saved. |
| PowerShell blocks `Activate.ps1` | The execution policy blocks scripts. | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |

---

## 10. Glossary

| Term | Definition |
|---|---|
| Corpus | A large collection of text. NLTK gives many corpora, for example the stop-word lists. |
| DataFrame | A `pandas` table with rows and named columns. |
| EDA (Exploratory Data Analysis) | A first look at a dataset: size, values, and class counts. |
| HTML tag | A markup code in angle brackets, for example `<br />`. It is not part of the visible text. |
| Lemma | The dictionary form of a word. The lemma of "better" (adjective) is "good". |
| Lemmatization | The process that changes a word to its lemma. It uses a dictionary (WordNet). |
| Negation | A word that reverses the meaning, for example "not" or "didn't". |
| NLTK | Natural Language Toolkit. A Python library for work with human language. |
| Noise | Text that has no value for the task, for example HTML tags. |
| POS tag | Part-of-speech tag. A label for the word class, for example `NN` (noun) or `VBG` (verb, -ing form). |
| Regular expression | A pattern that matches a set of strings. Python uses the `re` module for it. |
| Sentiment | The opinion in a text: positive or negative. |
| Special character | A character that is not a letter, a digit or whitespace, for example `!` or `?`. |
| Stem | The part of a word that stays after a rule removes the suffix. The stem of "likes" is "like". |
| Stemming | The process that cuts word endings by fixed rules. It does not use a dictionary. |
| Stop word | A very common word with little meaning, for example "the", "is", "he". |
| Token | One unit of text after tokenization: a word, a number or a punctuation mark. |
| Tokenization | The process that splits text into tokens. |
| Virtual environment | A folder with its own Python and packages, separate from other projects. |
| WordNet | A large English dictionary database that NLTK uses for lemmatization. |

---

## 11. Next Steps

1. Read `knowledge.md`. It explains the theory behind each step, with diagrams and examples.
2. Read `CHANGES.md`. It lists each difference from the original notebook.
3. Try a change: move `remove_stopwords` before `simple_stemmer` in `src/pipeline.py` (`PREPROCESSING_STEPS`). Then compare the output. `knowledge.md` section 10 explains the result.
