# PLACEHOLDER DATA — REPLACE WITH REAL FILE

The file `IMDB_Dataset.csv` in this folder is **placeholder data**.

- It has 20 made-up movie reviews (10 positive, 10 negative).
- It has the same columns as the real file: `review`, `sentiment`.
- The reviews contain HTML (`<br />`), square brackets, punctuation and negations, so every cleaning step has work to do.
- The reviews are **not** from IMDB. Do not use results from this file as real results.

## To replace the file

1. Get the real `IMDB_Dataset.csv` (50,000 rows). See `data/README.md`.
2. Put it in this folder with the same name: `data/raw/IMDB_Dataset.csv`.
3. Delete this file (`IMDB_Dataset.PLACEHOLDER.md`).

The code shows a WARNING each time it loads the data while this file exists.
