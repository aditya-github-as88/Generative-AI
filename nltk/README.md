# Text Preprocessing with NLTK

## 1. Purpose

This program cleans raw text before a language model uses it.

The program uses the NLTK library to do these tasks:

- Remove HTML tags and noise from text.
- Remove special characters.
- Change words to their root form (stemming and lemmatization).
- Split text into words (tokenization).
- Find the part of speech of each word (POS tagging).
- Remove stop words.

Clean text gives better results. A model can then find the important words easily.

## 2. Where NLTK Fits in Generative AI

NLTK does not make new text. It is a support tool.

- **Before generation:** NLTK cleans and organizes the raw text. Then you send the text to the model.
- **After generation:** NLTK can examine the model output. For example, it can count keywords or find repeated words.

## 3. Input Data

The program uses a CSV file of movie reviews. The file has 50,000 rows and 2 columns:

| Column      | Description                                  |
|-------------|----------------------------------------------|
| `review`    | The text of one movie review.                |
| `sentiment` | The label of the review: `positive` or `negative`. |

The dataset has 25,000 positive reviews and 25,000 negative reviews. Thus, the dataset is balanced.

## 4. Requirements

Install these Python packages:

```bash
pip install numpy pandas matplotlib seaborn nltk beautifulsoup4
```

Download these NLTK resources one time:

```python
import nltk

nltk.download('stopwords')                       # list of stop words
nltk.download('punkt')                           # tokenizer model
nltk.download('punkt_tab')                       # tokenizer tables
nltk.download('wordnet')                         # data for lemmatization
nltk.download('averaged_perceptron_tagger')      # POS tagger
nltk.download('averaged_perceptron_tagger_eng')  # POS tagger (English)
```

## 5. Program Flow

The program does the steps in this sequence:

```
Load data -> Examine data -> Remove noise -> Remove special characters
          -> Stemming -> Remove stop words -> Clean text
```

## 6. Procedure

### Step 1: Import the packages

```python
import re
import pandas as pd
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
```

### Step 2: Load the dataset

```python
df = pd.read_csv('IMDB_Dataset.csv')
print(df.shape)      # (50000, 2)
df.head(10)
```

### Step 3: Examine the data

Make sure that the classes are balanced.

```python
df['sentiment'].value_counts()

sns.barplot(x=df['sentiment'].value_counts().index,
            y=df['sentiment'].value_counts().values,
            hue=df['sentiment'].value_counts().index)
```

### Step 4: Remove noise

Raw text can contain HTML tags and square brackets. These items do not add meaning. Remove them.

```python
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    text = text.replace('[', '')
    text = text.replace(']', '')
    return text

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

df['review'] = df['review'].apply(denoise_text)
```

Example:

| Input                                   | Output                                 |
|-----------------------------------------|----------------------------------------|
| `<h1>This is a test heading.</h1>`      | `This is a test heading.`              |
| `I bought [apple, banana, orange] yesterday.` | `I bought apple, banana, orange yesterday.` |

### Step 5: Remove special characters

Keep only letters, numbers and spaces. Remove all other characters.

```python
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

df['review'] = df['review'].apply(remove_special_characters)
```

Example: `Hi There! How are you?` becomes `Hi There How are you`.

### Step 6: Lemmatization (demonstration)

Lemmatization changes a word to its dictionary root (lemma). It uses the meaning of the word.

```python
lemmatizer = nltk.stem.WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos="a"))   # good
```

Valid values for `pos`:

| Value | Part of speech       |
|-------|----------------------|
| `n`   | Noun                 |
| `v`   | Verb                 |
| `a`   | Adjective            |
| `r`   | Adverb               |
| `s`   | Satellite adjective  |

**Note:** This step shows how lemmatization works. The program does not apply it to the dataset.

### Step 7: Stemming

Stemming removes the end of a word to get a short root form (stem). It is fast. But the stem is not always a real word.

```python
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    return ' '.join([ps.stem(word) for word in text.split()])

df['review'] = df['review'].apply(simple_stemmer)
```

Example: `He likes lemons bananas and oranges.` becomes `he like lemon banana and oranges.`

**Note:** This step can take 2 to 3 minutes on the full dataset.

### Step 8: Tokenization (demonstration)

Tokenization splits a sentence into words and punctuation marks (tokens).

```python
tokens = word_tokenize('I enjoy playing football in the rain.')
# ['I', 'enjoy', 'playing', 'football', 'in', 'the', 'rain', '.']
```

### Step 9: Part-of-speech tagging (demonstration)

POS tagging gives a word class to each token. This helps when one word has different meanings.

```python
nltk.pos_tag(tokens)
# [('I', 'PRP'), ('enjoy', 'VBP'), ('playing', 'VBG'), ('football', 'NN'),
#  ('in', 'IN'), ('the', 'DT'), ('rain', 'NN'), ('.', '.')]
```

### Step 10: Remove stop words

Stop words are common words such as "the", "he" and "have". They add little meaning. Remove them.

**Caution:** Do not remove "not" and its short forms (for example, "don't" and "didn't"). These words change the sentiment.
For example, "not a good movie" is negative. If you remove "not", the text becomes "a good movie". This text is positive.

Make a new stop word list without these words:

```python
stopword_list = nltk.corpus.stopwords.words('english')

updated_stopword_list = []
for word in stopword_list:
    if word == 'not' or word.endswith("n't"):
        pass
    else:
        updated_stopword_list.append(word)
```

Remove the stop words from the text:

```python
def remove_stopwords(text, is_lower_case=False):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [t for t in tokens if t not in updated_stopword_list]
    else:
        filtered_tokens = [t for t in tokens if t.lower() not in updated_stopword_list]
    return ' '.join(filtered_tokens)

df['review'] = df['review'].apply(remove_stopwords)
```

Example: `The movie was not that great` becomes `movie not great`.

## 7. Result

The `review` column now contains clean text. Example:

| Before | After |
|--------|-------|
| `A wonderful little production. <br /><br />The filming technique...` | `wonder littl product film techniqu veri unassu...` |

You can now use this clean text for a model.

## 8. Summary of Techniques

| Technique             | What it does                                   | Example                         |
|-----------------------|------------------------------------------------|---------------------------------|
| Noise removal         | Removes HTML tags and brackets.                | `<p>Hi</p>` → `Hi`              |
| Special characters    | Removes punctuation and symbols.               | `Hi!` → `Hi`                    |
| Lemmatization         | Changes a word to its dictionary root.         | `better` → `good`               |
| Stemming              | Cuts a word to a short root.                   | `likes` → `like`                |
| Tokenization          | Splits text into tokens.                       | `I enjoy it.` → `['I', 'enjoy', 'it', '.']` |
| POS tagging           | Gives a word class to each token.              | `rain` → `NN` (noun)            |
| Stop word removal     | Removes common words with little meaning.      | `The movie was not great` → `movie not great` |
