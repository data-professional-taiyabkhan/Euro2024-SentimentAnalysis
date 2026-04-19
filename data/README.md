# Data

This folder contains the domain dataset used in the paper. The training dataset (TweetEval) is **not** included due to size and licensing — see download instructions below.

---

## Files included

### `euro2024_domain.csv`
871 Euro 2024 tweets collected July 7–21, 2024. Original collection before model scoring.

### `euro_all_models_final.csv`
Same 871 tweets with sentiment scores from all four models appended as columns. This is the file used for all analysis in the paper.

---

## Training data — download required

The models were trained on **TweetEval** (Barbieri et al., 2020):

**Option A — HuggingFace (easiest):**
```python
from datasets import load_dataset
ds = load_dataset('tweet_eval', 'sentiment')
```

**Option B — GitHub:**
```bash
git clone https://github.com/cardiffnlp/tweeteval.git
# Sentiment data is in: tweeteval/datasets/sentiment/
```

The sentiment split contains:
- `train_text.txt` + `train_labels.txt` — 45,615 tweets
- `val_text.txt` + `val_labels.txt` — 2,000 tweets  
- `test_text.txt` + `test_labels.txt` — 12,284 tweets
- Labels: 0 = negative, 1 = neutral, 2 = positive

Place the downloaded files in `data/tweeteval/` and update the path in `notebooks/02_baseline_models.ipynb`.

---

## Data collection

Euro 2024 tweets were collected using [APIFY Twitter Search Scraper](https://apify.com/apidojo/tweet-scraper). Keywords used:

```
#ThreeLions, #England, #ENG, #EURO2024, England football,
Bellingham, Saka, Kane, Southgate
```

Filters applied during preprocessing:
- English language only
- Minimum 4 tokens after cleaning
- Removed: duplicate texts, crypto/giveaway spam, non-ASCII dominant content

---

## Licence note

Euro 2024 tweet text is subject to X (Twitter) Terms of Service. This dataset is shared for non-commercial academic research purposes only. If you use this data, you accept responsibility for complying with the relevant platform terms.