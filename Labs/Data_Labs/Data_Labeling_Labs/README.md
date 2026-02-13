# Data Labeling Lab - Personal Submission

## 1. Assignment Goal

In this lab, I practiced data labeling with Snorkel on YouTube spam comments.
My goal was to keep the original lab structure, make small custom changes, and show that I understand the workflow.

## 2. What I Changed (My Version)

I kept the project minimal and only edited existing notebooks.

### Notebook 1: `01_spam_tutorial.ipynb`
I added one new labeling function:
- `promo_subscribe_combo`
- It marks a comment as spam when the text includes `subscribe` and also `channel` or `now`.

I added this function to the LF lists used in the notebook.

### Notebook 2: `02_spam_data_augmentation_tutorial.ipynb`
I added one new transformation function:
- `normalize_spammy_caps`
- It converts very loud all-caps words (length 4 or more) to lowercase.

I added this TF to the `tfs` list and updated the `MeanFieldPolicy` probabilities.

### Notebook 3: `03_spam_data_slicing_tutorial.ipynb`
I did not change this notebook.

## 3. Environment Setup (with `uv`)

### Step 1: Create virtual environment
```bash
uv venv
```

### Step 2: Activate environment
```bash
source .venv/bin/activate
```

### Step 3: Install requirements
```bash
uv pip install -r requirements.txt
```

## 3. How I Re-run the Lab

I run the notebooks in this order:

1. `01_spam_tutorial.ipynb`
2. `02_spam_data_augmentation_tutorial.ipynb`
3. `03_spam_data_slicing_tutorial.ipynb`

To start Jupyter:
```bash
uv run jupyter notebook
```

## 5. Data Used

The input CSV files are already in `data/`:
- `Youtube01-Psy.csv`
- `Youtube02-KatyPerry.csv`
- `Youtube03-LMFAO.csv`
- `Youtube04-Eminem.csv`
- `Youtube05-Shakira.csv`

## 6. Quick Troubleshooting

- If `wordnet` is missing in notebook 2, run the cell with `nltk.download("wordnet")` again.
- If imports fail, confirm `.venv` is active and run:
  ```bash
  uv pip install -r requirements.txt
  ```
