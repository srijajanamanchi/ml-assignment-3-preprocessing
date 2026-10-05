# ML Assignment 3 — Data preprocessing, in the right order

**Dataset:** UCI Adult (census income), classification of income `>50K` vs `<=50K`
- Source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
- Original: https://archive.ics.uci.edu/dataset/2/adult
- 48,842 rows in the file; the notebook uses a random 10,000-row sample (`random_state=42`) so KNNImputer runs quickly in Colab. Missing values are `?` in `workclass`, `occupation` and `native_country`. Target is ~24% positive.

## Files
- `assignment3_preprocessing.ipynb` — full notebook, every output visible (Q1–Q12)
- `pipeline.joblib` — fitted `Pipeline(ColumnTransformer)`, fitted on the training set only
- `freq_encoder.py` — small custom `FrequencyEncoder` used inside the pipeline. **Keep it in the same folder as `pipeline.joblib`**, otherwise `joblib.load` cannot import the class.

## Key results
- Train/test: 8,000 / 2,000 rows, stratified, `random_state=42`
- Pipeline output: 49 columns for both train and test
- Class ratio 3.15 : 1; SMOTE on train only: 6,070 / 1,930 -> 6,070 / 6,070 (test untouched)
