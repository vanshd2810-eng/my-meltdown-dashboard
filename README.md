# My Meltdown — kava café-bar idea validation

A Streamlit dashboard that tests a business idea (a kava café-come-bar) with a 30-question survey and four data-mining techniques:

| Technique | Question it answers | Target |
|---|---|---|
| Classification (logistic regression, random forest, gradient boosting) | Will this person buy? | Q30 |
| Clustering (k-means) | Who are the customers: most, mid or least probable? | Q5–Q14, Q17–Q20 |
| Association rule mining (Apriori) | Which of the 10 services do people want together? | Q23 |
| Regression (linear, random forest, gradient boosting, log scale) | How much will a buyer spend per month? | Q25 |

The data is **synthetic**: 3,500 simulated respondents with noise and outliers added on purpose (straight-liners, typos, extreme spenders, impossible values, missing answers, flipped answers). The app generates it on start-up with a fixed seed (42), so every run shows the same dataset; download it from the sidebar. It tests the survey and the models end to end. Replace it with real answers before making a real decision.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit dashboard (9 tabs, incl. a Critical review) |
| `.streamlit/config.toml` | Theme (brand colours) |
| `pipeline.py` | Cleaning plus the four models. `run(raw_df)` returns every result table |
| `simulate.py` | Synthetic data generator. `python simulate.py out.csv 3500 42` or `generate(n, seed)` |
| `survey.py` | The 30 questions and the 10 services |
| `requirements.txt` | Python packages |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use real survey answers

In the app's sidebar, choose **Upload real survey CSV**. The file needs the same column names as the downloadable synthetic dataset. Every chart and number updates.
