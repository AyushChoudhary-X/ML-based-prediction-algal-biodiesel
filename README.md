# Algal Biodiesel ML

Streamlit app + training pipeline for predicting biodiesel yield from algal-oil transesterification conditions.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (9 pages) |
| `core.py` | Data prep, models, Shapley-style attribution, optimiser, fuel standards |
| `train.py` | Re-creates everything in `artifacts/` (metrics, figures, SHAP values) |
| `data/algal_biodiesel_cleaned.xlsx` | Cleaned dataset (sheet `Cleaned_Data` is used) |
| `artifacts/` | Pre-computed metrics and figures shown by the app |

## Run locally
```
pip install -r requirements.txt
python train.py          # optional: regenerates artifacts/ (about a minute)
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (include `artifacts/` and `data/`).
2. Go to share.streamlit.io, click **Create app**, select the repo, branch, and set main file to `app.py`.
3. Deploy. The app fits the final models at start-up (about 1 s), so no model pickles are needed.

## Validation note
Scores are from random 5-fold cross-validation repeated 3 times. Species, catalyst and lipid content are constant within each study,
so predictions are meant for conditions inside the systems in the dataset.
