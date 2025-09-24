# PDDS — Personal Digital Data Scientist (Enhanced Prototype)

This is a deployable Streamlit prototype that demonstrates an end-to-end lightweight AutoML-style assistant for arbitrary datasets.

## New Features in Enhanced Version
- Hyperparameter tuning using GridSearchCV for RandomForest models
- SHAP explainability support for tree-based models (if `shap` installed)
- Auto time-series selection using `pmdarima.auto_arima` or `prophet` (fallback to ARIMA)
- Better instructions for deployment to Streamlit Cloud and Heroku

## Quick Run (local)
1. Create virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install requirements: `pip install -r requirements.txt`
   - Note: `prophet` and `shap` may require additional system packages; if installation fails, remove them from `requirements.txt` and install later.
4. Run the app: `streamlit run app.py`

## Deploy to Streamlit Cloud (recommended)
1. Create a GitHub repository and push this project.
2. Login to Streamlit Cloud and create a new app, connect it to the GitHub repo & branch.
3. Streamlit Cloud will install dependencies and run `streamlit run app.py` automatically.

## Deploy to Heroku (alternative)
1. Create a new Heroku app.
2. Add `Procfile` (already included) and push the repo to Heroku Git remote.
3. Set buildpacks & environment as required. For production, consider using PostgreSQL instead of SQLite.

