import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports (may not be installed)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("📊 PDDS — Personal Digital Data Scientist (Enhanced)")

# Sidebar
st.sidebar.header("Upload / Settings")
uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV / Excel)", type=["csv","xlsx"])
use_sqlite = st.sidebar.checkbox("Use local SQLite storage", value=True)
show_raw = st.sidebar.checkbox("Show raw data preview", value=True)

def to_sqlite(df, table_name="processed_data", db_path="project.db"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

def download_link(obj, filename, text):
    if isinstance(obj, pd.DataFrame):
        csv = obj.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        return href

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.success("File uploaded successfully.")
    if show_raw:
        st.subheader("Raw Data (first 10 rows)")
        st.dataframe(df.head(10))

    # Basic cleaning
    st.subheader("1) Data Cleaning")
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()

    # Fill numeric missing with mean, categorical with mode
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
    for c in numeric_cols:
        if df_clean[c].isnull().sum() > 0:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
    for c in cat_cols:
        if df_clean[c].isnull().sum() > 0:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0])

    st.write("Filled missing numeric values with mean; categorical with mode.")

    if use_sqlite:
        to_sqlite(df_clean)
        st.success("Saved cleaned data to local SQLite (project.db).")

    st.subheader("2) Exploratory Data Analysis (EDA)")
    st.write("Select columns to visualize / analyze:")
    cols = df_clean.columns.tolist()
    x_col = st.selectbox("X-axis / Feature", cols)
    y_col = st.selectbox("Y-axis / Feature (optional)", [None] + cols)

    st.write("Summary statistics:")
    st.dataframe(df_clean.describe().T)

    # Correlation heatmap for numeric cols
    if len(numeric_cols) >= 2:
        st.write("Correlation heatmap (numeric features):")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df_clean[numeric_cols].corr(), annot=True, ax=ax)
        st.pyplot(fig)

    # Plot selected chart
    st.write("Plot:")
    fig2, ax2 = plt.subplots()
    if y_col is None:
        try:
            df_clean[x_col].value_counts().head(20).plot(kind='bar', ax=ax2)
        except Exception:
            st.write("Unable to plot selected column.")
    else:
        if x_col in numeric_cols and y_col in numeric_cols:
            ax2.scatter(df_clean[x_col], df_clean[y_col])
            ax2.set_xlabel(x_col); ax2.set_ylabel(y_col)
        else:
            sns.boxplot(x=df_clean[x_col], y=df_clean[y_col], ax=ax2)
    st.pyplot(fig2)

    st.subheader("3) Automatic Task Detection & ML (with optional Hyperparameter Tuning)")
    st.write("Provide target column (if you want prediction). Leave empty to run clustering.")
    target = st.selectbox("Target column", [None] + cols)

    if target:
        task_type = "regression" if target in numeric_cols else "classification"
        st.write(f"Detected task: **{task_type}**")

        # Prepare data
        X = df_clean.drop(columns=[target])
        y = df_clean[target].copy()

        # Encode categorical features
        X_enc = X.copy()
        le_dict = {}
        for c in X_enc.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X_enc[c] = le.fit_transform(X_enc[c].astype(str))
            le_dict[c] = le

        # Encode target if classification and object
        if task_type == "classification" and y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
        else:
            le_y = None

        # Train/test split
        test_size = st.slider("Test set size (percent)", 10, 40, 20)
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size/100, random_state=42)

        tune = st.checkbox("Enable hyperparameter tuning (GridSearchCV) - may take longer", value=False)

        if task_type == "regression":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(random_state=42)
            }
            if tune:
                st.write("Running GridSearchCV for RandomForest regressor...")
                param_grid = {"n_estimators":[50,100], "max_depth":[None,5,10]}
                grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_rf = grid.best_estimator_
                st.write("Best RF params:", grid.best_params_)
                preds = best_rf.predict(X_test)
                score = r2_score(y_test, preds)
                st.write("Best RandomForest R2:", score)
                chosen_model = best_rf
            else:
                lr = LinearRegression().fit(X_train_scaled, y_train)
                rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
                preds_lr = lr.predict(X_test_scaled)
                preds_rf = rf.predict(X_test)
                st.write("Linear Regression R2:", r2_score(y_test, preds_lr))
                st.write("Random Forest R2:", r2_score(y_test, preds_rf))
                chosen_model = rf if r2_score(y_test, preds_rf) >= r2_score(y_test, preds_lr) else lr

            # Show predictions
            results = X_test.copy()
            results['actual'] = y_test
            try:
                preds_final = chosen_model.predict(X_test_scaled if isinstance(chosen_model, LinearRegression) else X_test)
            except Exception:
                preds_final = chosen_model.predict(X_test)
            results['predicted'] = preds_final
            st.write("Sample predictions:")
            st.dataframe(results.head())
            st.markdown(download_link(results.reset_index(drop=True), "predictions.csv", "Download predictions as CSV"), unsafe_allow_html=True)

        else:
            # classification
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            lr = LogisticRegression(max_iter=200)

            if tune:
                st.write("Running GridSearchCV for RandomForest classifier...")
                param_grid = {"n_estimators":[50,100], "max_depth":[None,5,10]}
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_rf = grid.best_estimator_
                st.write("Best RF params:", grid.best_params_)
                preds = best_rf.predict(X_test)
                st.write("RandomForest Accuracy (tuned):", accuracy_score(y_test, preds))
                chosen_model = best_rf
            else:
                rf.fit(X_train, y_train)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                lr.fit(X_train_scaled, y_train)
                preds_rf = rf.predict(X_test)
                preds_lr = lr.predict(X_test_scaled)
                st.write("Random Forest Accuracy:", accuracy_score(y_test, preds_rf))
                st.write("Logistic Regression Accuracy:", accuracy_score(y_test, preds_lr))
                chosen_model = rf if accuracy_score(y_test, preds_rf) >= accuracy_score(y_test, preds_lr) else lr

            results = X_test.copy()
            results['actual'] = y_test
            try:
                results['predicted'] = chosen_model.predict(X_test if isinstance(chosen_model, RandomForestClassifier) else X_test_scaled)
            except Exception:
                results['predicted'] = chosen_model.predict(X_test)
            st.write("Sample predictions:")
            st.dataframe(results.head())
            st.markdown(download_link(results.reset_index(drop=True), "predictions.csv", "Download predictions as CSV"), unsafe_allow_html=True)

            # SHAP explainability for tree-based models
            if SHAP_AVAILABLE and isinstance(chosen_model, (RandomForestClassifier, RandomForestRegressor)):
                st.subheader("Model Explainability (SHAP)")
                try:
                    explainer = shap.TreeExplainer(chosen_model)
                    # use a small sample for speed
                    sample_X = X_test.sample(n=min(100, X_test.shape[0]), random_state=42)
                    shap_values = explainer.shap_values(sample_X)
                    st.write("Generating SHAP summary plot (may take a moment)...")
                    fig_shap = shap.summary_plot(shap_values, sample_X, show=False)
                    st.pyplot(bbox_inches='tight')
                except Exception as e:
                    st.write("SHAP failed:", e)
            elif not SHAP_AVAILABLE:
                st.write("SHAP library not installed. To enable SHAP add 'shap' to requirements.")

    else:
        st.write("No target provided → Running KMeans clustering")
        # Use numeric columns for clustering
        if len(numeric_cols) == 0:
            st.write("No numeric columns available for clustering.")
        else:
            k = st.slider("Number of clusters (k)", 2, 10, 3)
            km = KMeans(n_clusters=k, random_state=42)
            df_num = df_clean[numeric_cols].fillna(0)
            km.fit(df_num)
            df_clean['cluster'] = km.labels_
            st.write("Cluster counts:")
            st.dataframe(df_clean['cluster'].value_counts().rename_axis('cluster').reset_index(name='counts'))
            st.markdown(download_link(df_clean, "clustered_data.csv", "Download clustered dataset as CSV"), unsafe_allow_html=True)

    st.subheader("4) Time Series Forecasting (auto selection)")
    st.write("Pick a date/time column and a numeric value column to forecast.")
    date_cols = [c for c in cols if 'date' in c.lower() or 'time' in c.lower()]
    st.write("Detected possible date/time columns:", date_cols)
    ts_date = st.selectbox("Date column (or select None)", [None] + cols)
    ts_value = st.selectbox("Value column (numeric)", [None] + numeric_cols)

    if ts_date and ts_value:
        try:
            ts = df_clean[[ts_date, ts_value]].copy()
            ts[ts_date] = pd.to_datetime(ts[ts_date])
            ts = ts.sort_values(by=ts_date).set_index(ts_date)
            # infer freq; if not, set to daily
            freq = pd.infer_freq(ts.index)
            if not freq:
                freq = 'D'
            ts = ts.asfreq(freq)
            ts[ts_value] = ts[ts_value].interpolate()

            st.write("Time series head:")
            st.dataframe(ts.head())

            forecast_steps = st.number_input("Forecast steps (periods)", min_value=1, max_value=365, value=30)

            # Try pmdarima auto_arima first
            if PMDARIMA_AVAILABLE:
                st.write("Running pmdarima.auto_arima...")
                model = auto_arima(ts[ts_value], seasonal=False, error_action='ignore', suppress_warnings=True)
                fc = model.predict(n_periods=forecast_steps)
                last = ts.index[-1]
                fc_index = pd.date_range(start=last, periods=forecast_steps+1, closed='right', freq=freq)
                fc_df = pd.DataFrame({'date': fc_index, 'forecast': fc})
                st.dataframe(fc_df.head())
                fig3, ax3 = plt.subplots()
                ts[ts_value].plot(ax=ax3, label='observed')
                ax3.plot(fc_index, fc, label='forecast')
                ax3.legend()
                st.pyplot(fig3)
                st.markdown(download_link(fc_df, "forecast_pmdarima.csv", "Download forecast as CSV"), unsafe_allow_html=True)
            elif PROPHET_AVAILABLE:
                st.write("pmdarima not available; using Prophet")
                df_prophet = ts.reset_index().rename(columns={ts_date:'ds', ts_value:'y'}) if ts_date else ts.reset_index().rename(columns={ts.index.name:'ds', ts_value:'y'})
                m = Prophet()
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=forecast_steps, freq=freq)
                forecast = m.predict(future)
                fc = forecast[['ds','yhat']].tail(forecast_steps)
                st.dataframe(fc.head())
                fig3, ax3 = plt.subplots()
                ax3.plot(ts.index, ts[ts_value], label='observed')
                ax3.plot(fc['ds'], fc['yhat'], label='forecast')
                ax3.legend()
                st.pyplot(fig3)
                st.markdown(download_link(fc.rename(columns={'ds':'date','yhat':'forecast'}), "forecast_prophet.csv", "Download forecast as CSV"), unsafe_allow_html=True)
            else:
                st.write("Auto ARIMA (pmdarima) and Prophet are not installed. Install 'pmdarima' or 'prophet' for better time-series support. Falling back to ARIMA(1,1,1).")
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(ts[ts_value], order=(1,1,1))
                model_fit = model.fit()
                fc = model_fit.forecast(steps=forecast_steps)
                last = ts.index[-1]
                fc_index = pd.date_range(start=last, periods=forecast_steps+1, closed='right', freq=freq)
                fc_df = pd.DataFrame({'date': fc_index, 'forecast': fc})
                st.dataframe(fc_df.head())
                fig3, ax3 = plt.subplots()
                ts[ts_value].plot(ax=ax3, label='observed')
                ax3.plot(fc_index, fc, label='forecast')
                ax3.legend()
                st.pyplot(fig3)
                st.markdown(download_link(fc_df, "forecast_arima.csv", "Download forecast as CSV"), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Time series forecasting failed: {e}")

    st.subheader("5) Save project snapshot")
    proj_name = st.text_input("Project name for this snapshot", value="pdds_snapshot")
    if st.button("Save snapshot to SQLite"):
        snapshot = df_clean.copy()
        snapshot['snapshot_name'] = proj_name
        to_sqlite(snapshot, table_name="snapshots")
        st.success("Snapshot saved.")

else:
    st.info("Upload a dataset to start. You can use the sample dataset provided in the repository.")
