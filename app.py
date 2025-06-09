try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    raise ImportError("Streamlit is not installed. Please install it using 'pip install streamlit'")

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

# SHAP library
try:
    import shap
except ImportError:
    st.error("SHAP library not installed. Please run 'pip install shap'")

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path="prediction_data.xlsx"):
    if not os.path.exists(file_path):
        st.error(f"The file '{file_path}' was not found. Please upload it to the working directory.")
        return None
    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        df["Year"] = pd.to_datetime(df["Year"], format="%Y").dt.year
        df["Average Tariff Rate (%)"] = df["Average Tariff Rate (%)"].astype(float)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def train_rf_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return model, score, X_train, X_test

def predict_rf(model, base_data, tariff, features, target, future_years):
    mean_features = base_data[features].drop(columns=["Year", "Average Tariff Rate (%)"]).mean()
    data = []
    for year in future_years:
        row = {"Year": year, "Average Tariff Rate (%)": tariff}
        row.update(mean_features)
        data.append(row)
    df_future = pd.DataFrame(data)[features]
    predictions = model.predict(df_future)
    return pd.DataFrame({"Year": future_years, target: predictions / 1e9})

def predict_lr(df, country, tariff):
    X = df[["Year", "Average Tariff Rate (%)"]]
    y = df["Import Value (USD)"]
    model = LinearRegression()
    model.fit(X, y)
    future_years = [df["Year"].max() + i for i in range(1, 4)]
    X_future = pd.DataFrame({"Year": future_years, "Average Tariff Rate (%)": [tariff] * 3})
    predictions = model.predict(X_future)
    return df["Year"], y, X_future["Year"], predictions, country

st.title("India's Import Value Prediction")
df = load_data()

if df is not None:
    features = ["Year", "Average Tariff Rate (%)", "GDP Growth (%)", "USD/INR (avg)", "Inflation (CPI, %)"]
    target = "Import Value (USD)"

    df_usa = df[df["Country"] == "USA"].copy()
    df_china = df[df["Country"] == "China"].copy()

    if not df_usa.empty and not df_china.empty:
        rf_usa, r2_usa, X_train_usa, X_test_usa = train_rf_model(df_usa, features, target)
        rf_china, r2_china, X_train_china, X_test_china = train_rf_model(df_china, features, target)

        st.sidebar.header("Settings")
        st.sidebar.metric("USA RF R² Score", f"{r2_usa:.2f}")
        st.sidebar.metric("China RF R² Score", f"{r2_china:.2f}")

        tariff_rate = st.sidebar.slider("Select Tariff Rate (%)", 0.0, 30.0, 5.0, 0.5)
        future_years = [df["Year"].max() + i for i in range(1, 4)]

        st.header("Random Forest Predictions")
        usa_rf_pred = predict_rf(rf_usa, df_usa, tariff_rate, features, target, future_years)
        china_rf_pred = predict_rf(rf_china, df_china, tariff_rate, features, target, future_years)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("USA RF Prediction")
            fig, ax = plt.subplots()
            ax.plot(usa_rf_pred["Year"], usa_rf_pred[target], marker='o', color='blue')
            for x, y in zip(usa_rf_pred["Year"], usa_rf_pred[target]):
                ax.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            ax.set_title("USA Import (Billion USD)")
            ax.grid(True)
            st.pyplot(fig)
            st.dataframe(usa_rf_pred.rename(columns={target: "Import Value (Billion USD)"}))

        with col2:
            st.subheader("China RF Prediction")
            fig, ax = plt.subplots()
            ax.plot(china_rf_pred["Year"], china_rf_pred[target], marker='o', color='red')
            for x, y in zip(china_rf_pred["Year"], china_rf_pred[target]):
                ax.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            ax.set_title("China Import (Billion USD)")
            ax.grid(True)
            st.pyplot(fig)
            st.dataframe(china_rf_pred.rename(columns={target: "Import Value (Billion USD)"}))

        st.header("Linear Regression Forecasts")
        us_act_years, us_act_vals, us_pred_years, us_preds, _ = predict_lr(df_usa, "USA", tariff_rate)
        china_act_years, china_act_vals, china_pred_years, china_preds, _ = predict_lr(df_china, "China", tariff_rate)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("USA LR Forecast")
            fig, ax = plt.subplots()
            ax.plot(us_act_years, us_act_vals / 1e9, label='Actual')
            ax.plot(us_pred_years, us_preds / 1e9, linestyle='--', marker='o', label='Predicted')
            for x, y in zip(us_pred_years, us_preds / 1e9):
                ax.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            ax.set_title("USA Import (Billion USD)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Year": us_pred_years, "Predicted Import Value (Billion USD)": us_preds / 1e9}))

        with col4:
            st.subheader("China LR Forecast")
            fig, ax = plt.subplots()
            ax.plot(china_act_years, china_act_vals / 1e9, label='Actual')
            ax.plot(china_pred_years, china_preds / 1e9, linestyle='--', marker='o', label='Predicted')
            for x, y in zip(china_pred_years, china_preds / 1e9):
                ax.text(x, y, f"{y:.2f}", ha='center', va='bottom')
            ax.set_title("China Import (Billion USD)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Year": china_pred_years, "Predicted Import Value (Billion USD)": china_preds / 1e9}))

        st.header("Feature Importances")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.barh(features, rf_usa.feature_importances_, color='blue')
        ax1.set_title("USA Feature Importance")
        ax1.invert_yaxis()
        ax2.barh(features, rf_china.feature_importances_, color='red')
        ax2.set_title("China Feature Importance")
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.header("SHAP Summary Analysis")
        if "shap" in globals():
            try:
                explainer_usa = shap.Explainer(rf_usa, X_train_usa)
                shap_values_usa = explainer_usa(X_test_usa)

                explainer_china = shap.Explainer(rf_china, X_train_china)
                shap_values_china = explainer_china(X_test_china)

                st.subheader("USA SHAP Summary")
                fig = plt.figure(figsize=(10, 4))
                shap.plots.beeswarm(shap_values_usa, show=False)
                st.pyplot(fig)

                st.subheader("China SHAP Summary")
                fig = plt.figure(figsize=(10, 4))
                shap.plots.beeswarm(shap_values_china, show=False)
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not compute SHAP values: {e}")
    else:
        st.warning("Data for USA or China not found in the file.")
