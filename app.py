# Combined Code1 and Code2 for Enhanced Import Prediction Dashboard

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

st.set_page_config(layout="wide")

# --- Load Data ---
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

# --- Model Training Function ---
def train_rf_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, score

# --- RF Prediction Function ---
def predict_rf(model, tariff_value, years, mean_features, features):
    future_data = pd.DataFrame([{**{"Year": y, "Average Tariff Rate (%)": tariff_value}, **mean_features} for y in years])
    future_data = future_data[features]
    preds = model.predict(future_data)
    return np.array(preds) / 1e9  # Convert to billions

# --- Linear Regression Function ---
def predict_lr(df, country, tariff_rate):
    X = df[["Year", "Average Tariff Rate (%)"]]
    y = df["Import Value (USD)"]
    model = LinearRegression()
    model.fit(X, y)
    last_year = df['Year'].max()
    future_df = pd.DataFrame({
        'Year': [last_year + i for i in range(1, 4)],
        'Average Tariff Rate (%)': [tariff_rate] * 3
    })
    preds = model.predict(future_df)
    return df['Year'], y, future_df['Year'], preds, country

# --- Main App ---
st.title("📈 India's Import Value Prediction Dashboard")

df = load_data()

if df is not None:
    features = ["Year", "Average Tariff Rate (%)", "GDP Growth (%)", "USD/INR (avg)", "Inflation (CPI, %)"]
    target = "Import Value (USD)"

    df_usa = df[df["Country"] == "USA"].copy()
    df_china = df[df["Country"] == "China"].copy()

    if not df_usa.empty and not df_china.empty:
        rf_usa, r2_usa = train_rf_model(df_usa, features, target)
        rf_china, r2_china = train_rf_model(df_china, features, target)

        st.sidebar.markdown(f"**USA Model R² Score:** {r2_usa:.2f}")
        st.sidebar.markdown(f"**China Model R² Score:** {r2_china:.2f}")
        tariff_value = st.sidebar.slider("Tariff Rate (%)", 0.0, 30.0, 15.0, 0.5)

        mean_features = df[features].drop(columns=["Year", "Average Tariff Rate (%)"]).mean().to_dict()
        future_years = [2025, 2026, 2027]

        usa_preds = predict_rf(rf_usa, tariff_value, future_years, mean_features, features)
        china_preds = predict_rf(rf_china, tariff_value, future_years, mean_features, features)

        fig_rf, axs_rf = plt.subplots(1, 2, figsize=(14, 6))
        axs_rf[0].plot(future_years, usa_preds, marker='o', color='blue')
        axs_rf[0].set_title(f"USA Import Forecast (Tariff: {tariff_value}%)")
        axs_rf[0].set_ylabel("Import Value (Billion USD)")
        axs_rf[0].grid(True)
        axs_rf[1].plot(future_years, china_preds, marker='o', color='red')
        axs_rf[1].set_title(f"China Import Forecast (Tariff: {tariff_value}%)")
        axs_rf[1].grid(True)
        plt.tight_layout()
        st.pyplot(fig_rf)

        fig_imp, axs_imp = plt.subplots(1, 2, figsize=(14, 4))
        axs_imp[0].barh(features, rf_usa.feature_importances_, color='blue')
        axs_imp[0].set_title("USA Feature Importance")
        axs_imp[0].invert_yaxis()
        axs_imp[1].barh(features, rf_china.feature_importances_, color='red')
        axs_imp[1].set_title("China Feature Importance")
        axs_imp[1].invert_yaxis()
        st.pyplot(fig_imp)

        # Linear Regression Forecasts
        st.header("🔍 Linear Regression Forecasts")

        usa_years, usa_actual, usa_pred_years, usa_preds_lr, _ = predict_lr(df_usa, 'USA', tariff_value)
        china_years, china_actual, china_pred_years, china_preds_lr, _ = predict_lr(df_china, 'China', tariff_value)

        fig_lr, ax_lr = plt.subplots(1, 2, figsize=(14, 5))
        ax_lr[0].plot(usa_years, usa_actual / 1e9, label='Actual')
        ax_lr[0].plot(usa_pred_years, usa_preds_lr / 1e9, label='Predicted', linestyle='--', marker='o')
        ax_lr[0].set_title("USA LR Forecast")
        ax_lr[0].legend()
        ax_lr[0].grid(True)

        ax_lr[1].plot(china_years, china_actual / 1e9, label='Actual')
        ax_lr[1].plot(china_pred_years, china_preds_lr / 1e9, label='Predicted', linestyle='--', marker='o')
        ax_lr[1].set_title("China LR Forecast")
        ax_lr[1].legend()
        ax_lr[1].grid(True)
        st.pyplot(fig_lr)

    else:
        st.warning("Data for USA or China is missing in the dataset.")
