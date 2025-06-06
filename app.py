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

# Train RF Model
def train_rf_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return model, score

# Predict using RF

def predict_rf(model, data, features, tariff_value, years=3):
    last_year = data['Year'].max()
    future_years = [last_year + i for i in range(1, years + 1)]
    mean_vals = data[features].drop(columns=["Average Tariff Rate (%)"]).mean()
    predictions = []
    for year in future_years:
        row = {"Year": year, "Average Tariff Rate (%)": tariff_value}
        for col in mean_vals.index:
            row[col] = mean_vals[col]
        predictions.append(row)
    future_df = pd.DataFrame(predictions)[features]
    predicted_values = model.predict(future_df)
    return pd.DataFrame({"Year": future_years, "Predicted Import Value (Billion USD)": predicted_values / 1e9})

# Predict using Linear Regression

def predict_lr(df, tariff_value, country):
    X = df[["Year", "Average Tariff Rate (%)"]]
    y = df["Import Value (USD)"]
    model = LinearRegression()
    model.fit(X, y)
    last_year = df['Year'].max()
    future_years = [last_year + i for i in range(1, 4)]
    future_df = pd.DataFrame({"Year": future_years, "Average Tariff Rate (%)": [tariff_value]*3})
    predictions = model.predict(future_df)
    return df["Year"], y, future_years, predictions, country

# App Layout
st.title("India's Import Value Forecast Dashboard")
df = load_data()

if df is not None:
    features = ["Year", "Average Tariff Rate (%)", "GDP Growth (%)", "USD/INR (avg)", "Inflation (CPI, %)"]
    target = "Import Value (USD)"

    df_usa = df[df["Country"] == "USA"].copy()
    df_china = df[df["Country"] == "China"].copy()

    if not df_usa.empty and not df_china.empty:
        rf_usa, r2_usa = train_rf_model(df_usa, features, target)
        rf_china, r2_china = train_rf_model(df_china, features, target)

        st.sidebar.markdown(f"**USA RF R² Score:** {r2_usa:.2f}")
        st.sidebar.markdown(f"**China RF R² Score:** {r2_china:.2f}")
        tariff_value = st.sidebar.slider("Select Tariff Rate (%)", 0.0, 30.0, 5.0, 0.5)
        years_to_predict = st.sidebar.slider("Years to Predict (RF)", 1, 5, 3, 1)

        st.header("Random Forest Predictions")

        usa_rf = predict_rf(rf_usa, df_usa, features, tariff_value, years_to_predict)
        china_rf = predict_rf(rf_china, df_china, features, tariff_value, years_to_predict)

        fig_rf, ax_rf = plt.subplots(1, 2, figsize=(14, 5))

        ax_rf[0].plot(usa_rf['Year'], usa_rf['Predicted Import Value (Billion USD)'], marker='o', label='USA', color='blue')
        for x, y in zip(usa_rf['Year'], usa_rf['Predicted Import Value (Billion USD)']):
            ax_rf[0].text(x, y, f"{y:.2f}", ha='center', va='bottom')
        ax_rf[0].set_title("USA Forecast (RF)")
        ax_rf[0].set_xlabel("Year")
        ax_rf[0].set_ylabel("Import Value (Billion USD)")
        ax_rf[0].grid(True)

        ax_rf[1].plot(china_rf['Year'], china_rf['Predicted Import Value (Billion USD)'], marker='o', label='China', color='red')
        for x, y in zip(china_rf['Year'], china_rf['Predicted Import Value (Billion USD)']):
            ax_rf[1].text(x, y, f"{y:.2f}", ha='center', va='bottom')
        ax_rf[1].set_title("China Forecast (RF)")
        ax_rf[1].set_xlabel("Year")
        ax_rf[1].set_ylabel("Import Value (Billion USD)")
        ax_rf[1].grid(True)

        st.pyplot(fig_rf)

        st.header("Linear Regression Predictions")

        usa_actual_years, usa_actual, usa_future_years, usa_pred_lr, _ = predict_lr(df_usa, tariff_value, "USA")
        china_actual_years, china_actual, china_future_years, china_pred_lr, _ = predict_lr(df_china, tariff_value, "China")

        fig_lr, ax_lr = plt.subplots(1, 2, figsize=(14, 5))

        ax_lr[0].plot(usa_actual_years, usa_actual / 1e9, label="Actual", color='skyblue')
        ax_lr[0].plot(usa_future_years, usa_pred_lr / 1e9, linestyle='--', marker='o', color='blue', label="Predicted")
        for x, y in zip(usa_future_years, usa_pred_lr / 1e9):
            ax_lr[0].text(x, y, f"{y:.2f}", ha='center', va='bottom')
        ax_lr[0].set_title("USA Forecast (LR)")
        ax_lr[0].set_xlabel("Year")
        ax_lr[0].set_ylabel("Import Value (Billion USD)")
        ax_lr[0].legend()
        ax_lr[0].grid(True)

        ax_lr[1].plot(china_actual_years, china_actual / 1e9, label="Actual", color='salmon')
        ax_lr[1].plot(china_future_years, china_pred_lr / 1e9, linestyle='--', marker='o', color='red', label="Predicted")
        for x, y in zip(china_future_years, china_pred_lr / 1e9):
            ax_lr[1].text(x, y, f"{y:.2f}", ha='center', va='bottom')
        ax_lr[1].set_title("China Forecast (LR)")
        ax_lr[1].set_xlabel("Year")
        ax_lr[1].set_ylabel("Import Value (Billion USD)")
        ax_lr[1].legend()
        ax_lr[1].grid(True)

        st.pyplot(fig_lr)
    else:
        st.warning("Data for USA or China not found in the file.")
