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

# Load the data
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

# Random Forest model training function
def train_rf_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return model, score

# Random Forest Prediction function
def predict_rf(model, data, tariff_value, features, target, years_to_predict=3):
    last_year = data['Year'].max()
    future_years = [last_year + i for i in range(1, years_to_predict + 1)]
    mean_features = data[features].mean().to_dict()
    mean_features["Average Tariff Rate (%)"] = tariff_value

    future_df = pd.DataFrame()
    future_df['Year'] = future_years

    repeated_df = pd.DataFrame([mean_features] * years_to_predict, columns=features)
    preds = model.predict(repeated_df)
    future_df[target] = preds / 1e9
    return future_df

# Linear Regression Prediction function
def predict_lr(df, country, tariff_rate):
    X = df[['Year', 'Average Tariff Rate (%)']]
    y = df['Import Value (USD)']
    model = LinearRegression()
    model.fit(X, y)
    last_year = df['Year'].max()
    future = pd.DataFrame({
        'Year': [last_year + i for i in range(1, 4)],
        'Average Tariff Rate (%)': [tariff_rate] * 3
    })
    pred = model.predict(future)
    return df['Year'], y, future['Year'], pred, country

# Main Streamlit App
st.title("India's Import Value Prediction")
df = load_data()

if df is not None:
    features = ["Year", "Average Tariff Rate (%)", "GDP Growth (%)", "USD/INR (avg)", "Inflation (CPI, %)"]
    target = "Import Value (USD)"

    df_usa = df[df["Country"] == "USA"].copy()
    df_china = df[df["Country"] == "China"].copy()

    if not df_usa.empty and not df_china.empty:
        rf_usa, r2_usa = train_rf_model(df_usa, features, target)
        rf_china, r2_china = train_rf_model(df_china, features, target)

        st.sidebar.header("Settings")
        st.sidebar.markdown(f"**USA Model R² Score:** {r2_usa:.2f}")
        st.sidebar.markdown(f"**China Model R² Score:** {r2_china:.2f}")
        tariff_val = st.sidebar.slider("Tariff Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        pred_years = st.sidebar.slider("Years to Predict (RF)", min_value=1, max_value=5, step=1, value=3)

        st.header("Random Forest Predictions")
        usa_rf_pred = predict_rf(rf_usa, df_usa, tariff_val, features, target, pred_years)
        china_rf_pred = predict_rf(rf_china, df_china, tariff_val, features, target, pred_years)

        fig_rf, ax_rf = plt.subplots(1, 2, figsize=(14, 5))

        ax_rf[0].plot(usa_rf_pred['Year'], usa_rf_pred[target], marker='o', color='blue')
        ax_rf[0].set_title(f"USA RF Prediction (Tariff: {tariff_val}%)")
        ax_rf[0].set_xlabel("Year")
        ax_rf[0].set_ylabel("Import Value (Billion USD)")
        ax_rf[0].grid(True)

        ax_rf[1].plot(china_rf_pred['Year'], china_rf_pred[target], marker='o', color='red')
        ax_rf[1].set_title(f"China RF Prediction (Tariff: {tariff_val}%)")
        ax_rf[1].set_xlabel("Year")
        ax_rf[1].set_ylabel("Import Value (Billion USD)")
        ax_rf[1].grid(True)

        st.pyplot(fig_rf)

        st.dataframe(usa_rf_pred.rename(columns={target: 'USA Import (Billion USD)'}))
        st.dataframe(china_rf_pred.rename(columns={target: 'China Import (Billion USD)'}))

        st.header("Linear Regression Forecast")

        usa_lr = df_usa[['Year', 'Average Tariff Rate (%)', 'Import Value (USD)']]
        china_lr = df_china[['Year', 'Average Tariff Rate (%)', 'Import Value (USD)']]

        y_act_usa, y_usa, y_pred_usa, pred_usa, country_usa = predict_lr(usa_lr, 'USA', tariff_val)
        fig_lr_usa, ax_usa = plt.subplots()
        ax_usa.plot(y_act_usa, y_usa/1e9, label='Actual')
        ax_usa.plot(y_pred_usa, pred_usa/1e9, label='Predicted', linestyle='--', marker='o')
        ax_usa.set_title(f'{country_usa} - LR Forecast')
        ax_usa.set_xlabel('Year')
        ax_usa.set_ylabel('Import Value (Billion USD)')
        ax_usa.legend()
        ax_usa.grid(True)
        st.pyplot(fig_lr_usa)

        y_act_china, y_china, y_pred_china, pred_china, country_china = predict_lr(china_lr, 'China', tariff_val)
        fig_lr_china, ax_china = plt.subplots()
        ax_china.plot(y_act_china, y_china/1e9, label='Actual')
        ax_china.plot(y_pred_china, pred_china/1e9, label='Predicted', linestyle='--', marker='o')
        ax_china.set_title(f'{country_china} - LR Forecast')
        ax_china.set_xlabel('Year')
        ax_china.set_ylabel('Import Value (Billion USD)')
        ax_china.legend()
        ax_china.grid(True)
        st.pyplot(fig_lr_china)

        # Feature Importance
        st.subheader("Feature Importance")
        fig_imp, axs_imp = plt.subplots(1, 2, figsize=(14, 4))

        axs_imp[0].barh(features, rf_usa.feature_importances_, color='blue')
        axs_imp[0].set_title("USA Feature Importance")
        axs_imp[0].invert_yaxis()

        axs_imp[1].barh(features, rf_china.feature_importances_, color='red')
        axs_imp[1].set_title("China Feature Importance")
        axs_imp[1].invert_yaxis()

        st.pyplot(fig_imp)

    else:
        st.warning("Data for USA or China not found.")
