import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Bitcoin Price Forecasting", layout="wide")

menu = st.sidebar.radio("Navigation", ["Home", "Sequence Viewer", "Prediction Dashboard"])

# -------------------------------
# Helper: Get latest 60 days from 2024
# -------------------------------
def get_latest_60(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Date','Price']).sort_values(by='Date', ascending=False)
    df_2024 = df[df['Date'].dt.year == 2024]
    latest_df = df_2024.head(60)
    return latest_df, 2024

# -------------------------------
# Home Page
# -------------------------------
if menu == "Home":
    st.markdown("<h1 style='text-align: center; color: purple;'>📈 Bitcoin Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: purple;'>Developed by <b>Kurmapu Lakshmi Tejasree</b></h3>", unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/resize:fit:1200/1*KthiCwwcT3TUmc5HA2sfQA.png",
             caption="Forecasting powered by CNN, LSTM, and RNN", use_column_width=True)
    st.markdown("""
    <div style='font-size:18px; line-height:1.6; text-align: justify;'>
    This application forecasts <b>Bitcoin Price</b> using the latest 60 days of data from 2024.  
    It predicts the next <b>1, 3, and 7 days</b using different deep learning models:  
    - 1-Day → CNN  
    - 3-Day → LSTM  
    - 7-Day → RNN  

    Navigate to the <b>Sequence Viewer</b> to see the latest 60 days, or to the <b>Prediction Dashboard</b> to run forecasts.  
    <br><br>
    <b>Note:</b> Predictions are illustrative and based on trained models. They should not be considered financial advice.  
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Sequence Viewer (NO LABELS)
# -------------------------------
elif menu == "Sequence Viewer":
    uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Price'", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        latest_df, latest_year = get_latest_60(df)
        if latest_df.empty:
            st.error("No valid 2024 rows found. Please check your CSV format.")
        else:
            st.subheader(f"Latest 60 Days Price Sequence ({latest_year})")
            fig, ax = plt.subplots(figsize=(12,5))
            trend_color = "green" if latest_df['Price'].iloc[-1] > latest_df['Price'].iloc[0] else "red"
            ax.plot(latest_df['Date'], latest_df['Price'], marker='o', color=trend_color, label="Past 60 Days")
            ax.set_title("Latest 60 Days Price Sequence", fontsize=14, color="purple")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            st.dataframe(latest_df[['Date','Price']])

# -------------------------------
# Prediction Dashboard
# -------------------------------
elif menu == "Prediction Dashboard":
    uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Price'", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        latest_df, latest_year = get_latest_60(df)
        if latest_df.empty:
            st.error("No valid 2024 rows found. Please check your CSV format.")
        else:
            latest_prices = latest_df['Price'].to_numpy().reshape(-1,1)
            scaler = MinMaxScaler()
            prices_scaled = scaler.fit_transform(latest_prices)
            seq_length = len(latest_prices)
            X_input = prices_scaled.reshape(1, seq_length, 1)

            if st.button("Predict"):
                cnn_model = load_model("E:/vscode/CryptoCast_bitcoin_project/cnn_model.h5", compile=False)
                lstm_model = load_model("E:/vscode/CryptoCast_bitcoin_project/lstm_model.h5", compile=False)
                rnn_model = load_model("E:/vscode/CryptoCast_bitcoin_project/rnn_model.h5", compile=False)

                y_pred_cnn = cnn_model.predict(X_input)
                y_pred_lstm = lstm_model.predict(X_input)
                y_pred_rnn = rnn_model.predict(X_input)

                predicted_prices = {
                    "1-Day (CNN)": scaler.inverse_transform(y_pred_cnn.reshape(-1,1))[0][0],
                    "3-Day (LSTM)": scaler.inverse_transform(y_pred_lstm.reshape(-1,1))[0][0],
                    "7-Day (RNN)": scaler.inverse_transform(y_pred_rnn.reshape(-1,1))[0][0],
                }

                st.subheader(f"Predicted Next Prices ({latest_year})")
                st.write(pd.DataFrame(predicted_prices.items(), columns=["Horizon","Predicted Price"]))

                # Chart 1: Past 60 Days + Predicted (no labels here)
                st.subheader("Past 60 Days + Predicted")
                x_actual = list(range(1, len(latest_df)+1))
                y_actual = latest_df['Price'].values
                x_pred = [len(latest_df)+1, len(latest_df)+3, len(latest_df)+7]
                y_pred = [predicted_prices["1-Day (CNN)"], predicted_prices["3-Day (LSTM)"], predicted_prices["7-Day (RNN)"]]

                fig, ax = plt.subplots(figsize=(12,5))
                trend_color = "green" if y_actual[-1] > y_actual[0] else "red"
                ax.plot(x_actual, y_actual, marker='o', color=trend_color, label="Past 60 Days")
                ax.plot([x_actual[-1], x_pred[0]], [y_actual[-1], y_pred[0]], color="blue", linestyle="--", marker='o', label="Predicted 1-Day")
                ax.plot([x_actual[-1], x_pred[1]], [y_actual[-1], y_pred[1]], color="orange", linestyle="--", marker='o', label="Predicted 3-Day")
                ax.plot([x_actual[-1], x_pred[2]], [y_actual[-1], y_pred[2]], color="purple", linestyle="--", marker='o', label="Predicted 7-Day")
                ax.set_title("Past 60 Days vs Predicted", fontsize=14, color="purple")
                ax.set_xlabel("Sequence Day")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)

                # Chart 2: Bar chart comparison (labels above bars)
                st.subheader("Predicted vs Last Actual Price")
                fig2, ax2 = plt.subplots(figsize=(8,5))
                labels = ["Last Actual", "Predicted 1-Day", "Predicted 3-Day", "Predicted 7-Day"]
                values = [y_actual[-1], y_pred[0], y_pred[1], y_pred[2]]
                colors = ["red", "blue", "orange", "purple"]
                ax2.bar(labels, values, color=colors)
                for i, v in enumerate(values):
                    ax2.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
                ax2.set_ylabel("Price")
                ax2.set_title("Comparison of Actual vs Predicted", fontsize=14, color="purple")
                st.pyplot(fig2)

                                # Chart 3: Continuous forecast extension (labels only on forecast path, path color green)
                st.subheader("Continuous Forecast Extension")
                fig3, ax3 = plt.subplots(figsize=(12,5))
                ax3.plot(x_actual, y_actual, marker='o', color=trend_color, label="Past 60 Days")
                # Forecast path in green
                ax3.plot([x_actual[-1]] + x_pred, [y_actual[-1]] + y_pred,
                         marker='o', linestyle="--", color="green", label="Forecast Path")
                # Labels only for forecast path points
                for i, val in enumerate(y_pred):
                    ax3.text(x_pred[i], val, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
                ax3.set_title("Forecast Extension Beyond 60 Days", fontsize=14, color="purple")
                ax3.set_xlabel("Sequence Day")
                ax3.set_ylabel("Price")
                ax3.legend()
                ax3.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig3)

                # Chart 4: Moving Average Overlay
                st.subheader("7-Day Moving Average Trend")
                fig4, ax4 = plt.subplots(figsize=(12,5))
                ax4.plot(x_actual, y_actual, marker='o', color=trend_color, label="Past 60 Days")
                latest_df['MA7'] = latest_df['Price'].rolling(window=7).mean()
                ax4.plot(x_actual, latest_df['MA7'], color="black", linestyle="-", linewidth=2, label="7-Day Moving Avg")
                ax4.scatter(x_pred, y_pred, color=["blue","orange","purple"], marker='o', label="Predicted Points")
                # Labels only for predicted points
                for i, val in enumerate(y_pred):
                    ax4.text(x_pred[i], val, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
                ax4.set_title("Past 60 Days with 7-Day Moving Average", fontsize=14, color="purple")
                ax4.set_xlabel("Sequence Day")
                ax4.set_ylabel("Price")
                ax4.legend()
                ax4.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig4)
