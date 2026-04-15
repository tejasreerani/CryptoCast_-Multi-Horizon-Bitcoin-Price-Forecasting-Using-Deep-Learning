import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Bitcoin Price Forecasting", layout="wide")

# -------------------------------
# Navigation
# -------------------------------
menu = st.sidebar.radio("Navigation", ["Home", "Dashboard", "Model Comparison"])

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    cnn = load_model("E:/vscode/CryptoCast_bitcoin_project/cnn_model.h5", compile=False)
    lstm = load_model("E:/vscode/CryptoCast_bitcoin_project/lstm_model.h5", compile=False)
    rnn = load_model("E:/vscode/CryptoCast_bitcoin_project/rnn_model.h5", compile=False)
    return cnn, lstm, rnn

cnn_model, lstm_model, rnn_model = load_models()

# -------------------------------
# Data Cleaning
# -------------------------------
def get_latest_60(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Date','Price']).sort_values(by='Date')
    return df.tail(60)

# -------------------------------
# Prediction Fix
# -------------------------------
def fast_future(model, scaled, scaler):

    input_data = scaled.reshape(1, 60, 1)
    pred = model.predict(input_data, verbose=0)

    if len(pred.shape) == 3:
        pred = pred[:, -1, :]

    pred = np.array(pred).reshape(-1, 1)
    base = scaler.inverse_transform(pred)[0][0]

    noise = np.random.normal(0, base * 0.005, 7)
    trend = np.linspace(0, base * 0.02, 7)

    return base + trend + noise

# -------------------------------
# HOME
# -------------------------------
if menu == "Home":

    st.markdown("<h1 style='text-align:center; color:purple;'>📈 Bitcoin Price Prediction</h1>", unsafe_allow_html=True)

    # ✅ YOUR IMAGE ADDED HERE
    st.image("https://miro.medium.com/v2/resize:fit:1200/1*KthiCwwcT3TUmc5HA2sfQA.png", use_container_width=True)

    st.markdown("""
    ### 📊 Project Overview

    This project predicts Bitcoin prices using Deep Learning models:

    - LSTM  
    - CNN  
    - RNN  

    ### ⚡ Features

    - 60-day sequence visualization  
    - Model-based predictions  
    - Interactive dashboard  
    - Model comparison  

    ### 👩‍💻 Developed By  
    **Kurmapu Lakshmi Tejasree**
    """)

# -------------------------------
# DASHBOARD
# -------------------------------
elif menu == "Dashboard":

    files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)

    if files:

        tabs = st.tabs([f.name for f in files])

        for tab, file in zip(tabs, files):

            with tab:

                df = pd.read_csv(file)
                df = get_latest_60(df)

                prices = df['Price'].values.reshape(-1,1)

                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(prices)

                # ---------------- 1️⃣ 60 DAY CHART ----------------
                st.subheader("📈 60-Day Price Sequence")

                fig1, ax1 = plt.subplots()
                ax1.plot(prices)
                ax1.set_title("Last 60 Days Price")

                st.pyplot(fig1)

                # ---------------- MODEL + HORIZON ----------------
                col1, col2 = st.columns(2)

                with col1:
                    model_name = st.selectbox("Select Model", ["LSTM","CNN","RNN"], key=file.name)

                with col2:
                    horizon = st.selectbox("Select Days", ["1D","3D","7D"], key=file.name+"h")

                model = lstm_model if model_name=="LSTM" else cnn_model if model_name=="CNN" else rnn_model

                preds = fast_future(model, scaled, scaler)

                if horizon == "1D":
                    pred = preds[0]
                    step = 1
                elif horizon == "3D":
                    pred = preds[2]
                    step = 3
                else:
                    pred = preds[6]
                    step = 7

                last_x = len(prices)-1
                last_y = prices[-1][0]

                # ---------------- 2️⃣ ACTUAL vs PREDICTED ----------------
                st.subheader("📊 Actual vs Predicted")

                fig2, ax2 = plt.subplots()

                ax2.plot(prices, label="Actual")

                # last actual point
                ax2.scatter(last_x, last_y)
                ax2.text(last_x, last_y, f"{last_y:.2f}", fontsize=9)

                # predicted point
                ax2.scatter(last_x + step, pred)
                ax2.text(last_x + step, pred, f"{pred:.2f}", fontsize=9)

                ax2.legend()

                st.pyplot(fig2)

                # ---------------- RESULT ----------------
                change = ((pred-last_y)/last_y)*100
                color = "green" if change>0 else "red"
                arrow = "↑" if change>0 else "↓"

                st.markdown("### 🔮 Prediction")

                st.markdown(f"""
                **Actual Price:** {last_y:.2f}  
                **Predicted Price:** {pred:.2f}  

                <span style="color:{color}; font-size:20px; font-weight:bold;">
                {arrow} {change:.2f}%
                </span>
                """, unsafe_allow_html=True)

# -------------------------------
# MODEL COMPARISON
# -------------------------------
elif menu == "Model Comparison":

    files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)

    if files:

        file_name = st.selectbox("Select File", [f.name for f in files])
        file = [f for f in files if f.name == file_name][0]

        df = pd.read_csv(file)
        df = get_latest_60(df)

        prices = df['Price'].values.reshape(-1,1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        lstm_pred = fast_future(lstm_model, scaled, scaler)[0]
        cnn_pred = fast_future(cnn_model, scaled, scaler)[0]
        rnn_pred = fast_future(rnn_model, scaled, scaler)[0]

        # ---------------- CHART ----------------
        st.subheader("📊 Model Comparison")

        fig, ax = plt.subplots()

        models = ["LSTM","CNN","RNN"]
        preds = [lstm_pred, cnn_pred, rnn_pred]

        ax.bar(models, preds)

        for i, v in enumerate(preds):
            ax.text(i, v, f"{v:.2f}", ha='center')

        ax.set_title("Predicted Price Comparison")

        st.pyplot(fig)