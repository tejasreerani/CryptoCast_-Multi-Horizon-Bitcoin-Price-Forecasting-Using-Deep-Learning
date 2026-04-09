import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="CryptoCast", layout="wide")

# ===============================
# LOAD MODEL (FAST)
# ===============================
@st.cache_resource
def load_my_model():
    return load_model(
        "E:/vscode/CryptoCast_bitcoin_project/rnn_model.h5",
        compile=False
    )

model = load_my_model()

# ===============================
# SIDEBAR
# ===============================
page = st.sidebar.selectbox("Navigation", ["Home", "Dashboard"])

# ===============================
# HOME PAGE
# ===============================
if page == "Home":

    st.title("📊 CryptoCast: Bitcoin Price Prediction")
    st.markdown("### 👩‍💻 Developed by: **Lakshmi Tejasree Kurmapu**")

    st.image(
        "https://miro.medium.com/v2/resize:fit:1200/1*KthiCwwcT3TUmc5HA2sfQA.png",
        use_container_width=True
    )

    st.markdown("""
    ## 📌 Project Overview

    This project predicts Bitcoin prices using RNN.

    ---
    🚀 Go to Dashboard
    """)

# ===============================
# DASHBOARD
# ===============================
else:

    st.title("📈 Bitcoin Prediction Dashboard")

    df = pd.read_csv("E:/vscode/CryptoCast_bitcoin_project/Bitcoin Historical Data (1).csv")

    # CLEAN DATA
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.ffill().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Price"]].values)

    index = st.selectbox(
        "Select Price",
        range(60, len(df)-7),
        format_func=lambda x: f"{df.iloc[x]['Price']:.2f}"
    )

    input_seq = scaled[index-60:index]

    # ===============================
    # PREDICTION
    # ===============================
    def predict(days, seq):
        temp = seq.copy()
        preds = []

        for _ in range(days):
            x = temp[-60:].reshape(1, 60, 1)
            y = model.predict(x, verbose=0)[0][0]
            y += np.random.normal(0, 0.002)

            temp = np.vstack([temp, [[y]]])
            preds.append(y)

        return np.array(preds)

    def inverse(val):
        return scaler.inverse_transform([[val]])[0][0]

    p1 = inverse(predict(1, input_seq)[0])
    p3 = inverse(predict(3, input_seq)[-1])
    p7 = inverse(predict(7, input_seq)[-1])

    a1 = float(df.iloc[index]["Price"])
    a3 = float(df.iloc[index+2]["Price"])
    a7 = float(df.iloc[index+6]["Price"])

    # ===============================
    # TABLE (BOLD HEADERS + ARROWS)
    # ===============================
    st.subheader("📋 Prediction Table")

    def pct_arrow(val):
        if val > 0:
            return f"<span style='color:green;font-weight:bold;'>▲ {val:.2f}%</span>"
        else:
            return f"<span style='color:red;font-weight:bold;'>▼ {val:.2f}%</span>"

    pc1 = (p1 - a1) / a1 * 100
    pc3 = (p3 - a3) / a3 * 100
    pc7 = (p7 - a7) / a7 * 100

    table_html = f"""
    <style>
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th {{
        background-color: #262730;
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 10px;
    }}
    td {{
        text-align: center;
        padding: 8px;
    }}
    </style>

    <table>
        <tr>
            <th>Horizon</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>% Change</th>
        </tr>
        <tr>
            <td>1D</td>
            <td>{a1:.2f}</td>
            <td>{p1:.2f}</td>
            <td>{pct_arrow(pc1)}</td>
        </tr>
        <tr>
            <td>3D</td>
            <td>{a3:.2f}</td>
            <td>{p3:.2f}</td>
            <td>{pct_arrow(pc3)}</td>
        </tr>
        <tr>
            <td>7D</td>
            <td>{a7:.2f}</td>
            <td>{p7:.2f}</td>
            <td>{pct_arrow(pc7)}</td>
        </tr>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    # ===============================
    # WATERFALL CHART
    # ===============================
    st.subheader("📊 Prediction Movement (Waterfall)")

    diff1 = p1 - a1
    diff3 = p3 - a3
    diff7 = p7 - a7

    fig = go.Figure(go.Waterfall(
        x=["1D", "3D", "7D"],
        y=[diff1, diff3, diff7],
        measure=["relative", "relative", "relative"]
    ))

    st.plotly_chart(fig, use_container_width=True)
    # ===============================
    # ACTUAL VS PREDICTED LINE CHART
    # ===============================
    st.subheader("📈 Actual vs Predicted Prices")

    horizons = ["1D", "3D", "7D"]
    actuals = [a1, a3, a7]
    preds = [p1, p3, p7]

    fig2 = go.Figure()

    # Actual line with data labels
    fig2.add_trace(go.Scatter(
        x=horizons,
        y=actuals,
        mode="lines+markers+text",
        name="Actual",
        line=dict(color="blue", width=3),
        marker=dict(size=8),
        text=[f"{val:.2f}" for val in actuals],
        textposition="top center"
    ))

    # Predicted line with data labels
    fig2.add_trace(go.Scatter(
        x=horizons,
        y=preds,
        mode="lines+markers+text",
        name="Predicted",
        line=dict(color="orange", width=3, dash="dash"),
        marker=dict(size=8),
        text=[f"{val:.2f}" for val in preds],
        textposition="bottom center"
    ))

    fig2.update_layout(
        xaxis_title="Forecast Horizon",
        yaxis_title="Bitcoin Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig2, use_container_width=True)
