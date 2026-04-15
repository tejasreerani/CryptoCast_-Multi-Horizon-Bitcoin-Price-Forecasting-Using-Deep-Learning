# 🚀 CryptoCast: Deep Learning-Based Bitcoin Price Prediction

An advanced deep learning project for forecasting Bitcoin prices using multiple architectures including RNN, CNN, LSTM, and Transformer models. This project provides interactive visualizations and trend analysis through a Streamlit web application.

---

## 📌 Overview

Bitcoin price prediction is a challenging time-series problem due to its high volatility. This project explores multiple deep learning techniques to capture temporal patterns and generate accurate forecasts.

The application allows users to visualize historical data, compare model predictions, and analyze future trends interactively.

---

## ✨ Features

* 📊 Interactive Bitcoin price visualization
* 🤖 Multiple Deep Learning Models:

  * RNN (Recurrent Neural Network)
  * CNN (1D Convolutional Neural Network)
  * LSTM (Long Short-Term Memory)
  * Transformer (Attention-based model)
* 📈 Future price prediction
* 🔺 Trend indicators (positive/negative arrows)
* 📉 Model comparison dashboard
* ⚡ Fast and interactive UI using Streamlit

---

## 🧠 Models Used

| Model       | Description                                          |
| ----------- | ---------------------------------------------------- |
| RNN         | Captures sequential dependencies in time-series data |
| CNN         | Extracts local temporal patterns using convolution   |
| LSTM        | Handles long-term dependencies effectively           |
| Transformer | Uses attention mechanism for improved forecasting    |

---

## 📊 Model Evaluation

The models are evaluated using standard regression metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

The project also includes:

* 📉 Prediction vs Actual comparison
* 📊 Visual performance comparison across models

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * Pandas, NumPy
  * Scikit-learn
  * TensorFlow / Keras
  * PyTorch (for Transformer)
  * Matplotlib, Seaborn, Plotly
  * Streamlit

---

## 📂 Project Structure

```
CryptoCast-DeepLearning-Bitcoin-Prediction/
│
├── app.py                  # Streamlit application
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── .gitignore
│
├── data/                  # Dataset
│   └── Bitcoin Historical Data (1).csv
│
├── models/                # Saved models

```

---

## ⚙️ Installation

```bash
# Clone the repository
https://github.com/tejasreerani/CryptoCast_-Multi-Horizon-Bitcoin-Price-Forecasting-Using-Deep-Learning

# Navigate to project folder
cd CryptoCast-DeepLearning-Bitcoin-Prediction

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

* Home page
<img width="709" height="490" alt="image" src="https://github.com/user-attachments/assets/03eadc79-5a8f-41a3-a351-5697d3c6cefb" />

* Model comparision
  <img width="713" height="461" alt="image" src="https://github.com/user-attachments/assets/65f9340e-b6c9-449d-8638-01f82cf71033" />

* Actual Vs Predicted graphs
  <img width="688" height="440" alt="image" src="https://github.com/user-attachments/assets/45da390a-b075-4ff0-bd0a-724763413b32" />

----

## Requirements.txt
 * streamlit
 * pandas
 * numpy
 * scikit-learn
 * tensorflow
 * plotly
 * matplotlib
 * seaborn


## 🚀 Key Highlights

* Implemented **multiple deep learning architectures** in a single project
* Built a **comparative forecasting system**
* Designed **interactive analytics dashboard**
* Integrated **trend visualization with directional indicators**
* Applied **time-series preprocessing and scaling techniques**

---

## 🔮 Future Enhancements

* 📡 Real-time Bitcoin data integration (API)
* ⚙️ Hyperparameter tuning for improved accuracy
* 🤝 Ensemble modeling (combine all models)
* 📊 Advanced financial indicators integration

---

## 👩‍💻 Author

**Lakshmi Tejasree Kurmapu**

---
## 📌 Disclaimer

This project is for educational purposes only and should not be used for financial investment decisions.

---
