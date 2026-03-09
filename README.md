# Stock Price Predictor

## Overview

This project predicts short-term stock price movements using Machine Learning.
The system downloads historical stock data from Yahoo Finance, performs feature engineering using technical indicators, trains multiple ML models, and forecasts future stock prices.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Statsmodels
* Plotly

---

## Machine Learning Models

The following models are trained and compared:

* Linear Regression
* Random Forest
* XGBoost
* LightGBM

The best model is selected based on **Directional Accuracy**.

---

## Features

* Automatic stock data download using Yahoo Finance
* Technical indicators (RSI, MACD, Bollinger Bands)
* Feature engineering for financial time-series
* Walk-forward cross validation
* Multi-model comparison
* 5-day stock price forecasting
* Interactive visualization using Plotly

---

## Installation

Clone the repository:

git clone https://github.com/tanmay-005/stock-price-predictor.git

Move into the project directory:

cd stock-price-predictor

Install dependencies:

pip install -r requirements.txt

---

## Run the Program

python stock_predictor.py

Example input:

Enter company ticker:
TSLA

Enter end date:
2026-03-09

---

## Output

The program will:

* Download stock data
* Train ML models
* Evaluate performance
* Select the best model
* Predict the next 5 trading days
* Display interactive charts

---

## Future Improvements

* Deep Learning models (LSTM)
* Real-time stock API
* Web dashboard for predictions
* Portfolio optimization

---

## Disclaimer

This project is for educational purposes only and should not be used for financial trading decisions.
