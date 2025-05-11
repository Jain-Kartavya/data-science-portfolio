# House Price Prediction

## Overview
This project predicts house prices using the Kaggle House Prices dataset with regression models (Linear Regression, Random Forest). It demonstrates data preprocessing, feature engineering, model evaluation, and visualization.

## Methodology
- **Dataset**: Kaggle House Prices (loaded via CSV).
- **Preprocessing**: Handle missing values, create new feature (`rooms_per_household`), encode categorical variables.
- **Models**: Linear Regression and Random Forest.
- **Evaluation**: Root Mean Squared Error (RMSE).
- **Visualization**: Scatter plot of actual vs. predicted prices.

## Files
- `house_price_prediction.py`: Main script for data processing, modeling, and visualization.
- `app.py`: Streamlit app for interactive price prediction.
- `house_price_prediction.png`: Visualization of Random Forest predictions.
- `requirements.txt`: Dependencies.

## Setup
```bash
pip install -r requirements.txt
python house_price_prediction.py
streamlit run app.py