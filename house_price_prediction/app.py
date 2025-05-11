import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title('House Price Prediction')
st.write('Predict house prices based on features using a Random Forest model.')

# Load data
data = pd.read_csv('https://raw.githubusercontent.com/ageron/housing-data/master/datasets/housing.csv')
data = data.dropna()
data['rooms_per_household'] = data['total_rooms'] / data['households']
data = pd.get_dummies(data, columns=['ocean_proximity'])

# Train model
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# User input
st.header('Enter House Features')
total_rooms = st.slider('Total Rooms', 1, 10000, 1500)
households = st.slider('Households', 1, 5000, 500)
ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', '1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
features = pd.DataFrame({
    'total_rooms': [total_rooms],
    'households': [households],
    'rooms_per_household': [total_rooms / households],
    'ocean_proximity_NEAR BAY': [1 if ocean_proximity == 'NEAR BAY' else 0],
    'ocean_proximity_1H OCEAN': [1 if ocean_proximity == '1H OCEAN' else 0],
    'ocean_proximity_INLAND': [1 if ocean_proximity == 'INLAND' else 0],
    'ocean_proximity_NEAR OCEAN': [1 if ocean_proximity == 'NEAR OCEAN' else 0],
    'ocean_proximity_ISLAND': [1 if ocean_proximity == 'ISLAND' else 0]
})

# Predict
if st.button('Predict'):
    prediction = rf.predict(features[X.columns])[0]
    st.write(f'Predicted House Price: ${prediction:,.2f}')