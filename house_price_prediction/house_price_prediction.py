
3. **Create `app.py`**:
```bash
cat << 'EOF' > app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.title('House Price Prediction')
st.write('Predict house prices using a Random Forest model.')

# Load data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target * 100000
data['RoomsPerHousehold'] = data['AveRooms'] / data['HouseAge']
data = data.dropna()

# Train model
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# User input
st.header('Enter House Features')
med_inc = st.slider('Median Income', 0.0, 15.0, 3.0)
house_age = st.slider('House Age', 1, 100, 30)
ave_rooms = st.slider('Average Rooms', 1, 20, 5)
population = st.slider('Population', 1, 10000, 1000)
features = pd.DataFrame({
 'MedInc': [med_inc],
 'HouseAge': [house_age],
 'AveRooms': [ave_rooms],
 'Population': [population],
 'RoomsPerHousehold': [ave_rooms / house_age]
})

# Predict
if st.button('Predict'):
 prediction = rf.predict(features[X.columns])[0]
 st.write(f'Predicted House Price: ${prediction:,.2f}')
EOF