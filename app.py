# train_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("C:\\Users\\kurup\\Downloads\\car data.csv")

# Print original column names
print("Original Columns:", df.columns.tolist())

# Clean and rename columns for consistency
df.rename(columns=lambda x: x.strip().replace(" ", "_").lower(), inplace=True)

# Drop 'car_name' if it exists
if 'car_name' in df.columns:
    df.drop('car_name', axis=1, inplace=True)

# Convert kms_driven if it's string type
if df['kms_driven'].dtype == 'object':
    df['kms_driven'] = df['kms_driven'].str.replace(',', '').astype(int)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
with open("car_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as car_model.pkl")



# app.py

import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("car_model.pkl", "rb"))

st.title("🚗 Used Car Price Estimator")

# Sidebar inputs
st.sidebar.header("Enter Car Details")

year = st.sidebar.number_input("Year", min_value=2000, max_value=2024, value=2015)
present_price = st.sidebar.number_input("Present Price (in lakhs)", value=5.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", value=30000)
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 3])
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])

# Encoding categorical variables
fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0
fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
seller_type_individual = 1 if seller_type == 'Individual' else 0
transmission_manual = 1 if transmission == 'Manual' else 0

# Input DataFrame
input_df = pd.DataFrame([{
    'year': year,
    'present_price': present_price,
    'kms_driven': kms_driven,
    'owner': owner,
    'fuel_type_diesel': fuel_type_diesel,
    'fuel_type_petrol': fuel_type_petrol,
    'seller_type_individual': seller_type_individual,
    'transmission_manual': transmission_manual
}])

# Predict and display
if st.button("Estimate Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹{round(prediction * 1e5):,}")

