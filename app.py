import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("💳 Credit Card Fraud Detection")
st.write("Manually enter transaction details below to check if it's fraudulent.")

st.markdown("""
ℹ️ **Feature Guide:**

- `V1`–`V10` are anonymized PCA features. Enter values typically in the range **-10.0 to 10.0**.
- `Amount`: Enter the transaction amount (e.g., 100.0).
- `Time`: Enter the number of seconds since the first transaction (e.g., 10000.0).
""")

# Input form
input_data = {}
for col in ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']:
    input_data[col] = st.number_input(f"{col} (Suggested: -10 to +10)", value=0.0, step=0.1, format="%.2f")

amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
time = st.number_input("Time (in seconds since first transaction)", min_value=0.0, value=10000.0)
hour = st.number_input("Hour (0 to 23)", min_value=0, max_value=23, value=12)

# Predict button
if st.button("🔍 Predict Transaction"):
    # Scale Amount and Time
    amount_scaled = scaler.transform([[amount]])[0][0]
    time_scaled = scaler.transform([[time]])[0][0]

    # Create DataFrame from V1-V10
    input_df = pd.DataFrame([input_data])

    # Add V11–V28 as zeros
    for i in range(11, 29):
        input_df[f'V{i}'] = 0.0

    input_df['Hour'] = hour
    input_df['Amount_Scaled'] = amount_scaled
    input_df['Time_Scaled'] = time_scaled

    # Reorder columns
    final_cols = [f'V{i}' for i in range(1, 29)] + ['Hour', 'Amount_Scaled', 'Time_Scaled']
    input_df = input_df[final_cols]

    # Predict
    pred = model.predict(input_df)[0]
    result = "⚠️ Fraud Detected!" if pred == 1 else "✅ Legitimate Transaction"
    st.success(f"Prediction: {result}")