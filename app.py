import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained models
lr_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Feature names (excluding 'Class' and 'Time') from the original dataset
feature_columns = ["V" + str(i) for i in range(1, 29)] + ["Amount"]

# Streamlit App
st.title("💳 Credit Card Fraud Detection")

st.markdown(
    "Enter transaction details below to check if it's **Fraudulent** or **Legitimate**."
)

# Create input fields for user entry
user_data = []
for feature in feature_columns:
    value = st.number_input(f"Enter value for {feature}:", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    user_data.append(value)

# Convert input to numpy array and reshape
user_data = np.array(user_data).reshape(1, -1)

# Standardize the 'Amount' feature (assuming mean=88.34, std=250.12 from dataset)
scaler = StandardScaler()
scaler.mean_ = np.array([88.34])  # Approximate mean from dataset
scaler.scale_ = np.array([250.12])  # Approximate std deviation from dataset
user_data[:, -1] = scaler.transform(user_data[:, -1].reshape(-1, 1)).flatten()

# Predict button
if st.button("🔍 Predict Fraud"):
    lr_prediction = lr_model.predict(user_data)
    rf_prediction = rf_model.predict(user_data)

    st.subheader("📝 Prediction Results:")
    st.write(f"**Logistic Regression Prediction:** {'🚨 Fraud' if lr_prediction[0] == 1 else '✅ Legitimate'}")
    st.write(f"**Random Forest Prediction:** {'🚨 Fraud' if rf_prediction[0] == 1 else '✅ Legitimate'}")

    # Highlight fraud transactions
    if lr_prediction[0] == 1 or rf_prediction[0] == 1:
        st.error("⚠️ This transaction is **potentially fraudulent**!")
    else:
        st.success("✅ This transaction is **legitimate**.")

