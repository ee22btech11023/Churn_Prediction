# ==========================================
# PART 2: app.py
# Run this using: streamlit run app.py
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('churn_model.keras')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    train_sample = joblib.load('train_sample.joblib')
    
    # [FIX] Explicitly convert the training sample to float32
    # This prevents the "Invalid dtype: object" error in Keras
    if isinstance(train_sample, pd.DataFrame):
        train_sample = train_sample.astype(np.float32)
        
    return model, scaler, model_columns, train_sample

try:
    model, scaler, model_columns, train_sample = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please run 'train_model.py' first.")
    st.stop()

def preprocess_data(df_input, scaler, model_columns):
    # Create a copy to avoid SettingWithCopy warnings
    df = df_input.copy()
    
    # Keep Customer ID for reporting, but drop for processing
    if 'customerID' in df.columns:
        customer_ids = df['customerID']
        df.drop('customerID', axis='columns', inplace=True)
    else:
        customer_ids = df.index

    # Handle TotalCharges
    # Force convert to numeric, coerce errors to NaN, then fill with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True) 

    # Replace No internet/phone service
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)

    # Convert Yes/No to 1/0
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 
                      'PaperlessBilling', 'Churn']
    
    # Only map columns that actually exist in input
    for col in yes_no_columns:
        if col in df.columns:
            df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

    # Convert Gender
    if 'gender' in df.columns:
        df['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

    # One Hot Encoding
    # [FIX] Add dtype=int to get_dummies to avoid boolean types
    df_encoded = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)
    
    # Reindex ensures that we have exactly the same columns as the training data, in the same order
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Scaling
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_final[cols_to_scale] = scaler.transform(df_final[cols_to_scale])
    
    return df_final, customer_ids

# Streamlit UI
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("Upload a CSV file containing customer data to predict churn probability.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # Read Data
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())

    if st.button("Predict Churn"):
        with st.spinner("Preprocessing data and making predictions..."):
            # Preprocess
            X_new, cust_ids = preprocess_data(df_raw, scaler, model_columns)
            
            # [FIX] Explicitly cast input data to float32 before prediction
            X_new = X_new.astype(np.float32)
            
            # Predict
            predictions = model.predict(X_new)
            
            # Create Results DataFrame
            results = pd.DataFrame({
                'CustomerID': cust_ids,
                'Churn_Probability': predictions.flatten()
            })
            
            # Thresholding
            results['Likely_To_Churn'] = results['Churn_Probability'] > 0.5
            results['Risk_Label'] = results['Likely_To_Churn'].map({True: 'High Risk', False: 'Low Risk'})
            
            # Stats
            churn_count = results['Likely_To_Churn'].sum()
            total_count = len(results)
            churn_rate = (churn_count / total_count) * 100
            
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", total_count)
            col2.metric("Predicted Churners", churn_count)
            col3.metric("Churn Rate", f"{churn_rate:.2f}%")
            
            # Display List of Churners
            st.subheader("🚨 Customers Likely to Churn")
            churners_list = results[results['Likely_To_Churn'] == True].sort_values(by='Churn_Probability', ascending=False)
            st.dataframe(churners_list.style.background_gradient(subset=['Churn_Probability'], cmap='Reds'))
            
            # Download Button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )

            # # SHAP Analysis
            # st.subheader("🧠 Model Explainability (SHAP)")
            # st.write("Generating SHAP values to understand feature impact. This may take a moment...")
            
            # # We use KernelExplainer for Keras models.
            # explainer = shap.KernelExplainer(model.predict, train_sample)
            
            # # We calculate SHAP values for the uploaded data (limit to 50 for speed)
            # shap_limit = 50
            # if len(X_new) > shap_limit:
            #     st.warning(f"Calculating SHAP for the first {shap_limit} customers to ensure performance.")
            #     X_shap = X_new.iloc[:shap_limit]
            # else:
            #     X_shap = X_new
                
            # shap_values = explainer.shap_values(X_shap)
            
            # # SHAP Summary Plot
            # st.write("### Feature Importance (Summary Plot)")
            # # Fix for matplotlib figure context
            # fig, ax = plt.subplots()
            
            # # shap_values[0] handling
            # vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            # shap.summary_plot(vals, X_shap, show=False)
            # st.pyplot(fig)
            
            # # SHAP Bar Plot
            # st.write("### Mean Absolute SHAP Value (Bar Plot)")
            # fig2, ax2 = plt.subplots()
            # shap.summary_plot(vals, X_shap, plot_type="bar", show=False)
            # st.pyplot(fig2)