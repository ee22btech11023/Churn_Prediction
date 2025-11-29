# ==========================================
# PART 1: train_model.py
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import joblib

# 1. Load Data
df = pd.read_csv("customer_churn.csv")

# 2. Preprocessing
df.drop('customerID', axis='columns', inplace=True)

df = df[df.TotalCharges != ' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)

df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                  'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

df['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

# [FIX] Added dtype=int to ensure we get integers, not booleans (which newer pandas does by default)
df2 = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

X = df2.drop('Churn', axis='columns')
y = df2['Churn']

# [FIX] Explicitly cast X to float32 to avoid any object dtype issues
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 3. Build Model
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=100, verbose=0)
print("Training complete.")

# 4. Save Artifacts
model.save('churn_model.keras')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X_train.columns, 'model_columns.joblib')
# Save sample for SHAP
joblib.dump(X_train.sample(100), 'train_sample.joblib')

print("Model and artifacts saved successfully.")