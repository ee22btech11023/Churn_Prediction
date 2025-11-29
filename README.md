
# 📊 Telecom Customer Churn Prediction App

A machine learning application that predicts whether a telecom customer is likely to churn (leave the service). This project uses a deep learning model (Artificial Neural Network) built with **TensorFlow/Keras** and serves predictions through an interactive **Streamlit** web interface. It also provides model explainability using **SHAP** values.

## 🚀 Features

* **Custom Model Training**: Script included to train a Keras ANN model on your dataset.
* **Interactive Web App**: User-friendly interface to upload data and get predictions instantly.
* **Batch Prediction**: Supports uploading CSV files to predict churn for hundreds of customers at once.
* **Visual Analytics**: View churn statistics, risk levels, and sortable data tables.
* **Model Explainability**: Integrated SHAP (SHapley Additive exPlanations) plots to understand *why* a specific customer is predicted to churn.
* **Downloadable Results**: Export prediction results with churn probabilities to CSV.

## 📂 Project Structure

```text
├── app.py                 # The main Streamlit web application
├── train_model.py         # Script to preprocess data, train the ANN, and save artifacts
├── requirements.txt       # List of Python dependencies
├── customer_churn.csv     # The dataset used for training (ensure this is in the folder)
├── README.md              # Project documentation
└── artifacts/             # Generated after running train_model.py
    ├── churn_model.keras  # Saved Keras model
    ├── scaler.joblib      # Saved MinMaxScaler for normalization
    ├── model_columns.joblib # Saved column names for alignment
    └── train_sample.joblib  # Sample data for SHAP background distribution
````

## 🛠️ Prerequisites

  * Python 3.8 or higher
  * pip (Python package installer)

## 📥 Installation

1.  **Clone or Download this repository** to your local machine.
2.  **Navigate to the project directory**:
    ```bash
    cd "path/to/your/project"
    ```
3.  **(Optional but Recommended) Create a Virtual Environment**:
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚦 How to Run

### Step 1: Train the Model

Before running the app, you must train the model and generate the necessary artifacts (scaler, model file, etc.).

1.  Ensure `customer_churn.csv` is in the project directory.
2.  Run the training script:
    ```bash
    python train_model.py
    ```
3.  Wait for the script to finish. It will create `churn_model.keras`, `scaler.joblib`, `model_columns.joblib`, and `train_sample.joblib`.

### Step 2: Launch the Web App

Once the artifacts are generated, start the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your default web browser (usually at `http://localhost:8501`).

## 🖥️ Usage Guide

1.  **Upload Data**: On the main page, upload a CSV file. The CSV should have the same columns as the training dataset (e.g., `gender`, `Partner`, `tenure`, `MonthlyCharges`, etc.).
2.  **View Raw Data**: The app will display a preview of the uploaded data.
3.  **Predict**: Click the **"Predict Churn"** button.
4.  **Analyze Results**:
      * See the total number of customers, predicted churners, and churn rate.
      * View a highlighted table of high-risk customers.
      * Download the full prediction report as a CSV.
5.  **Explainability**: Scroll down to see SHAP plots that explain which features (e.g., High Monthly Charges, Low Tenure) contributed most to the model's decisions.

## 🔧 Troubleshooting

**Error: `ValueError: Invalid dtype: object`**

  * **Cause**: This happens if Boolean data (True/False) or String data is passed directly to the TensorFlow model.
  * **Fix**: The provided code in `app.py` and `train_model.py` already includes a fix for this by explicitly casting data to `float32` and ensuring `get_dummies` returns integers. Ensure you are using the latest version of the code provided.

**Error: `FileNotFoundError`**

  * **Cause**: You tried to run `app.py` without running `train_model.py` first.
  * **Fix**: Run `python train_model.py` to generate the missing `.joblib` and `.keras` files.

## 📦 Dependencies

  * `pandas`: Data manipulation
  * `numpy`: Numerical computations
  * `scikit-learn`: Data preprocessing (Scaling)
  * `tensorflow`: Deep Learning model
  * `joblib`: Saving/Loading artifacts
  * `streamlit`: Web interface
  * `shap`: Model explainability
  * `matplotlib`: Plotting


