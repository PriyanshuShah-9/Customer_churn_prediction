# Customer_churn_prediction

<img width="1897" height="913" alt="image" src="https://github.com/user-attachments/assets/760e382d-02f4-4099-9928-cd0cb3e772a4" />

A complete end-to-end project for predicting telecom customer churn using Python, ML models, and an interactive Streamlit dashboard.

### 🚀 Features
  Machine Learning Pipeline
  Synthetic data generation
  Preprocessing + feature engineering
  Models trained: Logistic Regression, Random Forest, XGBoost
  Best model auto-selected (ROC-AUC)

### Artifacts saved (.pkl, dataset, features)
  🔹 Streamlit Dashboard
  📊 Overview metrics
  📈 Model performance comparison
  🎯 Real-time churn prediction
  👥 Customer segmentation & insights

### 📁 Project Structure
  churn_prediction.py   Train + Dashboard
  churn_model.pkl
  scaler.pkl
  label_encoders.pkl
  customer_data.csv
  test_data.pkl

### 🧰 Installation
  pip install -r requirements.txt

Or install manually:
pip install pandas numpy scikit-learn xgboost streamlit plotly joblib

⚙️ Train the Model
python churn_prediction.py

🖥️ Run the Dashboard
streamlit run churn_prediction.py --dashboard

### 🧠 Tech Stack
Python • Pandas • NumPy • Scikit-Learn • XGBoost •
Streamlit • Plotly • Joblib
