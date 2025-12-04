"""
Customer Churn Prediction - End-to-End ML Project with Streamlit Dashboard
===========================================================================
This project includes:
1. Complete ML pipeline (training script)
2. Interactive Streamlit dashboard (visualization & prediction)

Usage:
- Train model: python churn_prediction.py
- Run dashboard: streamlit run churn_prediction.py --dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, accuracy_score)
import xgboost as xgb
import joblib
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_customer_data(n_samples=5000, random_state=42):
    """Generate synthetic customer churn dataset"""
    np.random.seed(random_state)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                         n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                           'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                            n_samples, p=[0.4, 0.4, 0.2]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
        'num_support_calls': np.random.randint(0, 10, n_samples),
        'avg_session_duration': np.random.uniform(10, 180, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target with logical patterns
    churn_probability = (
        (df['tenure_months'] < 12) * 0.3 +
        (df['contract_type'] == 'Month-to-month') * 0.25 +
        (df['monthly_charges'] > 100) * 0.15 +
        (df['num_support_calls'] > 5) * 0.2 +
        (df['payment_method'] == 'Electronic check') * 0.1
    )
    
    df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    return df

def preprocess_data(df):
    """Clean and engineer features"""
    df_clean = df.copy()
    
    # Drop customer_id
    if 'customer_id' in df_clean.columns:
        df_clean = df_clean.drop('customer_id', axis=1)
    
    # Feature engineering
    df_clean['charges_per_month'] = df_clean['total_charges'] / (df_clean['tenure_months'] + 1)
    df_clean['is_new_customer'] = (df_clean['tenure_months'] <= 6).astype(int)
    df_clean['is_long_term'] = (df_clean['tenure_months'] >= 48).astype(int)
    df_clean['high_spender'] = (df_clean['monthly_charges'] > df_clean['monthly_charges'].median()).astype(int)
    df_clean['support_intensity'] = df_clean['num_support_calls'] / (df_clean['tenure_months'] + 1)
    
    return df_clean

def train_models():
    """Main training pipeline"""
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 70)
    
    # Generate data
    print("\n1. Loading data...")
    df = generate_customer_data()
    print(f"✓ Dataset: {len(df)} customers, Churn rate: {df['churn'].mean():.2%}")
    
    # Preprocess
    print("\n2. Preprocessing...")
    df_clean = preprocess_data(df)
    
    # Encode categoricals
    label_encoders = {}
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
    
    print(f"✓ Created {df_clean.shape[1]} features (5 engineered)")
    
    # Split data
    print("\n3. Splitting data...")
    X = df_clean.drop('churn', axis=1)
    y = df_clean['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    print("\n4. Training models...")
    models = {}
    results = {}
    
    # Logistic Regression
    print("   [1/3] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr.predict(X_test_scaled)),
        'roc_auc': roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    }
    
    # Random Forest
    print("   [2/3] Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf.predict(X_test)),
        'roc_auc': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    }
    
    # XGBoost
    print("   [3/3] XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                  random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
        'roc_auc': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    }
    
    # Select best model
    print("\n5. Model comparison:")
    for name, res in results.items():
        print(f"   {name:20s} - Accuracy: {res['accuracy']:.4f}, ROC-AUC: {res['roc_auc']:.4f}")
    
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = models[best_model_name]
    
    print(f"\n✓ Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    # Save artifacts
    print("\n6. Saving model artifacts...")
    joblib.dump(best_model, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')
    
    # Save training data for dashboard
    df.to_csv('customer_data.csv', index=False)
    
    # Save test set for evaluation
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'results': results,
        'models': models
    }
    joblib.dump(test_data, 'test_data.pkl')
    
    print("✓ Saved: churn_model.pkl, scaler.pkl, label_encoders.pkl, customer_data.csv")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! Run dashboard: streamlit run churn_prediction.py --dashboard")
    print("=" * 70)

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

def run_dashboard():
    """Interactive Streamlit dashboard"""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        print("Error: Missing required packages. Install with:")
        print("pip install streamlit plotly")
        return
    
    st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide", page_icon="📊")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {padding: 0rem 1rem;}
        .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎯 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    # Check if models exist
    if not os.path.exists('churn_model.pkl'):
        st.error("⚠️ Model not found! Please train the model first:")
        st.code("python churn_prediction.py", language="bash")
        st.stop()
    
    # Load artifacts
    @st.cache_resource
    def load_artifacts():
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    
    @st.cache_data
    def load_data():
        df = pd.read_csv('customer_data.csv')
        test_data = joblib.load('test_data.pkl')
        return df, test_data
    
    model, scaler, label_encoders, feature_names = load_artifacts()
    df, test_data = load_data()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Go to", ["📊 Overview", "🔍 Model Performance", "🎲 Make Prediction", "📈 Customer Analysis"])
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "📊 Overview":
        st.header("📊 Business Overview")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(df)
        churned = df['churn'].sum()
        churn_rate = df['churn'].mean()
        revenue_at_risk = df[df['churn'] == 1]['monthly_charges'].sum()
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Churned Customers", f"{churned:,}")
        with col3:
            st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"-{churn_rate*10:.1%} target")
        with col4:
            st.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Distribution")
            churn_counts = df['churn'].value_counts()
            fig = px.pie(values=churn_counts.values, 
                        names=['Retained', 'Churned'],
                        color_discrete_sequence=['#00CC96', '#EF553B'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Churn by Contract Type")
            churn_by_contract = df.groupby('contract_type')['churn'].mean().reset_index()
            fig = px.bar(churn_by_contract, x='contract_type', y='churn',
                        color='churn', color_continuous_scale='Reds')
            fig.update_layout(yaxis_title="Churn Rate", xaxis_title="Contract Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tenure vs Churn")
            fig = px.histogram(df, x='tenure_months', color='churn', 
                             nbins=30, barmode='overlay',
                             color_discrete_map={0: '#00CC96', 1: '#EF553B'})
            fig.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Monthly Charges Distribution")
            fig = px.box(df, x='churn', y='monthly_charges',
                        color='churn', color_discrete_map={0: '#00CC96', 1: '#EF553B'})
            fig.update_layout(xaxis_title="Churn Status", yaxis_title="Monthly Charges ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 2: MODEL PERFORMANCE
    # ========================================================================
    elif page == "🔍 Model Performance":
        st.header("🔍 Model Performance Analysis")
        
        results = test_data['results']
        
        # Model comparison
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                        color='Accuracy', color_continuous_scale='Blues')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='ROC-AUC',
                        color='ROC-AUC', color_continuous_scale='Greens')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix (Best Model)")
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Not Churned', 'Churned'],
                       y=['Not Churned', 'Churned'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Detailed Metrics")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 3: MAKE PREDICTION
    # ========================================================================
    elif page == "🎲 Make Prediction":
        st.header("🎲 Predict Customer Churn")
        st.markdown("Enter customer details to predict churn probability")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 150, 70)
        
        with col2:
            total_charges = st.slider("Total Charges ($)", 100, 8000, 1000)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col3:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet"])
            support_calls = st.slider("Support Calls", 0, 10, 2)
            session_duration = st.slider("Avg Session Duration (min)", 10, 180, 60)
        
        if st.button("🔮 Predict Churn", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'tenure_months': [tenure],
                'monthly_charges': [monthly_charges],
                'total_charges': [total_charges],
                'contract_type': [contract],
                'payment_method': [payment],
                'internet_service': [internet],
                'online_security': [online_security],
                'tech_support': [tech_support],
                'num_support_calls': [support_calls],
                'avg_session_duration': [session_duration]
            })
            
            # Preprocess
            input_processed = preprocess_data(input_data)
            
            # Encode categoricals
            for col in label_encoders.keys():
                if col in input_processed.columns:
                    input_processed[col] = label_encoders[col].transform(input_processed[col])
            
            # Ensure correct feature order
            input_processed = input_processed[feature_names]
            
            # Predict
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "WILL CHURN" if prediction == 1 else "WILL STAY",
                         delta="High Risk" if prediction == 1 else "Low Risk")
            
            with col2:
                st.metric("Churn Probability", f"{probability[1]:.1%}")
            
            with col3:
                risk_level = "🔴 High" if probability[1] > 0.7 else "🟡 Medium" if probability[1] > 0.4 else "🟢 Low"
                st.metric("Risk Level", risk_level)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if prediction == 1:
                st.warning("⚠️ **Action Required:** This customer is at high risk of churning!")
                st.markdown("""
                **Recommended Actions:**
                - 📞 Immediate outreach by retention team
                - 💰 Offer loyalty discount or upgrade incentive
                - 📋 Review support ticket history
                - 🎁 Personalized retention offer
                """)
    
    # ========================================================================
    # PAGE 4: CUSTOMER ANALYSIS
    # ========================================================================
    elif page == "📈 Customer Analysis":
        st.header("📈 Customer Segmentation Analysis")
        
        # Customer segments
        df['segment'] = pd.cut(df['tenure_months'], 
                              bins=[0, 12, 36, 72], 
                              labels=['New (0-12m)', 'Regular (12-36m)', 'Loyal (36m+)'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn by Customer Segment")
            segment_churn = df.groupby('segment')['churn'].mean().reset_index()
            fig = px.bar(segment_churn, x='segment', y='churn',
                        color='churn', color_continuous_scale='Reds')
            fig.update_layout(yaxis_title="Churn Rate", xaxis_title="Customer Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Revenue by Segment")
            segment_revenue = df.groupby('segment')['monthly_charges'].sum().reset_index()
            fig = px.bar(segment_revenue, x='segment', y='monthly_charges',
                        color='monthly_charges', color_continuous_scale='Blues')
            fig.update_layout(yaxis_title="Total Monthly Revenue ($)", xaxis_title="Customer Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Segment Statistics")
        segment_stats = df.groupby('segment').agg({
            'churn': ['count', 'sum', 'mean'],
            'monthly_charges': ['mean', 'sum'],
            'num_support_calls': 'mean'
        }).round(2)
        segment_stats.columns = ['Total Customers', 'Churned', 'Churn Rate', 
                                'Avg Monthly Charge', 'Total Revenue', 'Avg Support Calls']
        st.dataframe(segment_stats, use_container_width=True)
        
        # High-risk customers
        st.subheader("🚨 High-Risk Customers (Sample)")
        high_risk = df[df['churn'] == 1].head(10)
        st.dataframe(high_risk[['age', 'tenure_months', 'monthly_charges', 
                                'contract_type', 'num_support_calls']], use_container_width=True)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if running with streamlit
    try:
        import streamlit as st
        # If streamlit is imported and we're in a streamlit session, run dashboard
        run_dashboard()
    except (ImportError, Exception):
        # Otherwise run training
        train_models()