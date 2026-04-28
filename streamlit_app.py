
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and feature importances
@st.cache_resource
def load_resources():
    with open('gradient_boosting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_importances.pkl', 'rb') as f:
        importance_gb = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        model_features = pickle.load(f)
    return model, importance_gb, model_features

gb, importance_gb, X_columns = load_resources()

# Streamlit UI
st.set_page_config(layout="centered", page_title="Customer Campaign Response Predictor")

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #4f46e5, #7c3aed);
                padding: 24px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <h1 style="margin:0;">📊 Customer Campaign Response Predictor</h1>
        <p style="margin:4px 0 0;">Powered by Gradient Boosting — Enter a customer profile to predict campaign response</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("👤 Enter Customer Profile")

# Input widgets
col1, col2 = st.columns(2)

with col1:
    income = st.slider('Income ($)', min_value=0, max_value=200000, value=55000, step=1000)
    age = st.slider('Age', min_value=18, max_value=89, value=42, step=1)
    recency = st.slider('Days Since Last Purchase', min_value=0, max_value=100, value=30, step=1)
    kidhome = st.selectbox('Kids at Home', options=[0, 1, 2], index=0)
    teenhome = st.selectbox('Teens at Home', options=[0, 1, 2], index=0)

with col2:
    total_spend = st.slider('Total Spending ($)', min_value=0, max_value=2500, value=600, step=50)
    total_purch = st.slider('Total Purchases', min_value=0, max_value=30, value=10, step=1)
    mnt_wines = st.slider('Wine Spending ($)', min_value=0, max_value=1500, value=200, step=10)
    mnt_meat = st.slider('Meat Spending ($)', min_value=0, max_value=1500, value=100, step=10)
    num_deals = st.slider('Number of Deal Purchases', min_value=0, max_value=15, value=2, step=1)
    num_web_vis = st.slider('Number of Web Visits/Month', min_value=0, max_value=20, value=5, step=1)
    customer_tenure = st.slider('Customer Tenure (days)', min_value=100, max_value=4000, value=1200, step=50)

# Predict button
if st.button('🔮 Predict Response'):
    # Create input data dictionary, initializing all known model features to 0
    input_data = {col: 0 for col in X_columns}

    # Update with user inputs
    input_data.update({
        'Income': income,
        'Age': age,
        'Kidhome': kidhome,
        'Teenhome': teenhome,
        'Recency': recency,
        'Customer_Tenure': customer_tenure,
        'Total_Spending': total_spend,
        'Total_Purchases': total_purch,
        'MntWines': mnt_wines,
        'MntMeatProducts': mnt_meat,
        'NumDealsPurchases': num_deals,
        'NumWebVisitsMonth': num_web_vis,
    })

    # Create DataFrame in the exact order of training columns
    X_input = pd.DataFrame([input_data])[X_columns]

    pred = gb.predict(X_input)[0]
    proba = gb.predict_proba(X_input)[0][1]

    if pred == 1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #dc2626, #db2777);
                    padding: 24px; border-radius: 12px; color: white;
                    text-align: center; font-size: 1.3rem; font-weight: 600; margin-top: 16px;">
            ✅ Likely to Respond &nbsp;|&nbsp; Confidence: {proba:.1%}
        </div>
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0;
                    border-radius: 10px; padding: 16px; margin-top: 12px;">
            💡 <strong>Recommendation:</strong> Include this customer in the next campaign wave.
            High-confidence responders drive ROI.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #059669, #0d9488);
                    padding: 24px; border-radius: 12px; color: white;
                    text-align: center; font-size: 1.3rem; font-weight: 600; margin-top: 16px;">
            ❌ Unlikely to Respond &nbsp;|&nbsp; Confidence: {(1-proba):.1%} not responding
        </div>
        <div style="background: #fef2f2; border: 1px solid #fecaca;
                    border-radius: 10px; padding: 16px; margin-top: 12px;">
            💡 <strong>Recommendation:</strong> Deprioritize for this campaign.
            Consider a re-engagement offer instead.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top:24px;'>🔍 Top Features Driving This Prediction</h3>", unsafe_allow_html=True)
    top10 = importance_gb.sort_values(ascending=False).head(10)
    colors = ['#4f46e5' if i < 3 else '#a5b4fc' for i in range(len(top10))]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    top10[::-1].plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='none')
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 10 Features Influencing Customer Response", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

# Ethics & Risks Section (optional, can be moved to a separate page or removed for brevity)
st.markdown(
    """
    <div style="margin-top: 40px;">
    <h3>⚖️ Ethical Considerations</h3>

    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                padding:16px; margin-bottom:10px;">
        <strong>🔶 Fairness & Bias</strong><br>
        Historical data may encode bias toward certain demographics.
        We audit feature importance regularly to ensure sensitive attributes
        like Age or Marital Status are not the primary drivers of predictions.
    </div>

    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                padding:16px; margin-bottom:10px;">
        <strong>🔶 Transparency & Explainability</strong><br>
        The Top 10 Feature Importance chart ensures stakeholders understand
        what drives each prediction. Probability scores are shown alongside
        every result so analysts can apply their own judgment.
    </div>

    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                padding:16px; margin-bottom:10px;">
        <strong>🔶 Data Privacy (GDPR / CCPA)</strong><br>
        No PII such as name or email is used as a model feature.
        Data is processed in-session only and never stored server-side.
    </div>

    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                padding:16px; margin-bottom:10px;">
        <strong>🔶 Informed Consent</strong><br>
        Customers should be notified that their purchase behavior influences
        which marketing campaigns they receive.
    </div>

    <h3 style="margin-top:24px;">⚠️ Risks & Mitigations</h3>

    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
                padding:16px; margin-bottom:6px;">
        <strong>🔴 Model Drift</strong><br>
        Consumer behavior changes over time, degrading model accuracy.
    </div>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                padding:16px; margin-bottom:16px;">
        ✅ <strong>Mitigation:</strong> Retrain quarterly. Monitor F1-score monthly.
        Alert if F1 drops below 0.55.
    </div>

    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
                padding:16px; margin-bottom:6px;">
        <strong>🔴 Class Imbalance</strong><br>
        Only ~15% of customers respond, so the model may favour predicting No.
    </div>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                padding:16px; margin-bottom:16px;">
        ✅ <strong>Mitigation:</strong> Stratified splits, always report Recall and F1 for class 1.
    </div>

    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
                padding:16px; margin-bottom:6px;">
        <strong>🔴 Overreliance on Model Output</strong><br>
        Teams may stop applying human judgment and follow predictions blindly.
    </div>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                padding:16px; margin-bottom:16px;">
        ✅ <strong>Mitigation:</strong> Frame as a prioritization aid, not a final decision.
        Require human sign-off before excluding any customer segment.
    </div>

    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
                padding:16px; margin-bottom:6px;">
        <strong>🔴 Data Leakage</strong><br>
        Post-campaign transactions in Total Spending may inflate performance metrics.
    </div>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                padding:16px; margin-bottom:16px;">
        ✅ <strong>Mitigation:</strong> Enforce a temporal cutoff — only use data
        recorded before the campaign launch date.
    </div>

    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
                padding:16px; margin-bottom:6px;">
        <strong>🔴 Feedback Loop</strong><br>
        Always targeting the same segments creates a blind spot for untested customers.
    </div>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                padding:16px; margin-bottom:16px;">
        ✅ <strong>Mitigation:</strong> Randomly include a 5% holdout group in each
        campaign regardless of prediction to collect unbiased ground-truth data.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)
