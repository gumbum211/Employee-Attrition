import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import scipy.special
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background: #0f1117; }
    .stApp { background: #0f1117; color: #e8e6e0; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; }
    .risk-card { border-radius: 16px; padding: 32px; text-align: center; margin: 16px 0; }
    .risk-high { background: linear-gradient(135deg, #2d0f0f, #1a0a0a); border: 1px solid #ff4444; }
    .risk-low { background: linear-gradient(135deg, #0a1a0f, #0f2d18); border: 1px solid #22c55e; }
    .risk-label { font-size: 13px; letter-spacing: 3px; text-transform: uppercase; opacity: 0.6; margin-bottom: 8px; }
    .risk-value { font-size: 56px; font-weight: 600; font-family: 'DM Mono', monospace; line-height: 1; }
    .risk-high .risk-value { color: #ff6b6b; }
    .risk-low .risk-value  { color: #4ade80; }
    .risk-verdict { font-size: 18px; margin-top: 12px; font-weight: 500; opacity: 0.85; }
    .section-header { font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: #6b7280; margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #1f2937; }
    .insight-box { background: #161b27; border-radius: 12px; padding: 20px; margin: 8px 0; border-left: 3px solid; }
    .insight-risk  { border-color: #ef4444; }
    .insight-safe  { border-color: #22c55e; }
    .insight-text { font-size: 15px; line-height: 1.5; }
    div[data-testid="stSidebar"] { background: #0a0d14; border-right: 1px solid #1f2937; }
    .stButton > button { background: #2563eb; color: white; border: none; border-radius: 10px; padding: 14px 32px; font-family: 'DM Sans', sans-serif; font-size: 15px; font-weight: 500; width: 100%; cursor: pointer; transition: background 0.2s; }
    .stButton > button:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


# ── Load model & Initialize Explainer ────────────────────────────────────────
@st.cache_resource
def load_model_and_explainer():
    try:
        pipeline = joblib.load('pipeline.pkl')
        X_train  = joblib.load('X_train.pkl')
        df_original = joblib.load('df_original.pkl')
        
        # Step 1: Scale the background dataset (X_train or X_test)
        scaler = pipeline.named_steps['scaler']
        model  = pipeline.named_steps['model']
        X_train_scaled = scaler.transform(X_train)
        
        # Step 2: Create explainer ONCE using the full dataset
        explainer = shap.LinearExplainer(model, X_train_scaled)
        
        return pipeline, X_train, explainer, df_original
    except FileNotFoundError:
        return None, None, None, None

pipeline, X_train, explainer, df_original = load_model_and_explainer()


# ── Dropdown Options ─────────────────────────────────────────────────────────
EDUCATION_OPTIONS    = ["Below College", "College", "Bachelor", "Master", "Doctor"]
SATISFACTION_OPTIONS = ["Low", "Medium", "High", "Very High"]
JOB_LEVEL_OPTIONS    = ["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"]
PERFORMANCE_OPTIONS  = ["Low", "Good", "Excellent", "Outstanding"]
WLB_OPTIONS          = ["Bad", "Good", "Better", "Best"]


def predict_employee(raw_inputs: dict):
    # Convert inputs to a DataFrame and encode
    print("Raw inputs:")
    print(raw_inputs)
    df_new = pd.DataFrame([raw_inputs])
    temp = pd.concat([df_original.drop(columns='Attrition'), df_new], ignore_index=True)
    encoded = pd.get_dummies(temp, drop_first=True).tail(1)
    df_new = encoded.reindex(columns=X_train.columns, fill_value=0)


    # Predict
    proba = pipeline.predict_proba(df_new)[:, 1][0]
    pred = (proba >= 0.734).astype(int)

    print(f"Attrition Probability: {proba:.4f}")
    print(f"Predicted Label: {'Yes' if pred == 1 else 'No'}")

    # Scale the SINGLE input
    scaler = pipeline.named_steps['scaler']
    scaled_input = scaler.transform(df_new)

    # Step 3: Compute SHAP values for just this one employee
    shap_vals = explainer.shap_values(scaled_input)
    
    # Format as a SHAP Explanation object (extracting the [0] index so it's a 1D array for the single person)
    # This prevents shape mismatch issues in the waterfall plot
    base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    explanation = shap.Explanation(
        values       = shap_vals[0],
        base_values  = base_val,
        data         = scaled_input[0],
        feature_names= X_train.columns.tolist()
    )
    return proba, pred, explanation, df_new
    


def build_narrative(proba: float, pred: int, explanation, feature_names) -> dict:
    # Use the 1D values directly from our explanation object
    vals  = explanation.values
    pairs = sorted(zip(feature_names, vals), key=lambda x: x[1])

    risk_factors = [(f, v) for f, v in pairs if v > 0][-3:][::-1]
    safe_factors = [(f, v) for f, v in pairs if v < 0][:3]

    def humanise(name: str) -> str:
        if '_' in name:
            parts = name.split('_', 1)
            clean_category = ''.join([' '+c if c.isupper() else c for c in parts[0]]).strip()
            return f"**{clean_category}: {parts[1]}**"
        clean_name = ''.join([' '+c if c.isupper() else c for c in name]).strip()
        return f"**{clean_name}**"

    risk_sentences = []
    for feat, val in risk_factors:
        risk_sentences.append(f"{humanise(feat)} is significantly increasing the likelihood of leaving.")

    safe_sentences = []
    for feat, val in safe_factors:
        safe_sentences.append(f"{humanise(feat)} is helping to retain this employee.")

    return {"risk": risk_sentences, "safe": safe_sentences}


# ── Sidebar — input form ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Employee Profile")
    st.markdown('<div class="section-header">Personal</div>', unsafe_allow_html=True)

    age              = st.slider("Age", 18, 60, 36)
    gender           = st.selectbox("Gender", ["Male", "Female"])
    marital_status   = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
    education        = st.selectbox("Education", EDUCATION_OPTIONS, index=2)
    education_field  = st.selectbox("Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], index=0)
    distance         = st.slider("Distance From Home (km)", 1, 29, 8)

    st.markdown('<div class="section-header">Job</div>', unsafe_allow_html=True)

    department       = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], index=1)
    job_role         = st.selectbox("Job Role",
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative", "Manager",
         "Sales Representative", "Research Director", "Human Resources"], index=0)
    job_level        = st.selectbox("Job Level", JOB_LEVEL_OPTIONS, index=0)
    job_involvement  = st.selectbox("Job Involvement", SATISFACTION_OPTIONS, index=1)
    job_satisfaction = st.selectbox("Job Satisfaction", SATISFACTION_OPTIONS, index=1)
    overtime         = st.selectbox("OverTime", ["Yes", "No"], index=1)
    business_travel  = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)

    st.markdown('<div class="section-header">Compensation</div>', unsafe_allow_html=True)

    monthly_income   = st.number_input("Monthly Income ($)", 1000, 20000, 6500, step=500)
    daily_rate       = st.number_input("Daily Rate", 100, 1500, 800, step=50)
    hourly_rate      = st.number_input("Hourly Rate", 30, 100, 65, step=5)
    monthly_rate     = st.number_input("Monthly Rate", 2000, 27000, 14300, step=500)
    percent_hike     = st.slider("Percent Salary Hike (%)", 11, 25, 12)
    stock_option     = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)

    st.markdown('<div class="section-header">Experience</div>', unsafe_allow_html=True)

    total_working    = st.slider("Total Working Years", 0, 40, 3)
    num_companies    = st.slider("Num Companies Worked", 0, 9, 2)
    years_company    = st.slider("Years At Company", 0, 40, 7)
    years_role       = st.slider("Years In Current Role", 0, 18, 4)
    years_promotion  = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_manager    = st.slider("Years With Current Manager", 0, 17, 4)
    training_times   = st.slider("Training Times Last Year", 0, 6, 2)

    st.markdown('<div class="section-header">Satisfaction</div>', unsafe_allow_html=True)

    env_satisfaction  = st.selectbox("Environment Satisfaction", SATISFACTION_OPTIONS, index=2)
    rel_satisfaction  = st.selectbox("Relationship Satisfaction", SATISFACTION_OPTIONS, index=2)
    work_life_balance = st.selectbox("Work-Life Balance", WLB_OPTIONS, index=2)
    performance       = st.selectbox("Performance Rating", PERFORMANCE_OPTIONS, index=2)

    st.markdown("---")
    predict_btn = st.button("🔍 Analyse Attrition Risk")


# ── Main panel ───────────────────────────────────────────────────────────────
st.markdown("# Employee Attrition Analyser")
st.markdown("Explainable AI prediction powered by Logistic Regression + SHAP")

if pipeline is None:
    st.error("""
**Model files not found.**
Make sure `pipeline.pkl` and `X_train.pkl` are in the same directory as `app.py`.
""")
    st.stop()

if predict_btn:
    raw_inputs = {
        "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
        "Department": department, "DistanceFromHome": distance, "Education": education,
        "EducationField": education_field, "EnvironmentSatisfaction": env_satisfaction,
        "Gender": gender, "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
        "JobLevel": job_level, "JobRole": job_role, "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies,
        "OverTime": overtime, "PercentSalaryHike": percent_hike,
        "PerformanceRating": performance, "RelationshipSatisfaction": rel_satisfaction,
        "StockOptionLevel": stock_option, "TotalWorkingYears": total_working,
        "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion, "YearsWithCurrManager": years_manager,
    }

    with st.spinner("Running prediction and SHAP analysis..."):
        proba, pred, explanation, df_encoded = predict_employee(raw_inputs)

    # ── Risk card ──
    pct = proba * 100
    card_class  = "risk-high" if pred == 1 else "risk-low"
    verdict     = "⚠️ High Flight Risk" if pred == 1 else "✅ Likely to Stay"

    st.markdown(f"""
    <div class="risk-card {card_class}">
        <div class="risk-label">Attrition Probability</div>
        <div class="risk-value">{pct:.1f}%</div>
        <div class="risk-verdict">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: narrative + waterfall ──
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("### Why this prediction?")
        # Note: We pass `explanation` directly here now as it is properly formatted
        narrative = build_narrative(proba, pred, explanation, df_encoded.columns.tolist())

        if narrative["risk"]:
            st.markdown("🔴 **Primary Flight Risks:**")
            for line in narrative["risk"]:
                st.markdown(f"<div class='insight-box insight-risk'><div class='insight-text'>{line}</div></div>", unsafe_allow_html=True)

        if narrative["safe"]:
            st.markdown("🟢 **Retention Drivers:**")
            for line in narrative["safe"]:
                st.markdown(f"<div class='insight-box insight-safe'><div class='insight-text'>{line}</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### SHAP Waterfall")
        plt.close('all') 
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(7, 6))
            # No longer need [0] here because explanation is already instantiated as a 1D object
            shap.plots.waterfall(explanation, max_display=10, show=False)
            
            fig = plt.gcf()
            fig.patch.set_facecolor('#0f1117')
            plt.gca().set_facecolor('#0f1117')
            
            st.pyplot(fig)

    # ── Full feature table ──
    with st.expander("📋 View encoded input sent to model (Debug)"):
        st.dataframe(df_encoded.T.astype(str).rename(columns={0: "Encoded Value"}), use_container_width=True)

else:
    st.markdown("""
    <div style='background: #161b27; border-radius: 16px; padding: 48px; text-align: center; border: 1px dashed #2a3040; margin-top: 40px;'>
        <div style='font-size: 48px; margin-bottom: 16px;'>👈</div>
        <div style='font-size: 20px; font-weight: 500; margin-bottom: 8px;'>Fill in the employee profile</div>
        <div style='color: #6b7280; font-size: 15px;'>Adjust the sliders and dropdowns in the sidebar,<br>then click <b>Analyse Attrition Risk</b>.</div>
    </div>
    """, unsafe_allow_html=True)