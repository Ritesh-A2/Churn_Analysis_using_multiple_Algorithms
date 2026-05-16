import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

num_columns = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

multi_columns = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]

binary_columns = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
]

def basic_cleaning(df1):

    df1 = df1.copy()

    # Remove customerID
    df1 = df1.drop('customerID', axis=1,errors='ignore')

    # Convert TotalCharges
    df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'],errors='coerce')

    # Fill missing values
    df1['TotalCharges'] = df1['TotalCharges'].fillna(df1['TotalCharges'].median())

    # Binary mapping
    binary_map = {
        'Yes':1,
        'No':0,
        'Female':0,
        'Male':1
    }

    for col in binary_columns:
        df1[col] = df1[col].map(binary_map)

    return df1

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1f1f1f;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #aaaaaa !important;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Main background */
.main { background: #f5f3ee; }

/* Header card */
.header-card {
    background: #0d0d0d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.header-card h1 {
    color: #ffffff;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.02em;
}
.header-card p {
    color: #888;
    margin: 0.2rem 0 0 0;
    font-size: 0.95rem;
}

/* Result cards */
.result-churn {
    background: linear-gradient(135deg, #ff3b30, #c0392b);
    border-radius: 16px;
    padding: 2rem;
    color: white;
    text-align: center;
}
.result-stay {
    background: linear-gradient(135deg, #30d158, #1a8c3a);
    border-radius: 16px;
    padding: 2rem;
    color: white;
    text-align: center;
}
.result-card h2 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    margin: 0.5rem 0;
}
.result-card p {
    font-size: 0.9rem;
    opacity: 0.85;
    margin: 0;
}
.result-emoji { font-size: 3rem; }

/* Metric chip */
.chip {
    display: inline-block;
    background: #0d0d0d;
    color: #f0f0f0;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin: 0.2rem;
}

/* Section label */
.section-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

/* Input summary table */
.summary-table {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e8e4de;
}
.summary-table table { width: 100%; border-collapse: collapse; }
.summary-table td {
    padding: 0.35rem 0.5rem;
    font-size: 0.85rem;
    border-bottom: 1px solid #f0ece6;
    color: #333;
}
.summary-table td:first-child { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.summary-table tr:last-child td { border-bottom: none; }

/* Risk bar */
.risk-label { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #666; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "logistic_churn_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div style="font-size:2.8rem;">📡</div>
    <div>
        <h1>Customer Churn Predictor</h1>
        <p>Logistic Regression · Telecom Dataset · Real-time Inference</p>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ **Model file not found!**  \nPlace `logistic_churn_model.pkl` in the same directory as this app and restart.")
    st.stop()


# ─────────────────────────────────────────
#  SIDEBAR — INPUT FORM
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧾 Customer Profile")
    st.markdown("---")

    st.markdown("**👤 Demographics**")
    gender         = st.selectbox("Gender", ["Female", "Male"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner", ["Yes", "No"])
    dependents     = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown("---")
    st.markdown("**📞 Services**")
    phone          = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_sec     = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_bkp     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot    = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_sup       = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("---")
    st.markdown("**💳 Billing**")
    contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment        = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 85.5, step=0.5)
    total          = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly), step=1.0)

    st.markdown("---")
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True, type="primary")


# ─────────────────────────────────────────
#  BUILD DATAFRAME
# ─────────────────────────────────────────
def build_df():
    return pd.DataFrame({
        'customerID': ['CUST_001'],   # default dummy ID
        'gender':           [gender],
        'SeniorCitizen':    [1 if senior == "Yes" else 0],
        'Partner':          [partner],
        'Dependents':       [dependents],
        'tenure':           [tenure],
        'PhoneService':     [phone],
        'MultipleLines':    [multi_lines],
        'InternetService':  [internet],
        'OnlineSecurity':   [online_sec],
        'OnlineBackup':     [online_bkp],
        'DeviceProtection': [device_prot],
        'TechSupport':      [tech_sup],
        'StreamingTV':      [streaming_tv],
        'StreamingMovies':  [streaming_mov],
        'Contract':         [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod':    [payment],
        'MonthlyCharges':   [monthly],
        'TotalCharges':     [total],
    })


# ─────────────────────────────────────────
#  DEFAULT STATE — show instructions
# ─────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.5rem;border:1px solid #e8e4de;height:100%">
            <div style="font-size:2rem;">📝</div>
            <div class="section-label" style="margin-top:0.8rem">Step 1</div>
            <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem">Fill the Profile</div>
            <div style="color:#666;font-size:0.88rem">Use the sidebar to enter the customer's demographics, services, and billing details.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.5rem;border:1px solid #e8e4de;height:100%">
            <div style="font-size:2rem;">🔍</div>
            <div class="section-label" style="margin-top:0.8rem">Step 2</div>
            <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem">Run Prediction</div>
            <div style="color:#666;font-size:0.88rem">Click <b>Run Prediction</b> to send the inputs through the trained logistic regression model.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.5rem;border:1px solid #e8e4de;height:100%">
            <div style="font-size:2rem;">📊</div>
            <div class="section-label" style="margin-top:0.8rem">Step 3</div>
            <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem">Read the Results</div>
            <div style="color:#666;font-size:0.88rem">See the churn verdict, probability score, risk gauge, and a full input summary.</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────
sample_data = build_df()

try:
    prediction  = model.predict(sample_data)[0]
    probability = model.predict_proba(sample_data)[0]
    churn_prob  = probability[1]
    stay_prob   = probability[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


# ─────────────────────────────────────────
#  RESULTS LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

# ── LEFT: Verdict + Gauge ──
with left:
    if prediction == 1:
        st.markdown(f"""
        <div class="result-churn result-card">
            <div class="result-emoji">🚨</div>
            <h2>LIKELY TO CHURN</h2>
            <p>This customer shows a high risk of leaving.<br>
            Churn probability: <b>{churn_prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-stay result-card">
            <div class="result-emoji">✅</div>
            <h2>LIKELY TO STAY</h2>
            <p>This customer is expected to remain active.<br>
            Retention probability: <b>{stay_prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(churn_prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "family": "Space Mono"}},
        title={"text": "Churn Risk Score", "font": {"size": 14, "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10}},
            "bar": {"color": "#ff3b30" if prediction == 1 else "#30d158"},
            "steps": [
                {"range": [0, 33],  "color": "#e8f5e9"},
                {"range": [33, 66], "color": "#fff8e1"},
                {"range": [66, 100],"color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#0d0d0d", "width": 3},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Syne",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability bar
    st.markdown('<div class="risk-label">Probability Breakdown</div>', unsafe_allow_html=True)
    prob_fig = go.Figure()
    prob_fig.add_trace(go.Bar(
        name="Stay",  x=[stay_prob*100],  y=[""],
        orientation="h", marker_color="#30d158",
        text=f"{stay_prob*100:.1f}%", textposition="inside",
        textfont=dict(color="white", size=13, family="Space Mono"),
    ))
    prob_fig.add_trace(go.Bar(
        name="Churn", x=[churn_prob*100], y=[""],
        orientation="h", marker_color="#ff3b30",
        text=f"{churn_prob*100:.1f}%", textposition="inside",
        textfont=dict(color="white", size=13, family="Space Mono"),
    ))
    prob_fig.update_layout(
        barmode="stack", height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.5, font_size=11),
        xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(prob_fig, use_container_width=True)


# ── RIGHT: Input Summary ──
with right:
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)

    rows = [
        ("Gender", gender), ("Senior Citizen", senior),
        ("Partner", partner), ("Dependents", dependents),
        ("Tenure", f"{tenure} months"),
        ("Phone Service", phone), ("Multiple Lines", multi_lines),
        ("Internet Service", internet), ("Online Security", online_sec),
        ("Online Backup", online_bkp), ("Device Protection", device_prot),
        ("Tech Support", tech_sup), ("Streaming TV", streaming_tv),
        ("Streaming Movies", streaming_mov), ("Contract", contract),
        ("Paperless Billing", paperless), ("Payment Method", payment),
        ("Monthly Charges", f"${monthly:.2f}"), ("Total Charges", f"${total:.2f}"),
    ]

    table_html = '<div class="summary-table"><table>'
    for label, val in rows:
        table_html += f"<tr><td>{label}</td><td><b>{val}</b></td></tr>"
    table_html += "</table></div>"
    st.markdown(table_html, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  FOOTER CHIPS
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<span class="chip">Model: Logistic Regression</span>'
    f'<span class="chip">Churn Prob: {churn_prob*100:.2f}%</span>'
    f'<span class="chip">Stay Prob: {stay_prob*100:.2f}%</span>'
    f'<span class="chip">Verdict: {"CHURN 🚨" if prediction==1 else "STAY ✅"}</span>',
    unsafe_allow_html=True
)
