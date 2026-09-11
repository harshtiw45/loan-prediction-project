import streamlit as st
import numpy as np
import joblib
model = joblib.load("loan_model.pkl")

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)


# =========================================
# CUSTOM CSS
# =========================================

st.markdown(
    """
    <style>

    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main Title */
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #0f172a;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 20px;
        color: #475569;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Main Card */
    .card {
        background-color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    }

    /* Feature Cards */
    .feature-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #0f172a;
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
    }

    /* Labels */
    label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================
# TITLE
# =========================================

st.markdown(
    '<div class="title">💰 Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Banking Risk Analysis Dashboard</div>',
    unsafe_allow_html=True
)


# =========================================
# TOP IMAGE
# =========================================

st.image(
    "https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1200&auto=format&fit=crop",
    use_container_width=True
)


# =========================================
# FEATURE CARDS
# =========================================

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown(
        '<div class="feature-box">🤖 Machine Learning Model</div>',
        unsafe_allow_html=True
    )

with f2:
    st.markdown(
        '<div class="feature-box">⚡ Instant Prediction</div>',
        unsafe_allow_html=True
    )

with f3:
    st.markdown(
        '<div class="feature-box">📊 Risk Analysis System</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="subtitle">Predict whether a loan application will be approved or rejected using Machine Learning</div>',
    unsafe_allow_html=True
)



# MAIN CONTAINER
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
    <h2 style="
    color:#0f172a;
    text-align:center;
    margin-top:-60px;
    ">
    Fill the Required Fields Below
    </h2>
    """,
    unsafe_allow_html=True
)


# =========================================
# INPUT FIELDS
# =========================================

col1, col2 = st.columns(2)

with col1:

    Gender = st.selectbox("Gender", ["Male", "Female"])

    Married = st.selectbox("Married", ["Yes", "No"])

    Dependents = st.selectbox("Dependents", [0, 1, 2, 3])

    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    Property_Area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

with col2:

    ApplicantIncome = st.number_input(
        "Applicant Income",
        min_value=0,
        value=0
    )

    CoapplicantIncome = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0
    )

    LoanAmount = st.number_input(
         "Loan Amount (₹)",
        min_value=0,
        value=600000
    )

    Loan_Amount_Term = st.number_input(
        "Loan Amount Term (Monthly)",
        min_value=0,
        value=0
    )

    Credit_History = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

# =========================================
# MANUAL ENCODING
# =========================================

Gender = 1 if Gender == "Male" else 0
Married = 1 if Married == "Yes" else 0
Education = 0 if Education == "Graduate" else 1
Self_Employed = 1 if Self_Employed == "Yes" else 0

if Property_Area == "Rural":
    Property_Area = 0
elif Property_Area == "Semiurban":
    Property_Area = 1
else:
    Property_Area = 2


# =========================================
# PREDICT BUTTON
# =========================================

st.write("")
st.write("")

predict_button = st.button("🔍 Predict Loan Status", use_container_width=True)

# PREDICTION
# =========================================

if predict_button:
    total_income = (
        ApplicantIncome +
        CoapplicantIncome
    )

    loan_income_ratio = (
        (LoanAmount / 1000) /
        total_income
    )

    input_data = np.array([[
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount / 1000,
        Loan_Amount_Term,
        Credit_History,
        Property_Area,
        total_income,
        loan_income_ratio
    ]])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    confidence = np.max(probability) * 100


    st.write("")
    st.write("")

    if prediction[0] == 1:

        st.markdown(
            f'<div class="result-success">✅ Loan Approved<br><br>Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="result-fail">❌ Loan Rejected<br><br>Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )


st.markdown('</div>', unsafe_allow_html=True)

    