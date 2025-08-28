import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ======== Page Config ========
st.set_page_config(page_title="🎯 Student Performance Prediction", layout="centered")

# ======== Custom CSS ========
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF5733;
        font-weight: bold;
        font-size: 38px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #2E86C1;
        margin-bottom: 30px;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #333333;
        font-size: 16px;
    }
    .stSelectbox div[data-baseweb="select"] {
        border: 2px solid #FF5733;
        border-radius: 8px;
        background-color: #fdfdfd;
        padding: 6px;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #FF5733, #FFC300);
    }
    </style>
""", unsafe_allow_html=True)

# ======== Title & Description ========
st.markdown("<h1 class='main-title'>🎯 Student Performance Prediction App</h1>", unsafe_allow_html=True)
st.markdown("""
    <p class='description'>
        This interactive web app predicts student performance based on demographics, parental education level, and test scores.<br>
        Select the inputs below and get instant predictions!
    </p>
""", unsafe_allow_html=True)

# ======== Inputs ========
with st.form("prediction_form"):
    gender = st.selectbox("Select Gender", ["female", "male"])
    race_ethnicity = st.selectbox("Select Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.selectbox(
        "Select Parental Level of Education",
        ["bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"]
    )
    lunch = st.selectbox("Select Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Select Test Preparation Course Status", ["none", "completed"])
    reading_score = st.slider("Select Reading Score", 0, 100, 50)
    writing_score = st.slider("Select Writing Score", 0, 100, 50)

    submitted = st.form_submit_button("Predict")

# ======== Prediction ========
if submitted:
    custom_data_obj = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_education,
        lunch=lunch,
        test_preparation_course=test_prep,
        reading_score=reading_score,
        writing_score=writing_score
    )

    data_df = custom_data_obj.get_data_as_dataframe()
    predict_obj = PredictPipeline()
    result = predict_obj.predict(data_df)

    st.success(f"🎯 Predicted Result: {result[0]:.2f}")

