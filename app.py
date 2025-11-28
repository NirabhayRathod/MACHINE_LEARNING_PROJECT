import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="😴 Sleep Quality Prediction", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>😴 Sleep Quality Prediction App</h1>", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=4.0, max_value=10.0, value=7.0)
        physical_activity = st.number_input("Physical Activity Level", min_value=0, max_value=100, value=50)
        stress_level = st.number_input("Stress Level", min_value=1, max_value=10, value=5)
        daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=20000, value=8000)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.selectbox("Occupation", ["Manager", "Software Engineer", "Nurse", "Engineer", "Doctor", "Salesperson", "Scientist", "Lawyer", "Teacher"])
        bmi_category = st.selectbox("BMI Category", ["Normal Weight", "Overweight", "Obese", "Underweight"])
        sleep_disorder = st.selectbox("Sleep Disorder", ["No", "Insomnia", "Sleep Apnea"])

    submitted = st.form_submit_button("Predict Sleep Quality")

if submitted:
    try:
        custom_data_obj = CustomData(
            age=int(age),
            sleep_duration=float(sleep_duration),
            physical_activity_level=int(physical_activity),
            stress_level=int(stress_level),
            daily_steps=int(daily_steps),
            gender=gender,
            occupation=occupation,
            bmi_category=bmi_category,
            sleep_disorder=sleep_disorder
        )

        data_df = custom_data_obj.get_data_as_dataframe()
        predict_obj = PredictPipeline()
        result = predict_obj.predict(data_df)

        st.success(f"🎯 Predicted Sleep Quality: {result[0]}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")