
import streamlit as st
import pandas as pd, joblib
import numpy as np

st.set_page_config(page_title="Student Stress Predictor", layout="centered")

st.title("Student Lifestyle → Stress Predictor")
st.write("Enter lifestyle values and get a predicted stress level (Low / Moderate / High).")

@st.cache_resource
def load_model(path='stress_predictor_model.joblib'):
    m = joblib.load(path)
    return m['model'], m['scaler'], m['label_encoder']

model, scaler, le = load_model('/app/stress_predictor_model.joblib' if False else 'stress_predictor_model.joblib')

with st.form("input_form"):
    study = st.number_input('Study hours per day', min_value=0.0, value=5.0, step=0.1)
    extrac = st.number_input('Extracurricular hours per day', min_value=0.0, value=1.0, step=0.1)
    sleep = st.number_input('Sleep hours per day', min_value=0.0, value=7.5, step=0.1)
    social = st.number_input('Social hours per day', min_value=0.0, value=2.0, step=0.1)
    physical = st.number_input('Physical activity hours per day', min_value=0.0, value=1.0, step=0.1)
    gpa = st.number_input('GPA', min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([[study, extrac, sleep, social, physical, gpa]],
                      columns=['Study_Hours_Per_Day','Extracurricular_Hours_Per_Day','Sleep_Hours_Per_Day','Social_Hours_Per_Day','Physical_Activity_Hours_Per_Day','GPA'])
    Xs = scaler.transform(df)
    pred = model.predict(Xs)[0]
    probs = model.predict_proba(Xs)[0]
    label = le.inverse_transform([pred])[0]
    st.subheader(f"Predicted stress level: {label}")
    st.write(f"Confidence: {np.round(probs.max()*100,2)}%")
    # Show probability for each class
    classes = le.classes_
    prob_df = pd.DataFrame({'Stress_Level': classes, 'Probability': probs})
    st.table(prob_df)
    st.markdown('---')
    st.write("Feature values you entered:")
    st.json(df.to_dict(orient='records')[0])

st.caption("Model trained on the provided student lifestyle dataset. This is a demo; do not use for medical or clinical decisions.")
