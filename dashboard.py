import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="ML Model Monitoring", layout="wide")

st.title(" Model Monitoring Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

metrics = {}
drift_status = {}

try:
    with open("metrics.txt", "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                metrics[key] = value
except:
    pass

try:
    with open("drift_status.txt", "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                drift_status[key] = value
except:
    pass

with col1:
    st.metric(
        "Model Accuracy",
        metrics.get("accuracy", "N/A"),
        delta_color="normal"
    )

with col2:
    data_drift = drift_status.get("data_drift", "Unknown")
    drift_color = "inverse" if data_drift == "True" else "normal"
    st.metric("Data Drift", data_drift)

with col3:
    concept_drift = drift_status.get("concept_drift", "Unknown")
    drift_color = "inverse" if concept_drift == "True" else "normal"
    st.metric("Concept Drift", concept_drift)

with col4:
    st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

st.markdown("---")

col_drift, col_model = st.columns(2)

with col_drift:
    st.subheader("Drift Detection")
    
    drift_data = pd.DataFrame({
        "Metric": ["Data Drift", "Concept Drift"],
        "Status": [data_drift, concept_drift],
        "Alert": ["Yes" if d == "True" else "No" for d in [data_drift, concept_drift]]
    })
    st.dataframe(drift_data, use_container_width=True)
    
    if data_drift == "True" or concept_drift == "True":
        st.error("Drift detected! Consider retraining the model.")
    else:
        st.success("Model is stable.")

with col_model:
    st.subheader("Model Information")
    try:
        import joblib
        model = joblib.load("model/model.pkl")
        st.write(f"Model Type: {type(model).__name__}")
        st.write(f"Features: age, salary")
    except Exception as e:
        st.warning(f"Could not load model: {e}")

st.markdown("---")

st.subheader("Real-time Predictions")

if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

with st.form("prediction_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col_b:
        salary = st.number_input("Salary", min_value=0, max_value=1000000, value=50000)
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            model = joblib.load("model/model.pkl")
            prediction = model.predict([[age, salary]])[0]
            prob = model.predict_proba([[age, salary]])[0]
            
            st.success(f"Prediction: {'Will Buy' if prediction == 1 else 'Won't Buy'}")
            st.info(f"Probability: {max(prob):.2%}")
            
            st.session_state.prediction_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "age": age,
                "salary": salary,
                "prediction": prediction,
                "confidence": max(prob)
            })
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if st.session_state.prediction_log:
    st.subheader("Recent Predictions")
    log_df = pd.DataFrame(st.session_state.prediction_log[-10:])
    st.dataframe(log_df, use_container_width=True)

st.markdown("---")

st.caption(f"Dashboard last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

auto_refresh = st.checkbox("Auto-refresh (every 30s)")
if auto_refresh:
    time.sleep(30)
    st.rerun()