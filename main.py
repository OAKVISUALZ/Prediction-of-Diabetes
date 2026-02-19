import streamlit as st

# ... (keep your existing imports and load_model_from_s3 function) ...

st.title("Diabetes Prediction App")

# 1. Create inputs for the user
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    model = load_model_from_s3()
    
    # Prepare the features
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Make Prediction
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("Result: Diabetic")
    else:
        st.success("Result: Non-Diabetic")
