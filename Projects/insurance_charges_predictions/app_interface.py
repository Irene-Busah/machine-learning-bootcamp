import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    with open("insurance_cost_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Title and Subheader
st.title("🏥 Healthcare Cost Prediction")
st.subheader("Predict your medical insurance costs with this simple tool!")

# Sidebar Info
st.sidebar.markdown("### About the Tool")
st.sidebar.markdown(
    "This tool uses a machine learning model to estimate healthcare costs based on "
    "your age, gender, BMI, smoking habits, and region."
)

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Input Fields
st.markdown("### Enter Your Details Below:")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🔢 Age", min_value=0, max_value=100, help="Your age in years.")
    sex = st.selectbox("👤 Sex", ["Male", "Female"], help="Select your biological sex.")
    bmi = st.number_input("⚖️ BMI", min_value=0.0, help="Enter your body mass index.")

with col2:
    children = st.number_input("👶 Number of Children", min_value=0, max_value=10, help="Number of dependents.")
    smoker = st.selectbox("🚬 Smoker", ["Yes", "No"], help="Do you smoke?")
    region = st.selectbox(
        "📍 Region", 
        ["Northeast", "Southeast", "Southwest", "Northwest"], 
        help="Select your residential region."
    )

# Map categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_mapping = {"Northeast": 0, "Southeast": 1, "Southwest": 2, "Northwest": 3}
region = region_mapping[region]

# Prepare input for prediction
input_data = np.array([[age, sex, bmi, children, smoker, region]])

# Predict and Show Results
if st.button("💡 Predict Healthcare Costs"):
    prediction = model.predict(input_data)
    cost = prediction[0]
    st.success(f"🎉 Predicted Healthcare Costs: **${cost:.2f}**")

    # Add prediction to history
    st.session_state["history"].append(
        {"Age": age, "Sex": "Male" if sex == 1 else "Female", 
         "BMI": bmi, "Children": children, 
         "Smoker": "Yes" if smoker == 1 else "No", 
         "Region": region, "Predicted Costs": cost}
    )

# Display Prediction History
if st.session_state["history"]:
    st.markdown("### Prediction History")
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df)

    # Visualization: Age vs Predicted Costs
    st.markdown("### Age vs Predicted Costs")
    fig, ax = plt.subplots()
    ax.scatter(history_df["Age"], history_df["Predicted Costs"], color="skyblue", edgecolor="k")
    ax.set_title("Age vs Predicted Costs")
    ax.set_xlabel("Age")
    ax.set_ylabel("Predicted Costs ($)")
    st.pyplot(fig)

    # Visualization: BMI vs Predicted Costs
    st.markdown("### BMI vs Predicted Costs")
    fig, ax = plt.subplots()
    ax.scatter(history_df["BMI"], history_df["Predicted Costs"], color="green", edgecolor="k")
    ax.set_title("BMI vs Predicted Costs")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Predicted Costs ($)")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Created with ❤️ by Irene**")
