import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# --------------------------------------------------
# Load Trained Pipeline
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(
        "Unable to load the trained model. "
        "Please make sure 'churn_pipeline.pkl' is in the same folder as app.py."
    )
    # st.error(f"Error loading model: {e}")
    # st.exception(e)
    st.stop()


# --------------------------------------------------
# Title and Description
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    """
    This application predicts whether a customer is likely to churn based
    on their demographic information, service details, account information,
    and customer behavior.
    
    Enter the customer's information below and click **Predict Churn**.
    """
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=112,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    region = st.selectbox(
        "Region",
        ["North", "South", "East", "West"]
    )

    tenure_months = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=120,
        value=12,
        step=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0,
        step=10.0
    )

    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


with col2:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Bank transfer",
            "Credit card",
            "Mailed check"
        ]
    )

    num_support_calls = st.number_input(
        "Number of Support Calls",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    late_payments_last_year = st.number_input(
        "Late Payments Last Year",
        min_value=0,
        max_value=12,
        value=0,
        step=1
    )

    avg_monthly_usage_gb = st.number_input(
        "Average Monthly Usage (GB)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):

    try:

        # Create input DataFrame
        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "region": [region],
            "tenure_months": [tenure_months],
            "monthly_charges": [monthly_charges],
            "total_charges": [total_charges],
            "contract_type": [contract_type],
            "internet_service": [internet_service],
            "tech_support": [tech_support],
            "online_security": [online_security],
            "paperless_billing": [paperless_billing],
            "payment_method": [payment_method],
            "num_support_calls": [num_support_calls],
            "late_payments_last_year": [late_payments_last_year],
            "avg_monthly_usage_gb": [avg_monthly_usage_gb]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display prediction
        st.subheader("Prediction")

        if prediction == "Yes":

            st.error("⚠️ Churn Prediction: YES")

            st.write(
                "The model predicts that this customer is likely to churn."
            )

        else:

            st.success("✅ Churn Prediction: NO")

            st.write(
                "The model predicts that this customer is not likely to churn."
            )

        # --------------------------------------------------
        # Prediction Probability
        # --------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            yes_index = list(classes).index("Yes")

            churn_probability = probabilities[yes_index]

            st.subheader("Churn Probability")

            st.progress(float(churn_probability))

            st.write(
                f"Probability of Churn: "
                f"**{churn_probability * 100:.2f}%**"
            )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction. "
            "Please check your inputs and try again."
        )

        st.exception(e)

# Footer
# st.markdown(
#     """
#     <hr>
#     <div style="text-align: center; color: gray; font-size: 14px;">
#         Customer Churn Prediction App<br>
#         Built with Python, Scikit-learn & Streamlit<br>
#         © 2026 Zubair Naseer
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# updated Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray; font-size: 14px;">
        <p style="margin-bottom: 5px;">
            <strong>Customer Churn Prediction</strong>
        </p>
        <p style="margin-top: 0;">
            Developed by 
            <a href="https://github.com/iamzubairnaseer" target="_blank">
                Zubair Naseer
            </a>
            &nbsp;|&nbsp;
            <a href="https://github.com/iamzubairnaseer/data_science-streamlit-project" target="_blank">
                GitHub Repository
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
