import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model(
    "models.h5",
    compile=False
)

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("one_hot_encoder_geography.pkl", "rb") as file:
    onehot_encoder_geography = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# User Inputs
credit_score = st.slider("Credit Score", 300, 900, 650)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geography.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 100, 30)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4,
    1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

if st.button("Predict"):

    # Create DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Encode Gender
    input_data["Gender"] = label_encoder_gender.transform(
        input_data["Gender"]
    )

    # Encode Geography
    geography_encoded = onehot_encoder_geography.transform(
        input_data[["Geography"]]
    ).toarray()

    geography_encoded_df = pd.DataFrame(
        geography_encoded,
        columns=onehot_encoder_geography.get_feature_names_out(
            ["Geography"]
        )
    )

    # Combine data
    input_data = pd.concat(
        [
            input_data.drop("Geography", axis=1),
            geography_encoded_df
        ],
        axis=1
    )

    # Scale data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)

    prediction_probability = prediction[0][0]

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: {prediction_probability:.2%}"
    )

    if prediction_probability > 0.5:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is unlikely to churn.")