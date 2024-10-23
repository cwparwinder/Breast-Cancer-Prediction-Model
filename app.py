import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

def main():
    st.title("Breast Cancer Prediction App")

    model_path = "Breast_Cancer Model.keras"
    model = load_model(model_path)

    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    st.subheader("Data Frame")
    st.write(data_frame)

    st.subheader("Make a Prediction")
    user_input = {}
    for feature in breast_cancer_dataset.feature_names:
        user_input[feature] = st.number_input(f"Input {feature}", min_value=0.0, step=0.1)

    user_input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(user_input_df)
        prediction_label = "Malignant" if prediction[0][0] < 0.5 else "Benign"
        prediction_proba = prediction[0][0]

        st.write("Prediction:", prediction_label)
        st.write("Probability of Malignant:", f"{1 - prediction_proba:.2f}")
        st.write("Probability of Benign:", f"{prediction_proba:.2f}")

if __name__ == "__main__":
    main()

