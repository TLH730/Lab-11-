import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# Define a custom MSE loss function for model loading
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Paths to your saved model and preprocessing pipeline
model_path = "tf_bridge_model.h5"
pipeline_path = "preprocessing_pipeline.pkl"

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at '{model_path}'. Please ensure that the file is present in the directory.")
else:
    custom_objects = {"mse": mse_loss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Check if preprocessing pipeline file exists
if not os.path.exists(pipeline_path):
    st.error(f"Preprocessing pipeline file not found at '{pipeline_path}'. Please ensure that the file is present in the directory.")
else:
    with open(pipeline_path, "rb") as f:
        scaler = pickle.load(f)

# Only proceed if both the model and scaler are loaded
if "model" in locals() and "scaler" in locals():

    def preprocess_input(input_data):
        """
        Preprocess the input data into the format and scale expected by the model.
        """
        # Create a DataFrame from the input data
        df = pd.DataFrame([input_data])
        # One-hot encode the 'Material' column.
        df = pd.concat([df, pd.get_dummies(df["Material"], prefix="Material", drop_first=True)], axis=1)
        df = df.drop(columns=["Material"])
        # Ensure all expected columns are present.
        expected_cols = [
            "Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes", 
            "Condition_Rating", "Material_Concrete", "Material_Steel"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
        # Scale the features using the saved scaler
        X_scaled = scaler.transform(df)
        return X_scaled

    st.title("Bridge Maximum Load Prediction")
    st.write("Enter the details of the bridge:")

    span_ft = st.number_input("Span (ft)", min_value=0.0, value=250.0)
    deck_width_ft = st.number_input("Deck Width (ft)", min_value=0.0, value=40.0)
    age_years = st.number_input("Age (Years)", min_value=0.0, value=20.0)
    num_lanes = st.number_input("Number of Lanes", min_value=1, value=4)
    condition_rating = st.number_input("Condition Rating (1-5)", min_value=1, max_value=5, value=4)
    material = st.selectbox("Material", options=["Composite", "Concrete", "Steel"])

    input_data = {
        "Span_ft": span_ft,
        "Deck_Width_ft": deck_width_ft,
        "Age_Years": age_years,
        "Num_Lanes": num_lanes,
        "Condition_Rating": condition_rating,
        "Material": material
    }

    if st.button("Predict Maximum Load (Tons)"):
        X_input = preprocess_input(input_data)
        prediction = model.predict(X_input)
        predicted_load = prediction[0, 0]
        st.success(f"Predicted Maximum Load: {predicted_load:.2f} Tons")
