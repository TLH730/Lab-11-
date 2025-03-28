import streamlit as st
import numpy as np
import tensorflow as tf

# Define a custom MSE function if needed (for loading the model)
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the saved TensorFlow model using custom_objects
model = tf.keras.models.load_model(
    "tf_bridge_model.h5",
    custom_objects={"mse": custom_mse}
)

# Helper function to one-hot encode the Material category
def encode_material(material):
    """
    One-hot encoding for the 'Material' variable.
    Adjust the categories if your model was trained with different ones.
    """
    materials = ["Steel", "Concrete", "Composite"]
    encoding = [0] * len(materials)
    if material in materials:
        idx = materials.index(material)
        encoding[idx] = 1
    return encoding

# Build the Streamlit UI
st.title("Bridge Condition Prediction")

# Input fields for the predictors
span_ft = st.number_input("Span_ft (bridge span in feet):", min_value=0.0, value=100.0)
deck_width_ft = st.number_input("Deck_Width_ft (deck width in feet):", min_value=0.0, value=20.0)
age_years = st.number_input("Age_Years (age of the bridge):", min_value=0, value=30)
num_lanes = st.number_input("Num_Lanes (number of lanes on the bridge):", min_value=1, value=2)
material = st.selectbox("Material:", ["Steel", "Concrete", "Composite"])
# Condition_Rating UI is still provided for display or future use,
# but it's not used in the prediction because the model was trained with 7 features.
condition_rating = st.slider("Condition_Rating (1=Poor, 5=Excellent):", min_value=1, max_value=5, value=3)

if st.button("Predict"):
    # Prepare input for the model:
    # Use the four numerical inputs and the one-hot encoded material (totaling 7 features).
    mat_encoding = encode_material(material)
    input_data = np.array([
        span_ft,
        deck_width_ft,
        age_years,
        num_lanes,
        *mat_encoding  # this adds 3 features
    ], dtype=float).reshape(1, -1)

    # Get prediction from the model
    prediction = model.predict(input_data)

    # Display the prediction result
    st.subheader("Prediction Result")
    st.write("Predicted value:", float(prediction[0][0]))

