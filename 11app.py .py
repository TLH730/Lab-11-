import streamlit as st
import numpy as np
import tensorflow as tf

# 1. Load the saved TensorFlow model
@st.cache_resource
def load_saved_model():
    model = tf.keras.models.load_model("tf_bridge_model.h5")
    return model

model = load_saved_model()

# 2. Define a helper function to encode the Material category
def encode_material(material):
    """
    Example of one-hot encoding for the 'Material' variable.
    Adjust according to the categories your model was trained on.
    """
    materials = ["Steel", "Concrete", "Composite"]
    encoding = [0] * len(materials)
    if material in materials:
        idx = materials.index(material)
        encoding[idx] = 1
    return encoding

# 3. Build the Streamlit UI
st.title("Bridge Condition Prediction")

# Input fields
span_ft = st.number_input("Span_ft (bridge span in feet):", min_value=0.0, value=100.0)
deck_width_ft = st.number_input("Deck_Width_ft (deck width in feet):", min_value=0.0, value=20.0)
age_years = st.number_input("Age_Years (age of the bridge):", min_value=0, value=30)
num_lanes = st.number_input("Num_Lanes (number of lanes on the bridge):", min_value=1, value=2)
material = st.selectbox("Material:", ["Steel", "Concrete", "Composite"])
condition_rating = st.slider("Condition_Rating (1=Poor, 5=Excellent):", min_value=1, max_value=5, value=3)

# Button to run prediction
if st.button("Predict"):
    # 4. Prepare the input for the model
    mat_encoding = encode_material(material)
    # Example input shape: [Span_ft, Deck_Width_ft, Age_Years, Num_Lanes, Material_OneHot..., Condition_Rating]
    # Make sure the order/shape here matches how your model was trained.
    input_data = np.array([
        span_ft,
        deck_width_ft,
        age_years,
        num_lanes,
        *mat_encoding,
        condition_rating
    ], dtype=float)

    # Reshape to (1, -1) because we are predicting for a single example
    input_data = input_data.reshape(1, -1)

    # 5. Get prediction from the model
    prediction = model.predict(input_data)

    # 6. Display the prediction result
    # Depending on your model, 'prediction' may be a single value or multiple values.
    st.subheader("Prediction Result")
    st.write("Predicted value:", float(prediction[0][0]))
