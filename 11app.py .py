import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

################################################################################
# 1. Define or load your preprocessing pipeline
#    (In production, you'd typically load this from disk with joblib.load)
################################################################################
def preprocess_data(df):
    # Example target column to drop (if your new data has it):
    target_col = "Max_Load_Tons"
    if target_col in df.columns:
        df.drop(columns=[target_col], inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Identify categorical and numerical columns
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(exclude=['object']).columns.tolist()

    # Create pipelines
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # For a real production app, you'd fit on your training data once,
    # then only call transform() on new data. Here we do fit_transform()
    # just for demonstration.
    X_processed = preprocessor.fit_transform(df)
    return X_processed

################################################################################
# 2. Streamlit application
################################################################################
def main():
    # Page title
    st.title("Bridge Load Prediction App")

    st.write("""
    Upload a data file with the same structure as your training data.  
    The app will preprocess it and predict using **tf_bridge_model.h5**.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

    # If a file is uploaded, process it
    if uploaded_file is not None:
        # Load the file into a DataFrame
        if uploaded_file.name.endswith(".xlsx"):
            input_df = pd.read_excel(uploaded_file)
        else:
            input_df = pd.read_csv(uploaded_file)

        st.subheader("Raw Input Data")
        st.write(input_df.head())

        # Preprocess the data
        X_processed = preprocess_data(input_df)

        # Load the trained model
        try:
            model = tf.keras.models.load_model("tf_bridge_model.h5")
        except Exception as e:
            st.error(f"Could not load the model. Check the model file path and name. Error:\n{e}")
            return

        # Make predictions
        predictions = model.predict(X_processed)

        st.subheader("Predictions")
        st.write(predictions)
    else:
        st.info("Awaiting file upload...")

if __name__ == "__main__":
    main()
