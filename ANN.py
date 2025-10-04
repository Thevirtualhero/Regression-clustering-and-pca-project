import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ANN Diamond Price Predictor 💎", layout="wide")
st.title("ANN Diamond Price Predictor")
st.subheader("Enter the features to predict the price.")

# --- 1. Define Constants and Expected Features ---

MODEL_PATH = "ann_diamond_price_model.keras" 

# CRITICAL: Features and order as expected by the trained model
EXPECTED_FEATURES = ['carat', 'x', 'y', 'color_encoded', 'clarity_encoded'] 

# --- 2. Load Model (with Caching) ---

@st.cache_resource
def load_ann_model():
    """Loads the TensorFlow/Keras model."""
    try:
        loaded_ann_model = load_model(MODEL_PATH) 
        st.success(f"✅ ANN Model loaded from '{MODEL_PATH}'.")
        return loaded_ann_model
    except FileNotFoundError:
        st.error(f"❌ ANN Model not found at '{MODEL_PATH}'.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

loaded_ann_model = load_ann_model()

# --- 3. User Input and Feature Rendering ---

st.subheader("Enter Diamond Specifications")

input_data = {}
cols = st.columns(len(EXPECTED_FEATURES))

# Getting inputs
for i, feature in enumerate(EXPECTED_FEATURES):
    default_val = 0.0
    if feature == 'carat': default_val = 1.0
    if feature in ['color_encoded', 'clarity_encoded']: default_val = 5.0
    
    val = cols[i % len(EXPECTED_FEATURES)].number_input(
        label=feature.replace('_', ' ').title(),
        min_value=0.0,
        max_value=100.0,
        value=float(default_val),
        step=0.1,
        format="%.2f"
    )
    input_data[feature] = val

# --- 4. Prediction Logic ---

if st.button("Predict Price", type="primary"):
    if loaded_ann_model is None:
        st.error("Cannot predict: Model not loaded.")
    else:
        try:
            # 1. Create DataFrame in the CORRECT ORDER
            X_input = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
            
            # 2. Directly predict using the ANN model (assuming input is ready for prediction)
            prediction_array = loaded_ann_model.predict(X_input)
            
            # 3. Extract the price
            predicted_price = prediction_array[0][0] 
            
            # 4. Display the result
            st.markdown("---")
            st.success("✅ Price Prediction Complete")
            st.markdown(f"### Predicted Price:")
            st.markdown(f"## **₹{predicted_price:,.2f}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check if the number and order of input features match the model's expectations.")