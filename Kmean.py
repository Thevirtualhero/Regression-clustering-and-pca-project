import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="K-Means Cluster Predictor 💎", layout="wide")
st.title("K-Means Diamond Cluster Predictor")

# --- 1. Define Constants and Mappings ---

MODEL_PATH = "kmeans_diamond_model.pkl" 

# CRITICAL: This MUST match the feature order used during training
EXPECTED_FEATURES = ['carat', 'price', 'cut_encoded'] 

# Cluster names mapping
CLUSTER_NAMES = {
    0: "Premium Heavy Diamonds", 
    1: "Affordable Small Diamonds",
    2: "Mid-range Balanced Diamonds",
    3: "Very Good Value Buys",
    4: "Fair Low-Carat Diamonds"
}

# --- 2. Load Model (with Caching) ---

@st.cache_resource
def load_kmeans_model():
    """Loads the K-Means model from the pickle file."""
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_kmeans = pickle.load(f)
        st.success(f"✅ K-Means Model loaded from '{MODEL_PATH}'.")
        return loaded_kmeans
    except FileNotFoundError:
        st.error(f"❌ K-Means model not found at '{MODEL_PATH}'. Ensure the file is correct.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

loaded_kmeans = load_kmeans_model()

# --- 3. User Input (Unscaled Values) ---

st.subheader("Enter Diamond Values")

input_data = {}
cols = st.columns(len(EXPECTED_FEATURES))

# Render inputs based on the expected feature order (Using typical real-world ranges)
for i, feature in enumerate(EXPECTED_FEATURES):
    default_val = 0.0
    if feature == 'carat': default_val = 1.0
    elif feature == 'price': default_val = 5000.0
    elif feature in ['cut_encoded']: default_val = 4.0

    val = cols[i].number_input(
        label=feature.replace('_', ' ').title(),
        # Giving wide ranges for user flexibility
        min_value=0.0, 
        max_value=20000.0 if feature == 'price' else 10.0, 
        value=float(default_val),
        step=0.1,
        format="%.2f"
    )
    input_data[feature] = val

# --- 4. Prediction Logic (No Scaling) ---

if st.button("Predict Cluster Segment", type="primary"):
    if loaded_kmeans is None:
        st.error("Cannot predict: Model failed to load.")
    else:
        try:
            # 1. Create DataFrame in the CORRECT ORDER
            X_input = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
            
            # 2. Predict the numeric ID directly on the unscaled data (AS REQUESTED)
            numeric_prediction = loaded_kmeans.predict(X_input)
            
            # 3. Map ID to Name
            cluster_id = numeric_prediction[0]
            cluster_name = CLUSTER_NAMES.get(cluster_id, "UNKNOWN CLUSTER")
            
            # 4. Display Result
            st.markdown("---")
            st.success("✅ Prediction Complete")
            st.markdown(f"### Predicted Market Segment (Cluster {cluster_id}):")
            st.markdown(f"## **{cluster_name}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Prediction failed. Check the number and order of input features.")