import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "scaler_v1.pkl", "rb") as f:
    sc = pickle.load(f)

with open(BASE_DIR / "kmeans_v1.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "feature.pkl", "rb") as f:
    feature = pickle.load(f)
st.markdown("## 📥 Enter Customer Order Details")
user_inputs = []
user_data = {}
for col in feature:
    value = st.number_input(f"{col}",step=1.0)
    user_inputs.append(value)
    user_data[col] = value
if  st.button("Predict Cluster"):
    st.write("Feature order in Streamlit:",feature)
    st.write("User inputs:",user_inputs)
    data=np.array(user_inputs).reshape(1,-1)
    scaled_data=sc.transform(data)
    cluster = model.predict(scaled_data)[0]
    st.success(f"🎯 **Predicted Cluster:{cluster}**")
    meanings={
    0: "🟢 Cluster 0 — Fast shipping + low/medium price purchase customers (value-focused customers)",
    1: "🟡 Cluster 1 — Medium shipping + medium price purchase customers (Balance behavior customers)",
    2: "🔵 Cluster 2 — Slow shipping + high price purchase customers (Premium but less time-sensitive)",
    3: "🟣 Cluster 3 — Very slow shipping + very high price / high freight purchase customers (High-value luxury customers)"
    }    
    
    st.info(f"🧠 Cluster Interpretation:{meanings.get(cluster,'No interpretation available')}")
    st.markdown("### 📋 Your Entered Data")
    st.table(pd.DataFrame([user_data]))
    st.markdown("---")
st.markdown(
    "👤 *Author:* Mohamed Sameer  \n"
    "💼 Aspiring Data Scientist | Machine Learning Enthusiast"
)
