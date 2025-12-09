import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Customer Segmentation Using K-means")
pickle.load(open("Customer_Segmentation/scaler_v1.pkl", "rb"))
pickle.load(open("Customer_Segmentation/kmeans_v1.pkl", "rb"))
with open("Customer_Segmentation/feature.pkl", "rb") as f:
    feature = pickle.load(f)
st.write("### Enter Customer Order Details")
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
    cluster=kmean.predict(scaled_data) [0]
    st.success(f"🎯 **Predicted Cluster:{cluster}**")
    meanings={
    0: "🟢 Cluster 0 — Fast shipping + low/medium price purchases",
    1: "🟡 Cluster 1 — Medium shipping + medium price purchases",
    2: "🔵 Cluster 2 — Slow shipping + high price purchases",
    3: "🟣 Cluster 3 — Very slow shipping + very high price / high freight purchases"
    }    
    st.write("### Cluster Interpratation")
    st.info(meanings.get(cluster,"No interpreatation avaible"))
    st.write("### Your Entered Data")
    st.table(pd.DataFrame([user_data]))
