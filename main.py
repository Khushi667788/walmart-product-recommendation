import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Walmart Recommender", layout="wide")

st.title("🛒 Walmart Product Recommendation System")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/walmart_data.csv")
    return df

data = load_data()
model = joblib.load("../models/knn_model.pkl")
user_item_matrix = joblib.load("../models/user_item_matrix.pkl")

user_ids = user_item_matrix.index.tolist()
selected_user = st.selectbox("Select a User ID", user_ids)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Filtered Data Preview")
    st.dataframe(data.head(10))

with col2:
    st.markdown("### Dataset Summary")
    st.write(data.describe(include='all'))

def get_recommendations(user_id, n_recommendations=5):
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=n_recommendations + 1)
    recommended_ids = user_item_matrix.columns[indices.flatten()][1:]
    return recommended_ids

if st.button("Recommend Products"):
    recommendations = get_recommendations(selected_user)
    st.subheader("Recommended Products")
    for rec in recommendations:
        product_info = data[data['Product_ID'] == rec]
        if not product_info.empty:
            st.write(f"🔹 Product: {product_info.iloc[0]['Product_ID']} | Category: {product_info.iloc[0]['Product_Category']} | Price: {product_info.iloc[0].get('Price', 'N/A')}")
        else:
            st.write(f"🔹 Product ID: {rec}")

st.markdown("---")
st.markdown("#### About This App")
st.write("This system provides personalized product recommendations using KNN collaborative filtering.")
st.markdown("Built with Streamlit | Data source: Walmart sales dataset")
