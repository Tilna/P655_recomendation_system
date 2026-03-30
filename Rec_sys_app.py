#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle


# In[2]:


# load data 
product_features = pd.read_csv('product_features.csv')


# In[7]:


# recommendation function
def recommend_similar_products(product_id, top_n=5):
    if product_id not in product_features['productid'].values:
        return None
    cluster_id = product_features[
        product_features['productid'] == product_id]['km_cluster'].values[0]
    similar_products = product_features[
        product_features['km_cluster'] == cluster_id]
    similar_products = similar_products[
        similar_products['productid'] != product_id]
    similar_products = similar_products.sort_values(
        by=['rating_count', 'avg_rating'],
        ascending=False )
    
    return similar_products[['productid', 'avg_rating', 'rating_count']].head(top_n)

# UI
st.title("🛒 Product Recommendation System")

# 🔥 Dropdown instead of text input
product_list = product_features['productid'].unique()

selected_product = st.selectbox(
    "Select a Product",
    product_list)
selected_data = product_features[
    product_features['productid'] == selected_product]

st.write("### Selected Product Details")
st.dataframe(selected_data[['productid', 'avg_rating', 'rating_count']])

# button
if st.button("Recommend"):
    result = recommend_similar_products(selected_product)
    
    if result is not None:
        st.write("### Recommended Products")
        st.dataframe(result)
    else:
        st.write("Product not found!")


# In[ ]:




