#!/usr/bin/env python
# coding: utf-8

# In[34]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors


# In[35]:


# Page config
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide")


# In[36]:


# Cluster labels
CLUSTER_INFO = {
    0: {"name": "High-Rated Niche",    "emoji": "⭐", "color": "#27ae60"},
    1: {"name": "Popular & Well Rated","emoji": "🔥", "color": "#e67e22"},
    2: {"name": "Low Rated",           "emoji": "📦", "color": "#e74c3c"},
}


# In[37]:


# Load saved files
@st.cache_data
def load_data():
    product_features   = pd.read_csv('product_features.xls')
    X_content_scaled   = np.load('X_content_scaled.npy')
    product_factors    = np.load('product_factors.npy')
    with open('svd_products.pkl', 'rb') as f:
        svd_products   = pickle.load(f)
    return product_features, X_content_scaled, product_factors, svd_products


# In[38]:


## building knn model
@st.cache_resource
def build_models(X_content_scaled, product_factors):
    knn_content = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    knn_content.fit(X_content_scaled)

    knn_svd = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    knn_svd.fit(product_factors)

    return knn_content, knn_svd


# In[39]:


## hybrid reccomendation function
def recommend_hybrid(product_id, product_features, knn_content, X_content_scaled,knn_svd, product_factors, svd_products, top_n=5,w_cluster=0.30,
                     w_content=0.60, w_svd=0.10):
    if product_id not in product_features['productid'].values:
        return None

    ## get cluster of selected product
    cluster_id = product_features[
        product_features['productid'] == product_id
    ]['km_cluster'].values[0]

    ## content-based candidates
    idx_cb = product_features[product_features['productid'] == product_id].index[0]
    cb_dist, cb_ind = knn_content.kneighbors(
        X_content_scaled[idx_cb].reshape(1, -1), n_neighbors=51
    )
    cb_scores = dict(zip(cb_ind[0], 1 - cb_dist[0]))

    ## SVD candidates
    if product_id in svd_products:
        idx_sv = svd_products.index(product_id)
        sv_dist, sv_ind = knn_svd.kneighbors(
            product_factors[idx_sv].reshape(1, -1), n_neighbors=51
        )
        sv_scores = dict(zip(sv_ind[0], 1 - sv_dist[0]))
    else:
        sv_scores = {}

    ## combine candidates
    candidates = set(cb_ind[0].tolist()) | set(sv_scores.keys())
    candidates.discard(idx_cb)

    ## score each candidate
    records = []
    for i in candidates:
        pid        = product_features.iloc[i]['productid']
        same_cluster = int(product_features.iloc[i]['km_cluster'] == cluster_id)
        cb         = cb_scores.get(i, 0.0)
        sv         = sv_scores.get(i, 0.0)
        hybrid     = w_cluster * same_cluster + w_content * cb + w_svd * sv
        records.append({
            'productid':    pid,
            'avg_rating':   round(product_features.iloc[i]['avg_rating'], 2),
            'rating_count': int(product_features.iloc[i]['rating_count']),
            'km_cluster':   int(product_features.iloc[i]['km_cluster']),
            'hybrid_score': round(hybrid, 3),
        })

    result = pd.DataFrame(records).sort_values(
        'hybrid_score', ascending=False
    ).head(top_n).reset_index(drop=True)

    return result


# In[40]:


#Load data and build models
with st.spinner("Loading models..."):
    product_features, X_content_scaled, product_factors, svd_products = load_data()
    knn_content, knn_svd = build_models(X_content_scaled, product_factors)


# In[41]:


# UI
st.title("🛒 Product Recommendation System")
st.markdown("*Hybrid Recommender — Clustering · Content-Based · SVD*")
st.markdown("---")


# In[42]:


# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    selected_product = st.selectbox(
        "Select a Product",
        product_features['productid'].unique().tolist()
    )
    top_n = st.slider("Number of Recommendations", 3, 10, 5)
    recommend_btn = st.button("🚀 Recommend", use_container_width=True)


# In[43]:


# Selected product details
sel = product_features[product_features['productid'] == selected_product].iloc[0]
cluster_id    = int(sel['km_cluster'])
cluster_name  = CLUSTER_INFO[cluster_id]['name']
cluster_emoji = CLUSTER_INFO[cluster_id]['emoji']
cluster_color = CLUSTER_INFO[cluster_id]['color']

st.markdown("### 📦 Selected Product")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Product ID",   selected_product)
col2.metric("Avg Rating",   f"{sel['avg_rating']:.2f} / 5")
col3.metric("Reviews",      f"{int(sel['rating_count']):,}")
col4.metric("Cluster",      f"{cluster_emoji} {cluster_name}")

st.markdown("---")


# In[44]:


# Recommendations
if recommend_btn:
    with st.spinner("Finding recommendations..."):
        result = recommend_hybrid(
            selected_product, product_features,
            knn_content, X_content_scaled,
            knn_svd, product_factors, svd_products,
            top_n=top_n
        )

    if result is not None and not result.empty:
        st.markdown(f"### 🎯 Top {top_n} Recommendations for **{selected_product}**")

        for _, row in result.iterrows():
            c_name  = CLUSTER_INFO[row['km_cluster']]['name']
            c_emoji = CLUSTER_INFO[row['km_cluster']]['emoji']
            c_color = CLUSTER_INFO[row['km_cluster']]['color']
            stars   = "⭐" * int(round(row['avg_rating']))

            st.markdown(f"""
            <div style="background:#262730; border-left:5px solid {c_color};
                        border-radius:8px; padding:14px; margin:6px 0;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08)">
                <b>🔖 {row['productid']}</b>
                &nbsp;
                <span style="background:{c_color}; color:white; padding:2px 10px;
                             border-radius:12px; font-size:12px; font-weight:600">
                    {c_emoji} {c_name}
                </span>
                <br>
                <small>
                    {stars}&nbsp;
                    Avg Rating: <b>{row['avg_rating']}</b>&nbsp;|&nbsp;
                    Reviews: <b>{row['rating_count']:,}</b>&nbsp;|&nbsp;
                    Hybrid Score: <b>{row['hybrid_score']}</b>
                </small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this product.")
else:
    st.info("👈 Select a product and click Recommend to begin.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




