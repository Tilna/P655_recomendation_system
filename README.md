# 🛍️ Hybrid Product Recommendation System

A machine learning project that builds a hybrid recommendation system combining KMeans Clustering, Content-Based Filtering, and SVD Collaborative Filtering on 76K+ user-product interactions, deployed as an interactive Streamlit web app.

---

## 🔍 Project Overview

This project tackles the challenge of recommending products on a highly sparse dataset (99.99% sparsity) with 40K+ unique products and 76K+ user interactions. Three recommendation approaches were built, evaluated, and combined into a weighted hybrid system to maximize recommendation quality.

---

## 🚀 Features

- 📊 EDA on 76K+ user-product interactions with 99.99% data sparsity
- 🤖 Three recommendation methods built and compared
- 🏆 Hybrid recommender with data-driven weights
- 📉 Clustering evaluation using Silhouette Score
- 🌐 Interactive Streamlit web app with cluster badges and scoring
- 🔧 Feature engineering on product-level rating statistics

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Machine Learning | Scikit-learn, SciPy |
| Algorithms | KMeans, SVD, Cosine Similarity, kNN |
| Data Processing | Pandas, NumPy |
| Web App | Streamlit |

---

## 📁 Project Structure

```
P655_recomendation_system/
│
├── P655_Rec_sys_app.py         # Streamlit web application
├── X_content_scaled.npy        # Scaled product feature vectors
├── product_factors.npy         # SVD latent factors
├── product_features.xls        # Product features dataset
├── svd_products.pkl            # Trained SVD model
└── README.md
```

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tilna/P655_recomendation_system.git
cd P655_recomendation_system
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn scipy pandas numpy openpyxl
```

### 3. Run the app

```bash
streamlit run P655_Rec_sys_app.py
```

---

## 🧠 How It Works

### 1. Clustering (Weight: 0.30)
- Applied KMeans, Hierarchical Clustering, and DBSCAN
- Selected KMeans (Silhouette Score: 0.647) for balanced clusters and scalability
- Rejected DBSCAN despite higher score (0.888) due to one dominant cluster of 40K+ points

### 2. Content-Based Filtering (Weight: 0.60)
- Engineered product features: avg_rating, rating_count, log_rating_count
- Used Cosine Similarity and kNN on scaled feature vectors
- Recommends products with similar rating profiles

### 3. SVD Collaborative Filtering (Weight: 0.10)
- Matrix Factorization with 15 latent factors
- Built on sparse user-item matrix to avoid memory overflow
- Captures hidden patterns in user-product interactions

### 4. Hybrid Recommender
- Combined all three methods with data-driven weights
- Weights assigned based on each method's reliability on this sparse dataset

---

## 📊 Key Insights from EDA

- Dataset sparsity: **99.99%**
- Average ratings per user: **1.02**
- Unique products: **40K+**
- These findings guided all algorithm decisions in the project

---

## 📌 How the App Works

1. User selects a product from the dropdown
2. App identifies the product's cluster
3. Hybrid recommendations are displayed with cluster badges and scores

---

## 🙋 Author

**Tilna**
- GitHub: [@Tilna](https://github.com/Tilna)

---

