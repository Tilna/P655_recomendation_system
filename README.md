Built a clustering-based hybrid recommendation system on 76K+ user-product interactions with 99.99% data sparsity across 40K+ unique products.

- Performed EDA to understand user and product behavior — identified extreme sparsity (99.99%) with average 1.02 ratings per user, which guided all algorithm decisions throughout the project.
-Applied and compared 3 clustering algorithms — KMeans, Hierarchical Clustering, and DBSCAN. Selected KMeans as the final model (silhouette score: 0.647) based on balanced clusters, scalability, and interpretability. Rejected DBSCAN despite higher silhouette score (0.888) due to one dominant cluster of 40K+ points.
-Engineered product-level features — avg_rating, rating_count, and log_rating_count (log scaling applied to handle outliers and normalize rating popularity for similarity calculations).
-Built Content-Based Filtering using Cosine Similarity and kNN on scaled product feature vectors — recommended products with the most similar rating profile to the selected product.
-Implemented SVD Collaborative Filtering using Matrix Factorization with 15 latent factors on a sparse user-item matrix built directly from interaction data to avoid memory overflow.
-Combined all three methods into a Hybrid Recommender with data-driven weights — Clustering (0.30), Content-Based (0.60), SVD (0.10) — weights assigned based on each method's actual reliability on this sparse dataset.
-Deployed as an interactive Streamlit web app with product selection, cluster identification, and hybrid recommendations displayed with cluster badges and scoring.

Tools: Python · Pandas · NumPy · Scikit-learn · SciPy · Streamlit

Skills: Recommendation Systems · KMeans Clustering · Content-Based Filtering · Collaborative Filtering · Matrix Factorization (SVD) · Feature Engineering · Streamlit · Python · Scikit-learn · SciPy
