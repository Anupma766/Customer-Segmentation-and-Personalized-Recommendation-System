Overview:
This project implements a Customer Segmentation and Personalized Product Recommendation system using a combination of RFM-based segmentation and a Hybrid recommendation system. The system clusters customers based on their behavior (Recency, Frequency, and Monetary value) and provides personalized product recommendations through a hybrid approach combining Collaborative Filtering (using Singular Value Decomposition) and Content-Based Filtering.



Key Features:
Customer Segmentation: Uses Recency, Frequency, and Monetary (RFM) analysis to classify customers into segments such as Loyal Customers, Potential Loyalists, New Customers, and At Risk.

Hybrid Recommendation System: Combines Collaborative Filtering using SVD and Content-Based Filtering to provide accurate product recommendations.

Personalized Recommendations: Displays personalized product recommendations based on the customer's previous behavior and the customer segment.

Interactive Dashboard: Built using Streamlit to allow users to upload their own transaction datasets and explore customer segments and product recommendations.


Technologies Used:
Python: Main programming language used for implementing the system.

Pandas: For data manipulation and analysis.

Scikit-learn: Used for clustering (KMeans), normalization (StandardScaler), and Singular Value Decomposition (SVD).

Streamlit: For building the interactive web app.

Matplotlib and Seaborn: For visualizing the customer segments and recommendations.

Numpy: For numerical operations.


How It Works:
Data Preprocessing:
The dataset consists of customer transactions (customer_id, product_id, transaction_date, list_price, quantity, etc.).
Missing values are handled, and additional features like TotalPrice are created from existing columns.

RFM Segmentation:
Customers are clustered using Recency, Frequency, and Monetary (RFM) metrics, which provide insights into customer behavior.
K-Means clustering is used to segment customers into distinct categories.

Hybrid Recommendation System:
Collaborative Filtering: Applies Singular Value Decomposition (SVD) to create a customer-product interaction matrix and recommend products based on similar users.
Content-Based Filtering: Recommends products based on the similarity of product features (such as brand, product line, class, and size).

Hybrid Approach: 
Combines both methods by giving a weighted score to both approaches to generate final product recommendations.

Streamlit App:
Allows users to upload their own datasets and get segmentation insights.
Displays personalized product recommendations based on customer segment and behavior.
