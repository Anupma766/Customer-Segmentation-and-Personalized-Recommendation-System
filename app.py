import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime



st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation and Based Recommendation")
st.subheader("✨ Your Personalized Product Recommendations")

# Sidebar - Upload Dataset
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

def load_data():
    return pd.read_csv("C:/Users/anupm/streamlit(app)/transactions_Cleaned.csv")

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset uploaded successfully!")
else:
    df = load_data()
    st.info("ℹ️ Using default dataset")

# Preprocess
st.subheader("📊 Raw Dataset")
st.dataframe(df.head())

# Clean data
df.dropna(subset=['customer_id', 'transaction_date'], inplace=True)
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# TotalPrice = list_price * quantity assumed; use 'list_price' as proxy if no quantity
if 'list_price' in df.columns:
    df['TotalPrice'] = df['list_price']  # replace with quantity * list_price if quantity available
else:
    st.error("🛑 'list_price' column is missing.")
    st.stop()

# RFM Calculation
snapshot_date = df['transaction_date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_id').agg({
    'transaction_date': lambda x: (snapshot_date - x.max()).days,
    'transaction_id': 'nunique',
    'TotalPrice': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm.astype({'Recency': int, 'Frequency': int, 'Monetary': float})

# Normalize for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Clustering
st.sidebar.header("🧪 Clustering Settings")
k_rfm = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=4)
kmeans = KMeans(n_clusters=k_rfm, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Assign Segment Labels (Top 4 scoring clusters)
cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_stats['Score'] = (
    -cluster_stats['Recency'].rank() +
    cluster_stats['Frequency'].rank() +
    cluster_stats['Monetary'].rank()
)
ordered_clusters = cluster_stats['Score'].sort_values(ascending=False).index.tolist()

labels_map = {
    ordered_clusters[0]: 'Loyal Customers',
    ordered_clusters[1]: 'Potential Loyalist',
    ordered_clusters[2]: 'New Customers',
    ordered_clusters[3]: 'At Risk'
}
for cluster in rfm['Cluster'].unique():
    if cluster not in labels_map:
        labels_map[cluster] = f"Cluster {cluster}"
rfm['Segment'] = rfm['Cluster'].map(labels_map)

# Show results
st.subheader("🧮 RFM Table with Cluster & Segment")
st.dataframe(rfm.reset_index().head())

# Interpretation
if st.checkbox("📄 View Segment Interpretations"):
    st.subheader("📌 Cluster Insights")
    summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    for _, row in summary.iterrows():
        st.markdown(f"**{row['Segment']}**: Recency = {row['Recency']:.0f}, Frequency = {row['Frequency']:.0f}, Monetary = ${row['Monetary']:.2f}")

# Segment Recommendations
st.subheader("🎁 Segment Recommendations")
rec_texts = {
    'Loyal Customers': "Premium Products, Exclusive Deals",
    'At Risk': "Win-back Campaigns, Discounts",
    'Potential Loyalist': "Loyalty Programs, Mid-range Offers",
    'New Customers': "Welcome Offers, Popular Items"
}
for seg, rec in rec_texts.items():
    st.markdown(f"**{seg}:** {rec}")

# Cluster Count
st.subheader("📊 Cluster Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=rfm, x='Segment', palette='Set2', order=rfm['Segment'].value_counts().index)
plt.title("Customer Count per Segment")
st.pyplot(fig)

# PCA Visualization
st.subheader("🧬 PCA Cluster Visualization")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
rfm['PCA1'], rfm['PCA2'] = pca_result[:, 0], pca_result[:, 1]

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Segment', palette='Set2', s=80)
plt.title("Customer Segments (PCA Projection)")
st.pyplot(fig2)

# Individual Customer Profile
st.subheader("🔍 Explore Customer Profile")
rfm = rfm.reset_index()
selected_cust = st.selectbox("Select Customer ID", rfm['customer_id'].astype(str), key="rfm_customer_select")
cust_info = rfm[rfm['customer_id'].astype(str) == selected_cust].iloc[0]
st.json({
    "Recency": int(cust_info['Recency']),
    "Frequency": int(cust_info['Frequency']),
    "Monetary": round(cust_info['Monetary'], 2),
    "Segment": cust_info['Segment']
})

# Download Button
st.download_button("📥 Download RFM Data", data=rfm.to_csv(index=False).encode('utf-8'), file_name="rfm_segmented.csv", mime="text/csv")

rfm[['customer_id', 'Segment']].to_csv("C:/Users/anupm/streamlit(app)/rfm_segmented.csv", index=False)

# PERSONALIZED RECOMMENDATION SYSTEM

# Set file paths directly (ensure the files are in the given paths)
uploaded_file = "C:/Users/anupm/streamlit(app)/transactions_Cleaned.csv"
rfm_path = "C:/Users/anupm/streamlit(app)/rfm_segmented.csv"

# Load the data
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["transaction_date"])
    st.success("✅ File uploaded successfully!")
else:
    df = pd.read_csv("C:/Users/anupm/streamlit(app)/transactions_Cleaned.csv", parse_dates=["transaction_date"])
    st.info("Using default dataset.")

st.subheader("📄 Raw Data Sample")
st.dataframe(df.head())

#preparing the data
df.dropna(subset=['customer_id', 'product_id'], inplace=True)

if 'Quantity' not in df.columns:
    df['Quantity'] = 1

if 'list_price' in df.columns:
    df['TotalPrice'] = df['Quantity'] * df['list_price']
else:
    st.error("Missing 'list_price' column")
    st.stop()

#Collaborative filtering (using SVD):
# Create customer-product matrix
interaction_matrix = df.pivot_table(index='customer_id', columns='product_id', values='Quantity', aggfunc='sum',
                                        fill_value=0)

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)
svd_matrix = svd.fit_transform(interaction_matrix)

#Content-Based Filtering (Product Similarity)
# Get product features and encode them
product_info = df[['product_id', 'brand', 'product_line', 'product_class', 'product_size']].drop_duplicates()
product_features = pd.get_dummies(product_info.drop('product_id', axis=1))

# Re-attach product_id
product_features['product_id'] = product_info['product_id'].values
product_features.set_index('product_id', inplace=True)

# Calculate similarity
content_sim_matrix = cosine_similarity(product_features)
content_sim_df = pd.DataFrame(content_sim_matrix, index=product_features.index, columns=product_features.index)

#Build Hybrid Recommender Function
def get_hybrid_recommendations(customer_id, num_recs=5):
    # --- COLLABORATIVE ---
    if customer_id not in interaction_matrix.index:
        return []

    cust_idx = interaction_matrix.index.get_loc(customer_id)
    cust_vector = svd_matrix[cust_idx]
    product_scores = np.dot(svd.components_.T, cust_vector)

    # Products not yet purchased
    purchased = set(interaction_matrix.loc[customer_id][interaction_matrix.loc[customer_id] > 0].index)
    product_ids = interaction_matrix.columns

    collab_recs = pd.Series(product_scores, index=product_ids)
    collab_recs = collab_recs[~collab_recs.index.isin(purchased)].nlargest(num_recs * 2)

    # --- CONTENT-BASED ---
    bought_products = df[df['customer_id'] == customer_id]['product_id'].unique()
    content_recs = pd.Series(dtype=float)

    for pid in bought_products:
        if pid in content_sim_df.columns:
            sims = content_sim_df[pid].drop(labels=bought_products, errors='ignore')
            if isinstance(sims, pd.Series):
                content_recs = content_recs.add(sims, fill_value=0)

    content_recs = content_recs.sort_values(ascending=False).head(num_recs * 2)

    # --- COMBINE ---
    combined = pd.concat([collab_recs, content_recs], axis=1)
    combined.columns = ['collab', 'content']
    combined = combined.fillna(0)

    # Weighted average: 60% collab, 40% content
    combined['score'] = 0.6 * combined['collab'] + 0.4 * combined['content']
    combined = combined.sort_values('score', ascending=False)

    return combined.head(num_recs).index.tolist()

#Show in Streamlit
# Show in Streamlit
# Select a customer
st.subheader("🔍 Get Recommendations")
customer_ids = interaction_matrix.index.tolist()
selected_customer = st.selectbox("Select Customer ID", customer_ids)

# Show customer type from RFM segmentation
cust_segment_info = rfm[rfm['customer_id'].astype(str) == str(selected_customer)]
if not cust_segment_info.empty:
    segment = cust_segment_info.iloc[0]['Segment']
    st.markdown(f"### 🧑‍💼 Customer Type: **{segment}**")
else:
    st.warning("This customer is not found in the RFM segmentation.")

# Now show recommendations
if selected_customer:
    recommendations = get_hybrid_recommendations(selected_customer, num_recs=5)
    st.markdown(f"### 🧠 Top Product Recommendations for Customer {selected_customer}")
    if recommendations:
        recommended_products = df[df['product_id'].isin(recommendations)][
            ['product_id', 'brand', 'product_line', 'product_class', 'list_price']].drop_duplicates()
        st.dataframe(recommended_products)
    else:
        st.info("No recommendations available for this customer.")

    # Optional: Show Purchase History
    if st.checkbox("🧾 Show Purchase History"):
        history = df[df['customer_id'] == selected_customer][['transaction_date', 'product_id', 'Quantity']]
        st.write(history.sort_values('transaction_date', ascending=False))
