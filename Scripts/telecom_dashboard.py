import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from load_data import load_data_from_postgres
from utils import OverviewAnalysis



# Load the data
# @st.cache
def load_data():
    # Define your SQL query
    query = "SELECT * FROM xdr_data;"  # Replace with your actual table name

    # Load data from PostgreSQL
    df = load_data_from_postgres(query)

    # Display the first few rows of the dataframe
    if df is not None:
        print("Successfully loaded the data")
    else:
        print("Failed to load data.")
    return df




# Load dataset
df = load_data()

# Dashboard title
st.title('Telecom User Overview and Engagement Analysis Dashboard')

# Data Preprocessing
# Handle missing values by replacing with the mean
# df.fillna(df.mean(), inplace=True)
# Select only numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

# Replace missing values in non-numeric columns with the mode (most frequent value)
df[non_numeric_cols] = df[non_numeric_cols].fillna(df[non_numeric_cols].mode().iloc[0])

# Replace missing values in numeric columns with the mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(df[numeric_cols].head())
numeric_cols = df.select_dtypes(include=['float64', 'int64'])

Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Filter out rows with outliers (values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR)
df_filtered = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

# df.fillna(df.mean(), inplace=True)


# Aggregating metrics for engagement analysis
engagement_metrics = df.groupby('MSISDN/Number').agg(
    session_frequency=('Bearer Id', 'count'),
    session_duration=('Dur. (ms)', 'sum'),
    total_traffic=('Total DL (Bytes)', 'sum')
)

# User Overview: Top 10 Handsets
top_10_handsets = df['Handset Type'].value_counts().head(10)
st.subheader('Top 10 Handsets Used by Customers')
st.bar_chart(top_10_handsets)

# Engagement Analysis: Top 10 Customers by Session Count
top_10_sessions = engagement_metrics['session_frequency'].nlargest(10)
st.subheader('Top 10 Customers by Session Count')
st.bar_chart(top_10_sessions)

# Engagement Analysis: Top 10 Customers by Total Traffic
top_10_traffic = engagement_metrics['total_traffic'].nlargest(10)
st.subheader('Top 10 Customers by Total Traffic (Download + Upload)')
st.bar_chart(top_10_traffic)

# Network Performance: TCP retransmission, RTT, Throughput
network_performance = df.groupby('MSISDN/Number').agg(
    avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
    avg_rtt=('Avg RTT DL (ms)', 'mean'),
    avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
)

# Visualization of Average Throughput per Handset Type
st.subheader('Average Throughput per Handset Type')
handset_throughput = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False).head(10)
st.bar_chart(handset_throughput)

# Clustering users into engagement clusters using K-means (k=3)
scaler = StandardScaler()
engagement_metrics_scaled = scaler.fit_transform(engagement_metrics)

kmeans = KMeans(n_clusters=3, random_state=42)
engagement_metrics['cluster'] = kmeans.fit_predict(engagement_metrics_scaled)

# Visualizing clusters
st.subheader('User Engagement Clusters (K-Means Clustering)')
fig, ax = plt.subplots()
sns.scatterplot(
    x=engagement_metrics['session_duration'], y=engagement_metrics['total_traffic'],
    hue=engagement_metrics['cluster'], palette='viridis', ax=ax
)
ax.set_xlabel('Session Duration (s)')
ax.set_ylabel('Total Traffic (Bytes)')
st.pyplot(fig)

# Elbow Method for Optimal K
st.subheader('Elbow Method for Optimal K in Clustering')
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(engagement_metrics_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(K, inertia, 'bx-')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal K')
st.pyplot(fig)

# Network Performance: Clustering based on user experience
network_metrics_scaled = scaler.fit_transform(network_performance)
kmeans_network = KMeans(n_clusters=3, random_state=42)
network_performance['experience_cluster'] = kmeans_network.fit_predict(network_metrics_scaled)

# Visualize clusters based on TCP retransmission and Throughput
st.subheader('User Experience Clusters (K-Means Clustering)')
fig, ax = plt.subplots()
sns.scatterplot(
    x=network_performance['avg_tcp_retransmission'], y=network_performance['avg_throughput'],
    hue=network_performance['experience_cluster'], palette='coolwarm', ax=ax
)
ax.set_xlabel('Average TCP Retransmission (Bytes)')
ax.set_ylabel('Average Throughput (kbps)')
st.pyplot(fig)

# User Satisfaction: Analyzing satisfaction based on throughput
st.subheader('User Satisfaction: Throughput vs Handset Type')
handset_throughput_satisfaction = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
handset_throughput_satisfaction.plot(kind='bar', ax=ax)
ax.set_xlabel('Handset Type')
ax.set_ylabel('Average Throughput (kbps)')
ax.set_title('Average Throughput by Handset Type')
st.pyplot(fig)
