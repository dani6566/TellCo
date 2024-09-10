import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from load_data import load_data_from_postgres



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

df = load_data()

print(df.head())

# Sidebar options
st.sidebar.header('Telecom User Engagement Dashboard')
selected_metric = st.sidebar.selectbox("Select Engagement Metric", ['Session Frequency', 'Session Duration', 'Total Traffic (DL+UL)'])

# Header
st.title('Telecom User Engagement and Experience Dashboard')

# Main KPIs
total_sessions = df['session_count'].sum()
avg_duration = df['session_duration'].mean()
total_traffic = df['total_traffic'].sum()

st.write("### Key Performance Indicators (KPIs)")
st.metric(label="Total Sessions", value=total_sessions)
st.metric(label="Average Session Duration (s)", value=round(avg_duration, 2))
st.metric(label="Total Traffic (DL + UL) in Bytes", value=total_traffic)

# Plot based on the selected engagement metric
if selected_metric == 'Session Frequency':
    top_customers_session = df.groupby('MSISDN')['session_count'].sum().nlargest(10).reset_index()
    fig = px.bar(top_customers_session, x='MSISDN', y='session_count', title="Top 10 Users by Session Frequency")
    st.plotly_chart(fig)

elif selected_metric == 'Session Duration':
    top_customers_duration = df.groupby('MSISDN')['session_duration'].sum().nlargest(10).reset_index()
    fig = px.bar(top_customers_duration, x='MSISDN', y='session_duration', title="Top 10 Users by Session Duration")
    st.plotly_chart(fig)

elif selected_metric == 'Total Traffic (DL+UL)':
    df['total_traffic'] = df['total_download'] + df['total_upload']
    top_customers_traffic = df.groupby('MSISDN')['total_traffic'].sum().nlargest(10).reset_index()
    fig = px.bar(top_customers_traffic, x='MSISDN', y='total_traffic', title="Top 10 Users by Total Traffic")
    st.plotly_chart(fig)

# Display application traffic
st.write("### Application Usage (Top 3 Most Used Applications)")
app_usage = df[['social_media_traffic', 'youtube_traffic', 'netflix_traffic', 'gaming_traffic']].sum().reset_index()
app_usage.columns = ['Application', 'Traffic']
fig_app_usage = px.pie(app_usage, names='Application', values='Traffic', title="Traffic Distribution by Application")
st.plotly_chart(fig_app_usage)

# Clustering - K-means clustering for user segmentation based on engagement metrics
st.write("### Customer Segmentation Using K-means Clustering")

# Normalize data and run K-means
engagement_metrics = df[['session_count', 'session_duration', 'total_traffic']]
scaler = StandardScaler()
engagement_metrics_scaled = scaler.fit_transform(engagement_metrics)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(engagement_metrics_scaled)

# Add cluster labels to the dataframe
df['cluster'] = clusters

# Visualize clusters
fig_cluster = px.scatter_3d(df, x='session_count', y='session_duration', z='total_traffic', color='cluster',
                            title="3D Scatter Plot of Customer Segments Based on Engagement Metrics")
st.plotly_chart(fig_cluster)

# Clustering details
cluster_summary = df.groupby('cluster')[['session_count', 'session_duration', 'total_traffic']].agg(['min', 'max', 'mean', 'sum'])
st.write("### Cluster Summary")
st.write(cluster_summary)

# Throughput per handset type
st.write("### Average Throughput per Handset Type")
throughput_per_handset = df.groupby('Handset Type')['throughput'].mean().reset_index()
fig_throughput = px.bar(throughput_per_handset, x='Handset Type', y='throughput', title="Average Throughput per Handset Type", height=600)
st.plotly_chart(fig_throughput)

# TCP Retransmission per handset type
st.write("### Average TCP Retransmission per Handset Type")
tcp_per_handset = df.groupby('Handset Type')['tcp_retransmission'].mean().reset_index()
fig_tcp = px.bar(tcp_per_handset, x='Handset Type', y='tcp_retransmission', title="Average TCP Retransmission per Handset Type", height=600)
st.plotly_chart(fig_tcp)

# Streamlit app footer
st.write("Dashboard created using Streamlit and Plotly for Telecom User Engagement and Experience Analysis.")
