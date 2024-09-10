# Telecom User Analysis

This project provides detailed analysis and insights into user behavior, engagement, and experience using telecom data. The analysis focuses on three main aspects: **User Overview**, **User Engagement**, and **User Experience**. The results are visualized through an interactive dashboard built using Streamlit.

## Table of Contents

- [Project Overview](#project-overview)
- [Analysis Breakdown](#analysis-breakdown)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [License](#license)

## Project Overview

The goal of this project is to analyze opportunities for growth in the telecom industry by assessing various metrics related to user behavior, network performance, and device usage. By examining these aspects, we can provide recommendations to improve customer satisfaction, enhance network performance, and target specific customer segments for marketing efforts.

The analysis is broken down into three key sections:

- **User Overview Analysis**: Insight into the types of handsets customers use and their distribution.
- **User Engagement Analysis**: Tracking user activity based on session frequency, duration, and data usage (download/upload).
- **User Experience Analysis**: Analyzing how network parameters such as TCP retransmission, Round Trip Time (RTT), and throughput impact user experience.

## Analysis Breakdown

### 1. User Overview Analysis

The **User Overview** focuses on understanding the types of handsets used by the customers and the manufacturers. This section of the analysis provides insights into:

- **Top 10 Handsets**: The most popular devices used by customers.
- **Top 3 Manufacturers**: The leading handset manufacturers in the network.
- **Top 5 Handsets per Manufacturer**: Device-level details for each of the top manufacturers.

### 2. User Engagement Analysis

The **User Engagement Analysis** evaluates how customers interact with the telecom network by examining:

- **Session Frequency**: How often users access the network.
- **Session Duration**: The average length of user sessions.
- **Data Usage**: Total upload and download volume per session.
- **Top 10 Customers by Engagement**: Identify the most engaged users based on their session count, duration, and data usage.

Additionally, this analysis includes clustering users based on their engagement metrics using K-means clustering to group them into three distinct clusters.

### 3. User Experience Analysis

The **User Experience Analysis** focuses on the performance of the telecom network and how it affects customer experience. Metrics analyzed include:

- **TCP Retransmission**: The rate of data retransmissions due to errors.
- **Round Trip Time (RTT)**: The time taken for data to travel to the server and back.
- **Throughput**: The amount of data successfully delivered over the network.
  
The analysis also clusters users into groups based on these metrics, providing insights into which users experience the best or worst network performance.

## Data Requirements

To perform the analysis, the dataset must include the following columns:

- `Handset Type`: The type of device used by the customer.
- `Handset Manufacturer`: The manufacturer of the device.
- `MSISDN/Number`: The unique identifier for each customer.
- `Dur. (s)`: Session duration in seconds.
- `Total DL (Bytes)`: Total download traffic in bytes.
- `Total UL (Bytes)`: Total upload traffic in bytes.
- `TCP DL Retrans. Vol (Bytes)`: TCP download retransmission volume.
- `TCP UL Retrans. Vol (Bytes)`: TCP upload retransmission volume.
- `Avg RTT DL (ms)`: Average Round Trip Time for download.
- `Avg RTT UL (ms)`: Average Round Trip Time for upload.
- `Avg Bearer TP DL (kbps)`: Average download throughput in kilobits per second.
- `Avg Bearer TP UL (kbps)`: Average upload throughput in kilobits per second.

## Installation dependencies<br>
       pip install -r requirements.txt
### Dependencies include:

streamlit<br>
pandas<br>
matplotlib<br>
scikit-learn<br>
seaborn<br>
