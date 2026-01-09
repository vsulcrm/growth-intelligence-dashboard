# Growth Intelligence Dashboard
Link: https://growth-intelligence-dashboard-4hzzl8ygntxbpigxgmoipy.streamlit.app

A streamlined Streamlit application for analyzing usage retention, calculating Lifetime Value (LTV), and performing RFM (Recency, Frequency, Monetary) segmentation using DuckDB and Plotly.

## Overview

This dashboard allows growth teams to visualization and predict user behavior:
1.  **Retention Matrix**: View cohort retention over time (classic triangular view).
2.  **LTV Prediction**: Analyze Cumulative ARPU per cohort with Month-0 to Month-12 projections using Linear Regression.
3.  **RFM Analysis**: Segment users based on Recency, Frequency, and Monetary value, with Month-Over-Month migration flows.

## Features

### 1. Retention & Cohort Analysis
-   **User Cohorts**: Group users by acquisition month.
-   **Retention Heatmap**: Track active user percentage over time.
-   **Masking**: Future months are automatically masked as "N/A".

### 2. LTV Prediction (Lifetime Value)
-   **Matrix View**: A table displaying Cumulative ARPU for Months 0-12.
    -   **Actuals**: Historical data (black text).
    -   **Predictions**: Future values (blue italic text), projected using linear regression.
-   **Formulas**: View the regression model (`y = mx + c`) for each cohort.
-   **Fallback**: Cohorts with insufficient data use an Average Slope fallback.

### 3. RFM Segmentation
-   **RFM Scores**:
    -   **Recency (R)**: Tertiles (3=Recent, 1=Old).
    -   **Frequency (F)**: Custom Bins (0, 1=Low, 2=Med, 3=High).
    -   **Monetary (M)**: Spending Tiers (0, 1=Small, 2=Med, 3=High).
-   **MoM Flow**: Analyze user migration between months (e.g., Jan -> Feb).
    -   **Strict Migration**: comparisons track specific users from the base month to see their status in the next month (excluding new users).
-   **Visuals**:
    -   **Delta Heatmap**: Net change in user counts per RFM segment.
    -   **Composition Bar Chart**: Distribution of segments.

## Installation & Usage

1.  **Prerequisites**: Python 3.8+.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires: streamlit, pandas, plotly, duckdb, numpy)*

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Data Source
The app currently generates realistic **Synthetic Data** (Users & Events) on startup for demonstration purposes.

## License
Internal Tool.
