import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

# 1. Page Configuration (Professional Branding)
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")
st.title("📊 Strategic Growth & Retention Dashboard")
st.sidebar.header("Dashboard Controls")
st.sidebar.info("This dashboard uses a high-performance SQL engine (DuckDB) to process cohorts in real-time.")

# 2. Synthetic Data Engine (Simulating E-Commerce/Subscription Data)
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    data = []
    # Simulating 300 unique users with multiple transactions
    for i in range(1, 301):
        # Assign a random "birth month" in 2024
        start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="d")
        for _ in range(np.random.randint(1, 10)):
            # Subsequent purchases within 180 days after first purchase
            purchase_date = start_date + pd.to_timedelta(np.random.randint(0, 180), unit="d")
            data.append([f"User_{i:03d}", purchase_date, np.random.uniform(20, 150)])
    
    df = pd.DataFrame(data, columns=["user_id", "purchase_date", "revenue"])
    return df

df = generate_synthetic_data()

# 3. SQL Engine (Executing your Cohort Logic)
# We use your SQL logic from Project C here
sql_query = """
WITH user_cohorts AS (
    SELECT user_id, MIN(DATE_TRUNC('month', purchase_date)) AS cohort_month
    FROM df GROUP BY 1
),
order_activities AS (
    SELECT t.user_id, uc.cohort_month,
           (EXTRACT(year FROM t.purchase_date) - EXTRACT(year FROM uc.cohort_month)) * 12 +
           (EXTRACT(month FROM t.purchase_date) - EXTRACT(month FROM uc.cohort_month)) AS month_number
    FROM df t JOIN user_cohorts uc ON t.user_id = uc.user_id
)
SELECT cohort_month, month_number, COUNT(DISTINCT user_id) AS active_users
FROM order_activities
GROUP BY 1, 2 ORDER BY 1, 2
"""
cohort_results = duckdb.query(sql_query).df()

# 4. Visualization Layer
st.subheader("Retention Analysis: Cohort Decay over Time")

# Pivot for the line chart
fig = px.line(cohort_results, 
              x="month_number", 
              y="active_users", 
              color="cohort_month",
              labels={"month_number": "Months since First Purchase", "active_users": "Active Users"},
              template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

# 5. Data Preview
with st.expander("View Raw Data Processed by SQL"):
    st.dataframe(cohort_results, use_container_width=True)

st.markdown("---")
st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio")