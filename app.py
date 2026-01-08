import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")
st.title("📊 Strategic Growth & Retention Dashboard")

# 2. Synthetic Data Engine
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    data = []
    for i in range(1, 501): # Erhöht auf 500 User für bessere Sichtbarkeit
        start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="d")
        for _ in range(np.random.randint(1, 10)):
            purchase_date = start_date + pd.to_timedelta(np.random.randint(0, 210), unit="d")
            data.append([f"User_{i:03d}", purchase_date, np.random.uniform(20, 150)])
    return pd.DataFrame(data, columns=["user_id", "purchase_date", "revenue"])

df = generate_synthetic_data()

# 3. SQL Engine (Cohort Logic)
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
SELECT CAST(cohort_month AS DATE) as cohort_month, month_number, COUNT(DISTINCT user_id) AS active_users
FROM order_activities
GROUP BY 1, 2 ORDER BY 1, 2
"""
cohort_results = duckdb.query(sql_query).df()

# --- NEU: HEATMAP LOGIK ---
st.subheader("Retention Heatmap: Percentage of Active Users")

# Pivotieren der Daten für die Matrix
pivot_df = cohort_results.pivot(index='cohort_month', columns='month_number', values='active_users')

# Berechnung der Retention Rate in %
# Wir teilen jede Spalte durch die erste Spalte (Month 0)
retention_matrix = pivot_df.divide(pivot_df.iloc[:, 0], axis=0) * 100

# Plotly Heatmap erstellen
fig = px.imshow(retention_matrix,
                labels=dict(x="Months since First Purchase", y="Cohort Month", color="Retention %"),
                x=retention_matrix.columns,
                y=retention_matrix.index.astype(str),
                color_continuous_scale='Blues',
                text_auto='.1f', # Zeigt Prozentwerte in den Feldern
                aspect="auto")

st.plotly_chart(fig, use_container_width=True)

# 4. Ergänzende Linien-Grafik
st.subheader("Cohort Decay (Absolute Numbers)")
fig_line = px.line(cohort_results, x="month_number", y="active_users", color="cohort_month", template="plotly_white")
st.plotly_chart(fig_line, use_container_width=True)

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio")