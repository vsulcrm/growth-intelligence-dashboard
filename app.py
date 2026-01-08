import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")

# 2. DATA ENGINE (Synthetic Data Generator)
@st.cache_data
def generate_data():
    np.random.seed(42)
    users = []
    events = []
    now = datetime(2026, 1, 8, 12, 0) # Current Time
    
    for i in range(1, 1001):
        user_id = f"User_{i:04d}"
        acq_date = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 360))
        users.append([user_id, acq_date])
        
        # Visits (Engagement)
        num_visits = np.random.randint(1, 50)
        for _ in range(num_visits):
            v_date = acq_date + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date < now:
                events.append([user_id, v_date, "visit", 0])
            
        # Purchases (Value)
        if np.random.random() > 0.5: # 50% conversion rate
            num_purchases = np.random.randint(1, 8)
            for _ in range(num_purchases):
                p_date = acq_date + timedelta(days=np.random.randint(0, 180))
                if p_date < now:
                    events.append([user_id, p_date, "purchase", np.random.uniform(10, 600)])
                
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()

# --- GLOBAL SIDEBAR FILTERS ---
st.sidebar.title("Global Filters")
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

selected_months = st.sidebar.multiselect(
    "Select Acquisition Cohorts:", 
    options=all_months, 
    default=all_months,
    format_func=lambda x: x.strftime('%B %Y')
)

# Filter Logic
filtered_users = df_users[df_users['cohort'].isin(selected_months)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

# MAIN UI
st.title("🚀 Growth & Product Intelligence")
st.markdown("---")

tab1, tab2 = st.tabs(["📉 Retention Analysis", "👤 Engagement & RFM Deep-Dive"])

# --- TAB 1: RETENTION ---
with tab1:
    st.header("Monthly User Retention")
    if not df_filtered_events.empty:
        cohort_query = """
            WITH user_cohorts AS (
                SELECT user_id, MIN(DATE_TRUNC('month', event_date)) AS cohort_month
                FROM df_filtered_events WHERE type = 'visit' GROUP BY 1
            ),
            activities AS (
                SELECT e.user_id, uc.cohort_month,
                       (EXTRACT(year FROM e.event_date) - EXTRACT(year FROM uc.cohort_month)) * 12 +
                       (EXTRACT(month FROM e.event_date) - EXTRACT(month FROM uc.cohort_month)) AS month_number
                FROM df_filtered_events e JOIN user_cohorts uc ON e.user_id = uc.user_id
            )
            SELECT strftime(cohort_month, '%b %Y') as cohort_name, cohort_month, month_number, COUNT(DISTINCT user_id) AS active_users
            FROM activities GROUP BY 1, 2, 3 ORDER BY 2, 3
        """
        cohort_data = duckdb.query(cohort_query).df()
        pivot_df = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
        retention_matrix = pivot_df.divide(pivot_df.iloc[:, 0], axis=0) * 100
        
        # Adaptive Heatmap
        fig_heat = px.imshow(retention_matrix, 
                             labels=dict(x="Months Since Acquisition", y="Cohort", color="Retention %"),
                             color_continuous_scale='RdYlGn', text_auto='.1f', aspect="auto")
        fig_heat.update_layout(xaxis=dict(tickmode='linear', dtick=1, side='top'))
        st.plotly_chart(fig_heat, use_container_width=True, theme="streamlit")
    else:
        st.warning("No data available for the selected cohorts.")

# --- TAB 2: RFM DEEP-DIVE ---
with tab2:
    st.header("Engagement vs. Monetary Value")
    
    # SQL: RFM Metrics
    rfm_metrics = duckdb.query("""
        SELECT 
            u.user_id,
            strftime(u.acq_date, '%b %Y') as joined_month,
            date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as hours_since_visit,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as visit_count,
            COALESCE(SUM(e.revenue), 0) as total_revenue
        FROM filtered_users u
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id
        GROUP BY 1, 2
    """).df()

    def apply_buckets(row):
        # Recency (Engagement)
        if row['hours_since_visit'] < 1: r = 'A. < 1h'
        elif row['hours_since_visit'] <= 10: r = 'B. 1-10h'
        else: r = 'C. > 10h'
        # Frequency (Usage)
        if row['visit_count'] > 25: f = 'A. Daily'
        elif row['visit_count'] > 8: f = 'B. Weekly'
        else: f = 'C. Rare'
        # Monetary (Value)
        if row['total_revenue'] == 0: m = 'D. Not Converted'
        elif row['total_revenue'] <= 50: m = 'C. Starter (< 50$)'
        elif row['total_revenue'] <= 400: m = 'B. Core Business'
        else: m = 'A. High Value'
        return