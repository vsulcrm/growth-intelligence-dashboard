import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# 1. PAGE SETUP
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")

# --- THEME SELECTOR ---
st.sidebar.title("App Settings")
is_dark = st.sidebar.toggle("Enable Dark Mode Optimization", value=True)

# Dynamic CSS and Plotly Template based on Theme
if is_dark:
    chart_template = "plotly_dark"
    bg_color = "#0e1117"
    text_color = "#ffffff"
    # Ensure visibility of tables and metrics in dark mode
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        [data-testid="stMetricValue"] {{ color: #00d4ff !important; }}
        thead tr th {{ background-color: #262730 !important; color: white !important; }}
        </style>
    """, unsafe_allow_html=True)
else:
    chart_template = "plotly_white"
    bg_color = "#ffffff"
    text_color = "#000000"
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE (Synthetic Engagement & Value)
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    # Setting current time to Jan 2026
    now = datetime(2026, 1, 8, 12, 0)
    
    for i in range(1, 1001):
        uid = f"User_{i:04d}"
        # Users joined throughout 2025
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 360))
        users.append([uid, acq])
        
        # Visits (Engagement - Frequency & Recency)
        for _ in range(np.random.randint(1, 40)):
            v_date = acq + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date < now:
                events.append([uid, v_date, "visit", 0])
            
        # Purchases (Monetary Value)
        if np.random.random() > 0.5: # 50% Conversion
            for _ in range(np.random.randint(1, 6)):
                p_date = acq + timedelta(days=np.random.randint(0, 180))
                if p_date < now:
                    events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
                
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()

# --- GLOBAL FILTERS ---
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

selected_cohorts = st.sidebar.multiselect(
    "Select Acquisition Cohorts:", 
    options=all_months, 
    default=all_months,
    format_func=lambda x: x.strftime('%b %Y')
)

filtered_users = df_users[df_users['cohort'].isin(selected_cohorts)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.title("🚀 Product Growth Intelligence")
st.markdown(f"**Current Status:** {len(filtered_users)} Leads/Users analyzed.")

tab1, tab2 = st.tabs(["📉 Usage Retention", "👤 Engagement & RFM Deep-Dive"])

# --- TAB 1: RETENTION MATRIX ---
with tab1:
    st.subheader("Monthly Retention Matrix (%)")
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
                FROM df_filtered_events e JOIN user_cohorts uc