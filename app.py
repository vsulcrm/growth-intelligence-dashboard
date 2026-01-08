import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Growth Intelligence", layout="wide")

st.sidebar.title("App Settings")
is_dark = st.sidebar.toggle("Dark Mode Optimized", value=True)

if is_dark:
    chart_template = "plotly_dark"
    bg_color = "#0e1117"; text_color = "#ffffff"
    st.markdown(f"<style>.stApp {{ background-color: {bg_color}; color: {text_color}; }}</style>", unsafe_allow_html=True)
else:
    chart_template = "plotly_white"
    bg_color = "#ffffff"; text_color = "#000000"

# --- 2. DATA ENGINE ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    now = datetime(2026, 1, 8, 12, 0)
    for i in range(1, 1001):
        uid = f"User_{i:04d}"
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 360))
        users.append([uid, acq])
        for _ in range(np.random.randint(1, 40)):
            v_date = acq + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date < now: events.append([uid, v_date, "visit", 0])
        if np.random.random() > 0.5:
            for _ in range(np.random.randint(1, 8)):
                p_date = acq + timedelta(days=np.random.randint(0, 180))
                if p_date < now: events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- 3. HEADER (Selection Logic) ---
st.title("🚀 Product Growth Intelligence")

# Section: View Selection
view_mode = st.radio("Select Dashboard View:", ["📉 Retention Analysis", "👤 RFM Comparison"], horizontal=True)

# Section: Month Selection (Sidebar with AC Logic)
st.sidebar.subheader("Filter Cohorts")

if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

def toggle_all():
    if st.session_state.master_cb:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

st.sidebar.checkbox("Select All Months", value=True, key="master_cb", on_change=toggle_all)

final_selected = []
for m in all_months:
    m_str = m.strftime('%B %Y')
    is_on = m in st.session_state.selected_months
    if st.sidebar.checkbox(m_str, value=is_on, key=f"key_{m_str}"):
        final_selected.append(m)
st.session_state.selected_months = final_selected

# Filtering Logic
filtered_users = df_users[df_users['cohort'].isin(final_selected)]
if not final_selected:
    st.warning("Please select at least one month.")
    st.stop()
    
# Calculate max date to prevent January leak
max_date_limit = max(final_selected) + pd.offsets.MonthEnd(1)
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.markdown("---")

# --- 4. BODY ---
if view_mode == "📉 Retention Analysis":
    st.header("Monthly User Retention")
    
    cohort_query = f"""
        WITH user_cohorts AS (
            SELECT user_id, acq_date, DATE_TRUNC('month', acq_date) AS cohort_month FROM filtered_users
        ),
        activities AS (
            SELECT e.user_id, uc.cohort_month,
                   (EXTRACT(year FROM e.event_date) - EXTRACT(year FROM uc.cohort_month)) * 12 +
                   (EXTRACT(month FROM e.event_date) - EXTRACT(month FROM uc.cohort_month)) AS month_number
            FROM df_filtered_events e 
            JOIN user_cohorts uc ON e.user_id = uc.user_id
            WHERE e.type = 'visit' AND e.event_date <= '{max_date_limit.strftime('%Y-%m-%d')}'
        )
        SELECT strftime(cohort_month, '%b %Y') as cohort_name, month_number, COUNT(DISTINCT user_id) AS active_users
        FROM activities WHERE month_number >= 0 GROUP BY 1, 2 ORDER BY 1, 2
    """
    cohort_data = duckdb.query(cohort_query).df()
    
    if not cohort_data.empty:
        pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
        retention = pivot.divide(pivot.iloc[:, 0], axis=0) * 100
        retention.columns = [f"M{int(c)}" for c in retention.columns]
        
        fig_heat = px.imshow(retention, text_auto='.1f', template=chart_template, color_continuous_scale='Viridis', aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No retention data for selected filters.")

else: # RFM Comparison
    st.header("Behavioral RFM Comparison")
    
    rfm_raw = duckdb.query(f"""
        SELECT 
            u.user_id, 
            strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
            date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_days,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_visits,
            COALESCE(SUM(e.revenue), 0) as m_revenue
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        WHERE e.event_date <= '{max_date_limit.strftime('%Y-%m-%d')}' OR e.event_date IS NULL
        GROUP BY 1, 2
    """).df()

    if len(final_selected) >= 2:
        col1, col2 = st.columns(2)
        options = sorted(rfm_raw['joined_month'].unique())
        m_a = col1.selectbox("Cohort A (Baseline):", options, index=0)
        m_b = col2.selectbox("Cohort B (Comparison):", options, index=1)
        
        comp_df = rfm_raw[rfm_raw['joined_month'].isin([m_a, m_b])]

        # AC: Money Distribution (Revenue development)
        st.subheader("💰 Monetary: Revenue Development")
        fig_m = px.box(comp_df, x="joined_month", y="m_revenue", color="joined_month", 
                       points="all", title="Revenue per User Comparison", template=chart_template)
        st.plotly_chart(fig_m, use_container_width=True)

        # AC: Recency (How it has developed)
        st.subheader("🕒 Recency: Engagement Freshness")
        fig_r = px.histogram(comp_df, x="r_days", color="joined_month", barmode="group",
                             nbins=20, title="Days since last visit (Lower is better)", template=chart_template)
        st.plotly_chart(fig_r, use_container_width=True)

        # AC: Frequency (How usage has changed)
        st.subheader("📈 Frequency: Usage Intensity")
        fig_f = px.violin(comp_df, x="joined_month", y="f_visits", color="joined_month", 
                          box=True, title="Visit Frequency Distribution", template=chart_template)
        st.plotly_chart(fig_f, use_container_width=True)
        
    else:
        st.info("Please select at least 2 months in the sidebar for comparison.")

# --- 5. FOOTER ---
st.markdown("---")
st.caption(f"Growth Intelligence Dashboard | Data Status: Jan 2026 | User: Volker Schulz")