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

if is_dark:
    chart_template = "plotly_dark"
    bg_color = "#0e1117"
    text_color = "#ffffff"
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
    st.markdown(f"""<style> .stApp {{ background-color: {bg_color}; color: {text_color}; }} </style>""", unsafe_allow_html=True)

# 2. DATA ENGINE
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
        if np.random.random() > 0.6:
            for _ in range(np.random.randint(1, 6)):
                p_date = acq + timedelta(days=np.random.randint(0, 180))
                if p_date < now: events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- SIDEBAR: SELECT ALL LOGIC (AC COMPLIANT) ---
st.sidebar.write("---")
st.sidebar.subheader("Filter Acquisition Cohorts")

# Initialisierung des Session States für alle Monat-Checkboxen
for month in all_months:
    cb_key = f"cb_{month.strftime('%Y-%m')}"
    if cb_key not in st.session_state:
        st.session_state[cb_key] = True

# Callback Funktion für den Master-Schalter
def toggle_all_months():
    new_status = st.session_state.master_select
    for m in all_months:
        st.session_state[f"cb_{m.strftime('%Y-%m')}"] = new_status

# Der Master-Schalter
st.sidebar.checkbox("Select All Months", value=True, key="master_select", on_change=toggle_all_months)

selected_months = []
for month in all_months:
    label = month.strftime('%B %Y')
    # Jede Checkbox ist an den Session State gebunden
    if st.sidebar.checkbox(label, key=f"cb_{month.strftime('%Y-%m')}"):
        selected_months.append(month)

# --- FILTERING ---
filtered_users = df_users[df_users['cohort'].isin(selected_months)].copy()
max_date_limit = max(selected_months) + pd.offsets.MonthEnd(1) if selected_months else datetime.now()
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.title("🚀 Product Growth Intelligence")

tab1, tab2 = st.tabs(["📉 Usage Retention Matrix", "👤 Behavioral RFM Deep-Dive"])

# --- TAB 1: RETENTION MATRIX ---
with tab1:
    st.subheader("Monthly Retention Matrix (%)")
    if not filtered_users.empty:
        cohort_query = f"""
            WITH user_cohorts AS (
                SELECT user_id, acq_date as cohort_date, DATE_TRUNC('month', acq_date) AS cohort_month FROM filtered_users
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
            fig_heat = px.imshow(retention, text_auto='.1f', template=chart_template, color_continuous_scale='Viridis')
            st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 2: RFM COMPARISON (AC COMPLIANT) ---
with tab2:
    st.subheader("MoM RFM Development Comparison")
    
    if len(selected_months) >= 2:
        # RFM Data Preparation
        rfm_raw = duckdb.query(f"""
            SELECT 
                u.user_id, 
                strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
                date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_days,
                COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_visits,
                SUM(e.revenue) as m_revenue
            FROM filtered_users u 
            LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
            WHERE e.event_date <= '{max_date_limit.strftime('%Y-%m-%d')}' OR e.event_date IS NULL
            GROUP BY 1, 2
        """).df()

        c1, c2 = st.columns(2)
        month_a = c1.selectbox("Select Cohort A:", options=sorted(rfm_raw['joined_month'].unique()), index=0)
        month_b = c2.selectbox("Select Cohort B:", options=sorted(rfm_raw['joined_month'].unique()), index=1)
        
        comp_df = rfm_raw[rfm_raw['joined_month'].isin([month_a, month_b])]

        # --- 1. MONEY DISTRIBUTION (AC) ---
        st.write("### 💰 Money Distribution (Revenue development)")
        fig_m = px.box(comp_df, x="joined_month", y="m_revenue", color="joined_month", 
                       title="Revenue Comparison per User (€)", template=chart_template, points="outliers")
        st.plotly_chart(fig_m, use_container_width=True)

        # --- 2. RECENCY DEVELOPMENT (AC) ---
        st.write("### 🕒 Recency Development (Days since last visit)")
        # Wir zeigen hier die Verteilung der Tage. Weniger Tage = Besser.
        fig_r = px.histogram(comp_df, x="r_days", color="joined_month", barmode="overlay", 
                             title="Recency Distribution (Lower is better)", template=chart_template)
        st.plotly_chart(fig_r, use_container_width=True)

        # --- 3. FREQUENCY CHANGE (AC) ---
        st.write("### 📈 Frequency Change (Number of visits)")
        # Durchschnittliche Frequenz pro Kohorte
        freq_stats = comp_df.groupby('joined_month')['f_visits'].mean().reset_index()
        fig_f = px.bar(freq_stats, x="joined_month", y="f_visits", color="joined_month",
                       title="Average Visit Frequency per User", template=chart_template)
        st.plotly_chart(fig_f, use_container_width=True)

        # Zusammenfassende Tabelle
        st.write("### Summary Metrics comparison")
        summary = comp_df.groupby('joined_month').agg({
            'm_revenue': 'sum',
            'f_visits': 'mean',
            'r_days': 'mean'
        }).rename(columns={'m_revenue': 'Total Revenue (€)', 'f_visits': 'Avg Frequency', 'r_days': 'Avg Recency (Days)'})
        st.table(summary)
    else:
        st.info("Please select at least two months in the sidebar to compare development.")

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio | 2026")