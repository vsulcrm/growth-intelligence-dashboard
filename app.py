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
        # Simulation of engagement (Visits)
        for _ in range(np.random.randint(1, 40)):
            v_date = acq + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date < now: events.append([uid, v_date, "visit", 0])
        # Simulation of revenue (Purchases)
        if np.random.random() > 0.6:
            for _ in range(np.random.randint(1, 6)):
                p_date = acq + timedelta(days=np.random.randint(0, 180))
                if p_date < now: events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()

# --- SIDEBAR CHECKBOX FILTER (FIXED LOGIC) ---
st.sidebar.write("---")
st.sidebar.subheader("Filter Acquisition Cohorts")
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# Initialisierung Session State
if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

# Callback für Select All
def on_select_all_change():
    if st.session_state.select_all_cb:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

st.sidebar.checkbox("Select All Months", 
                    value=len(st.session_state.selected_months) == len(all_months), 
                    key="select_all_cb", 
                    on_change=on_select_all_change)

# Einzelne Monate
current_selected = []
for month in all_months:
    label = month.strftime('%B %Y')
    is_sel = month in st.session_state.selected_months
    if st.sidebar.checkbox(label, value=is_sel, key=f"month_{label}"):
        current_selected.append(month)

st.session_state.selected_months = current_selected

# --- FILTERING LOGIC ---
filtered_users = df_users[df_users['cohort'].isin(st.session_state.selected_months)].copy()

# Fix für den Januar-Fehler: Max-Datum berechnen
if not st.session_state.selected_months:
    max_date_limit = datetime(2026, 1, 8)
else:
    max_date_limit = max(st.session_state.selected_months) + pd.offsets.MonthEnd(1)

df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.title("🚀 Product Growth Intelligence")
st.markdown(f"**Current Status:** {len(filtered_users)} Users in Focus")

tab1, tab2 = st.tabs(["📉 Usage Retention Matrix", "👤 Behavioral RFM Deep-Dive"])

# --- TAB 1: RETENTION MATRIX ---
with tab1:
    st.subheader("Monthly Retention Matrix (%)")
    if not filtered_users.empty and not df_filtered_events.empty:
        cohort_query = f"""
            WITH user_cohorts AS (
                SELECT user_id, acq_date as cohort_date, DATE_TRUNC('month', acq_date) AS cohort_month
                FROM filtered_users
            ),
            activities AS (
                SELECT e.user_id, uc.cohort_month,
                       (EXTRACT(year FROM e.event_date) - EXTRACT(year FROM uc.cohort_month)) * 12 +
                       (EXTRACT(month FROM e.event_date) - EXTRACT(month FROM uc.cohort_month)) AS month_number
                FROM df_filtered_events e 
                JOIN user_cohorts uc ON e.user_id = uc.user_id
                WHERE e.type = 'visit'
                AND e.event_date <= '{max_date_limit.strftime('%Y-%m-%d')}'
            )
            SELECT 
                strftime(cohort_month, '%Y-%m') as sort_key,
                strftime(cohort_month, '%b %Y') as cohort_name, 
                month_number, 
                COUNT(DISTINCT user_id) AS active_users
            FROM activities 
            WHERE month_number >= 0 
            GROUP BY 1, 2, 3 
            ORDER BY 1, 3
        """
        cohort_data = duckdb.query(cohort_query).df()
        
        if not cohort_data.empty:
            pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
            pivot.index = pd.Categorical(pivot.index, categories=cohort_data.sort_values('sort_key')['cohort_name'].unique(), ordered=True)
            pivot = pivot.sort_index()
            
            retention = pivot.divide(pivot.iloc[:, 0], axis=0) * 100
            retention.columns = [f"Month {int(c)}" for c in retention.columns]

            fig_heat = px.imshow(retention, color_continuous_scale='Viridis', text_auto='.1f', aspect="auto", template=chart_template)
            fig_heat.update_layout(xaxis=dict(tickmode='linear', dtick=1, side='top'), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Please select at least one cohort in the sidebar.")

# --- TAB 2: RFM DEEP-DIVE ---
with tab2:
    st.subheader("Engagement vs. Monetary Value")
    
    if not filtered_users.empty:
        rfm = duckdb.query(f"""
            SELECT 
                u.user_id, 
                strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
                strftime(DATE_TRUNC('month', u.acq_date), '%Y-%m') as sort_key,
                COALESCE(date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00'), 9999) as r_hours,
                COALESCE(COUNT(CASE WHEN e.type = 'visit' THEN 1 END), 0) as f_visits,
                COALESCE(SUM(e.revenue), 0) as m_revenue
            FROM filtered_users u 
            LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
            WHERE e.event_date <= '{max_date_limit.strftime('%Y-%m-%d')}' OR e.event_date IS NULL
            GROUP BY 1, 2, 3
        """).df()

        def segment_behavior(row):
            r = 'A. < 1h' if row['r_hours'] < 1 else 'B. 1-24h' if row['r_hours'] <= 24 else 'C. > 24h'
            f = 'A. Power User (>25)' if row['f_visits'] > 25 else 'B. Regular (8-25)' if row['f_visits'] > 8 else 'C. Occasional'
            m = 'D. €0' if row['m_revenue'] == 0 else 'C. Starter (<€300)' if row['m_revenue'] <= 300 else 'B. Core (<€1k)' if row['m_revenue'] <= 1000 else 'A. High Value'
            return pd.Series([r, f, m])

        rfm[['Recency', 'Frequency', 'Monetary']] = rfm.apply(segment_behavior, axis=1)

        # A/B Vergleich
        st.write("### Cohort Revenue Comparison")
        available_comp = sorted(rfm['joined_month'].unique())
        
        if len(available_comp) >= 2:
            c1, c2 = st.columns(2)
            m_a = c1.selectbox("Cohort A:", options=available_comp, index=0)
            m_b = c2.selectbox("Cohort B:", options=available_comp, index=1)
            
            comp_df = rfm[rfm['joined_month'].isin([m_a, m_b])]
            fig_comp = px.box(comp_df, x="joined_month", y="m_revenue", color="joined_month", 
                             points="all", title="Monetary Distribution (€)", template=chart_template)
            st.plotly_chart(fig_comp, use_container_width=True)

        # DISTRIBUTION CHARTS
        st.write("---")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(px.bar(rfm['Recency'].value_counts().sort_index(), title="Users by Recency"), use_container_width=True)
        c2.plotly_chart(px.bar(rfm['Frequency'].value_counts().sort_index(), title="Users by Usage Frequency"), use_container_width=True)
        
        # Monetary Chart Fix: Summe statt Count
        m_dist = rfm.groupby('Monetary')['m_revenue'].sum().reset_index()
        c3.plotly_chart(px.bar(m_dist, x='Monetary', y='m_revenue', title="Total Revenue by Segment (€)", color='Monetary'), use_container_width=True)

        st.subheader("Detailed Performance Data")
        st.dataframe(rfm[['user_id', 'joined_month', 'Recency', 'Frequency', 'Monetary', 'm_revenue']].sort_values('m_revenue', ascending=False), use_container_width=True)

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio | 2026")