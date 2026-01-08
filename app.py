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
    st.markdown(f"<style>.stApp {{ background-color: #0e1117; color: #ffffff; }}</style>", unsafe_allow_html=True)
else:
    chart_template = "plotly_white"

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
        # Engagement
        for _ in range(np.random.randint(1, 40)):
            v_date = acq + timedelta(days=np.random.randint(0, 360))
            if v_date < now: events.append([uid, v_date, "visit", 0])
        # Revenue
        if np.random.random() > 0.5:
            for _ in range(np.random.randint(1, 6)):
                p_date = acq + timedelta(days=np.random.randint(0, 360))
                if p_date < now: events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- 3. HEADER BEREICH (Input & Auswahl) ---
st.title("🚀 Product Growth Intelligence")

# View Switcher
view_mode = st.radio("Ansicht wählen:", ["📉 Retention Analysis", "👤 RFM Comparison"], horizontal=True)

# SIDEBAR: Select All Logic
st.sidebar.subheader("Kohorten Auswahl")

if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

def toggle_all():
    if st.session_state.master_cb:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

st.sidebar.checkbox("Select All Months", value=len(st.session_state.selected_months) == len(all_months), 
                    key="master_cb", on_change=toggle_all)

current_selection = []
for m in all_months:
    m_str = m.strftime('%B %Y')
    is_on = m in st.session_state.selected_months
    if st.sidebar.checkbox(m_str, value=is_on, key=f"key_{m_str}"):
        current_selection.append(m)
st.session_state.selected_months = current_selection

if not current_selection:
    st.warning("Bitte wähle mindestens einen Monat aus.")
    st.stop()

# Filterung
filtered_users = df_users[df_users['cohort'].isin(current_selection)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.markdown("---")

# --- 4. BODY BEREICH (Output) ---

if view_mode == "📉 Retention Analysis":
    st.header("Monthly Retention (12-Month Horizon)")
    
    # Retention Logik: Monat 0 ist immer der Startmonat der Kohorte
    retention_query = """
        WITH user_cohorts AS (
            SELECT user_id, DATE_TRUNC('month', acq_date) AS cohort_month FROM filtered_users
        ),
        activity AS (
            SELECT e.user_id, uc.cohort_month,
                   (EXTRACT(year FROM e.event_date) - EXTRACT(year FROM uc.cohort_month)) * 12 +
                   (EXTRACT(month FROM e.event_date) - EXTRACT(month FROM uc.cohort_month)) AS month_number
            FROM df_filtered_events e
            JOIN user_cohorts uc ON e.user_id = uc.user_id
            WHERE e.type = 'visit'
        )
        SELECT 
            strftime(cohort_month, '%Y-%m') as sort_key,
            strftime(cohort_month, '%b %Y') as cohort_name, 
            month_number, 
            COUNT(DISTINCT user_id) AS active_users
        FROM activity
        WHERE month_number >= 0 AND month_number <= 12
        GROUP BY 1, 2, 3 ORDER BY 1, 3
    """
    cohort_data = duckdb.query(retention_query).df()
    
    if not cohort_data.empty:
        pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
        # Sortierung nach Datum sicherstellen
        pivot.index = pd.Categorical(pivot.index, categories=cohort_data.sort_values('sort_key')['cohort_name'].unique(), ordered=True)
        pivot = pivot.sort_index()
        
        # In Prozent umwandeln (Basis ist Monat 0)
        retention_matrix = pivot.divide(pivot.iloc[:, 0], axis=0) * 100
        
        fig_heat = px.imshow(retention_matrix, text_auto='.1f', color_continuous_scale='Viridis',
                             labels=dict(x="Monate seit Akquisition", y="Kohorte", color="Retention %"),
                             template=chart_template, aspect="auto")
        fig_heat.update_layout(xaxis=dict(tickmode='linear', dtick=1, side='top'))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Keine Daten für die gewählten Kohorten verfügbar.")

else: # RFM Comparison
    st.header("RFM Comparison: Kohorte A vs B")
    
    rfm_raw = duckdb.query("""
        SELECT 
            u.user_id, 
            strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
            date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_days,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_visits,
            COALESCE(SUM(e.revenue), 0) as m_revenue
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        GROUP BY 1, 2
    """).df()

    if len(current_selection) >= 2:
        c1, c2 = st.columns(2)
        opts = sorted(rfm_raw['joined_month'].unique())
        mA = c1.selectbox("Wähle Kohorte A", opts, index=0)
        mB = c2.selectbox("Wähle Kohorte B", opts, index=1)
        
        # Stats berechnen für Bar Charts
        compare_df = rfm_raw[rfm_raw['joined_month'].isin([mA, mB])].groupby('joined_month').agg({
            'm_revenue': 'mean',
            'f_visits': 'mean',
            'r_days': 'mean'
        }).reset_index()

        col_a, col_b, col_c = st.columns(3)
        col_a.plotly_chart(px.bar(compare_df, x='joined_month', y='m_revenue', color='joined_month', title="Avg Revenue (€)", template=chart_template), use_container_width=True)
        col_b.plotly_chart(px.bar(compare_df, x='joined_month', y='f_visits', color='joined_month', title="Avg Visits", template=chart_template), use_container_width=True)
        col_c.plotly_chart(px.bar(compare_df, x='joined_month', y='r_days', color='joined_month', title="Avg Recency (Days)", template=chart_template), use_container_width=True)
        
        st.table(compare_df.set_index('joined_month'))
    else:
        st.info("Bitte wähle mindestens zwei Monate für den Vergleich aus.")

# --- 5. FOOTER BEREICH ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | Synthetic Data View")