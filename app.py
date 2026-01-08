import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Growth Intelligence", layout="wide")
st.title("📊 Engagement-Driven RFM Dashboard")
st.markdown("---")

# 2. DATA ENGINE (Simuliert Besuche vs. Käufe)
@st.cache_data
def generate_data():
    np.random.seed(42)
    users = []
    events = []
    now = datetime(2025, 1, 8, 12, 0)
    
    for i in range(1, 1001):
        user_id = f"User_{i:04d}"
        acq_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        users.append([user_id, acq_date])
        
        # Besuche (Engagement)
        num_visits = np.random.randint(1, 40)
        for _ in range(num_visits):
            v_date = acq_date + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date > now: v_date = now - timedelta(minutes=np.random.randint(1, 60))
            events.append([user_id, v_date, "visit", 0])
            
        # Käufe (Monetary)
        if np.random.random() > 0.6:
            num_purchases = np.random.randint(1, 6)
            for _ in range(num_purchases):
                p_date = acq_date + timedelta(days=np.random.randint(0, 180))
                if p_date > now: p_date = now - timedelta(minutes=np.random.randint(1, 10))
                events.append([user_id, p_date, "purchase", np.random.uniform(10, 500)])
                
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()

# --- GLOBALER SIDEBAR FILTER ---
st.sidebar.header("Strategische Filter")
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

selected_months = st.sidebar.multiselect(
    "Akquise-Monate auswählen:", 
    options=all_months, 
    default=all_months,
    format_func=lambda x: x.strftime('%B %Y')
)

filtered_users = df_users[df_users['cohort'].isin(selected_months)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

tab1, tab2 = st.tabs(["📉 Retention (Nutzung)", "👤 RFM Deep-Dive"])

# --- TAB 1: RETENTION ---
with tab1:
    st.header("Monthly Retention Analysis")
    if not df_filtered_events.empty:
        # SQL Logik für Kohorten
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
        fig = px.imshow(retention_matrix, color_continuous_scale='RdYlGn', text_auto='.1f', aspect="auto")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# --- TAB 2: RFM DEEP-DIVE ---
with tab2:
    st.header("RFM Analysis: Usage & Conversion")
    
    # RFM Metriken berechnen
    rfm_metrics = duckdb.query("""
        SELECT 
            u.user_id,
            strftime(u.acq_date, '%b %Y') as joined_month,
            u.acq_date as joined_date,
            date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2025-01-08 12:00:00') as hours_since_visit,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as visit_count,
            COALESCE(SUM(e.revenue), 0) as total_revenue
        FROM filtered_users u
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id
        GROUP BY 1, 2, 3
    """).df()

    def apply_buckets(row):
        if row['hours_since_visit'] < 1: r = 'A. < 1h'
        elif row['hours_since_visit'] <= 10: r = 'B. 1-10h'
        else: r = 'C. > 10h'
        
        if row['visit_count'] > 25: f = 'A. Daily'
        elif row['visit_count'] > 8: f = 'B. Weekly'
        else: f = 'C. Monthly'
        
        if row['total_revenue'] == 0: m = 'D. Not Converted'
        elif row['total_revenue'] <= 30: m = 'C. Starter (< 30€)'
        elif row['total_revenue'] <= 300: m = 'B. Core'
        else: m = 'A. High Value'
        return pd.Series([r, f, m])

    rfm_metrics[['Recency', 'Frequency', 'Monetary']] = rfm_metrics.apply(apply_buckets, axis=1)

    # --- DER VERMISSTE VERGLEICHS-FILTER ---
    st.subheader("📊 Kohorten-Vergleich (A/B Test)")
    
    # Verfügbare Monate aus den gefilterten Daten
    month_options = sorted(rfm_metrics['joined_month'].unique(), 
                           key=lambda x: datetime.strptime(x, '%b %Y'))

    if len(month_options) >= 2:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            month_a = st.selectbox("Monat A (Basis):", options=month_options, index=0)
        with col_sel2:
            month_b = st.selectbox("Monat B (Vergleich):", options=month_options, index=1)

        # Daten für Vergleich filtern
        comp_df = rfm_metrics[rfm_metrics['joined_month'].isin([month_a, month_b])]
        comp_viz = comp_df.groupby(['joined_month', 'Monetary']).size().reset_index(name='Anzahl')

        fig_compare = px.bar(comp_viz, x='Monetary', y='Anzahl', color='joined_month',
                             barmode='group', template="plotly_dark",
                             title=f"Umsatz-Verteilung: {month_a} vs. {month_b}")
        st.plotly_chart(fig_compare, use_container_width=True, theme="streamlit")
    else:
        st.info("Wähle mindestens zwei Monate in der Sidebar aus, um sie zu vergleichen.")

    # Roh-Übersicht
    st.markdown("---")
    st.subheader("Engagement & Value Verteilung (Gesamt)")
    c1, c2, c3 = st.columns(3)
    c1.bar_chart(rfm_metrics['Recency'].value_counts())
    c2.bar_chart(rfm_metrics['Frequency'].value_counts())
    c3.bar_chart(rfm_metrics['Monetary'].value_counts())

st.caption("Growth Intelligence Dashboard | Volker Schulz")