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
    events = [] # Besuche und Käufe
    now = datetime(2025, 1, 8, 12, 0)
    
    for i in range(1, 1001): # 1000 User
        user_id = f"User_{i:04d}"
        acq_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        users.append([user_id, acq_date])
        
        # Besuche simulieren (Engagement - Frequency & Recency)
        num_visits = np.random.randint(1, 50)
        for _ in range(num_visits):
            v_date = acq_date + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date > now: v_date = now - timedelta(minutes=np.random.randint(1, 60))
            events.append([user_id, v_date, "visit", 0])
            
        # Käufe simulieren (Monetary - nur für manche User)
        if np.random.random() > 0.6: # 40% Conversion Rate
            num_purchases = np.random.randint(1, 5)
            for _ in range(num_purchases):
                p_date = acq_date + timedelta(days=np.random.randint(0, 180))
                if p_date > now: p_date = now - timedelta(minutes=np.random.randint(1, 10))
                events.append([user_id, p_date, "purchase", np.random.uniform(10, 500)])
                
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()

# --- SIDEBAR FILTER ---
st.sidebar.header("Strategische Filter")
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
selected_months = st.sidebar.multiselect("Kohorten wählen:", options=sorted(df_users['cohort'].unique()), default=sorted(df_users['cohort'].unique()), format_func=lambda x: x.strftime('%B %Y'))

filtered_users = df_users[df_users['cohort'].isin(selected_months)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

tab1, tab2 = st.tabs(["📉 Retention (Nutzung)", "👤 RFM (Engagement vs. Value)"])

# --- TAB 2: RFM DEEP-DIVE ---
with tab2:
    st.header("RFM Analysis: Usage & Conversion")
    
    # SQL: RFM Metriken berechnen basierend auf DEINER Logik
    rfm_metrics = duckdb.query("""
        SELECT 
            u.user_id,
            -- Recency: Letzter Besuch
            date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2025-01-08 12:00:00') as hours_since_visit,
            -- Frequency: Anzahl der Besuche
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as visit_count,
            -- Monetary: Summe Umsatz
            COALESCE(SUM(e.revenue), 0) as total_revenue
        FROM filtered_users u
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id
        GROUP BY 1
    """).df()

    def apply_buckets(row):
        # Recency (Letzter Besuch)
        if row['hours_since_visit'] < 1: r = 'A. < 1h (Active Now)'
        elif row['hours_since_visit'] <= 10: r = 'B. 1-10h (Today)'
        else: r = 'C. > 10h (Inactive)'
        
        # Frequency (Besuche/Engagement)
        if row['visit_count'] > 30: f = 'A. Daily User'
        elif row['visit_count'] > 10: f = 'B. Weekly User'
        else: f = 'C. Monthly/Rare'
        
        # Monetary (Umsatz)
        if row['total_revenue'] == 0: m = 'D. Not Converted'
        elif row['total_revenue'] <= 30: m = 'C. Starter (< 30€)'
        elif row['total_revenue'] <= 300: m = 'B. Core Customer'
        else: m = 'A. High Value'
        
        return pd.Series([r, f, m])

    rfm_metrics[['Recency', 'Frequency', 'Monetary']] = rfm_metrics.apply(apply_buckets, axis=1)

    # VISUALISIERUNG
    st.subheader("Nutzer-Verhalten & Monetarisierung")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.info("**Recency (Besuch)**")
        st.bar_chart(rfm_metrics['Recency'].value_counts().sort_index())
    with c2:
        st.info("**Frequency (App-Opens)**")
        st.bar_chart(rfm_metrics['Frequency'].value_counts().sort_index())
    with c3:
        st.info("**Monetary (Umsatz)**")
        st.bar_chart(rfm_metrics['Monetary'].value_counts().sort_index())

    # KPI Sektion
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    converters = rfm_metrics[rfm_metrics['total_revenue'] > 0]
    k1.metric("Anzahl User (Leads)", len(rfm_metrics))
    k2.metric("Davon Käufer", len(converters))
    k3.metric("Conversion Rate", f"{(len(converters)/len(rfm_metrics)*100):.1f}%")

    st.subheader("Strategische Kundenliste")
    st.dataframe(rfm_metrics[['user_id', 'Recency', 'Frequency', 'Monetary', 'total_revenue']].sort_values('total_revenue', ascending=False), use_container_width=True)

st.caption("Growth Portfolio | Recency = Visit | Frequency = Usage | Monetary = Revenue")