import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")

# CSS für bessere Lesbarkeit der Metriken
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Strategic Growth & RFM Dashboard")
st.markdown("---")

# 2. SYNTHETIC DATA ENGINE (Director Level Simulation)
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = []
    # Erzeugung von 600 fiktiven Kunden für stabile statistische Segmente
    for i in range(1, 601):
        # Akquise-Zeitpunkt (Beitritt im Jahr 2024)
        start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="d")
        
        # Kauf-Häufigkeit (Frequency)
        num_purchases = np.random.randint(1, 15)
        for _ in range(num_purchases):
            # Käufe innerhalb der nächsten 200 Tage
            purchase_date = start_date + pd.to_timedelta(np.random.randint(0, 200), unit="d")
            data.append([f"User_{i:03d}", purchase_date, np.random.uniform(25, 300)])
            
    return pd.DataFrame(data, columns=["user_id", "purchase_date", "revenue"])

df = generate_data()

# TABS INITIALISIEREN
tab1, tab2 = st.tabs(["📉 Kohorten-Analyse (Retention)", "👤 RFM Segmentierung & Vergleich"])

# --- TAB 1: KOHORTEN-ANALYSE ---
with tab1:
    st.header("Monthly Retention Analysis")
    st.info("Diese Heatmap visualisiert die Nutzerbindung (Retention %) basierend auf dem Akquise-Monat.")
    
    # SQL LOGIK: Kohorten-Berechnung via DuckDB
    cohort_raw = duckdb.query("""
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
        SELECT 
            strftime(cohort_month, '%b %Y') as cohort_name,
            cohort_month, 
            month_number, 
            COUNT(DISTINCT user_id) AS active_users