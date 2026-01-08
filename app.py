import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Growth Intelligence", layout="wide")
st.title("🚀 Strategic Growth & RFM Dashboard")

# 2. Synthetic Data Engine
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = []
    # Simuliere 500 User für aussagekräftige RFM-Daten
    for i in range(1, 501):
        # Startdatum (Acquisition)
        start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="d")
        # Anzahl der Käufe (Frequency)
        num_purchases = np.random.randint(1, 12)
        for _ in range(num_purchases):
            # Kaufdaten (verteilt über Zeit)
            purchase_date = start_date + pd.to_timedelta(np.random.randint(0, 200), unit="d")
            data.append([f"User_{i:03d}", purchase_date, np.random.uniform(20, 250)])
    return pd.DataFrame(data, columns=["user_id", "purchase_date", "revenue"])

df = generate_data()

# Tabs für die Übersichtlichkeit
tab1, tab2 = st.tabs(["📉 Kohorten-Analyse", "👤 RFM Segmentierung"])

# --- TAB 1: KOHORTEN-ANALYSE ---
with tab1:
    st.header("Monthly Retention Analysis")
    
    # 1. SQL Logik für Kohorten
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
        SELECT CAST(cohort_month AS DATE) as cohort_month, month_number, COUNT(DISTINCT user_id) AS active_users
        FROM order_activities
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()

    # 2. FILTER: Monate auswählen (Sidebar)
    all_months = sorted(cohort_raw['cohort_month'].unique())
    selected_months = st.sidebar.multiselect(
        "Kohorten filtern (Startmonat):", 
        options=all_months, 
        default=all_months
    )

    # Daten filtern
    filtered_cohorts = cohort_raw[cohort_raw['cohort_month'].isin(selected_months)]

    if not filtered_cohorts.empty:
        # Pivot & Heatmap
        pivot_df = filtered_cohorts.pivot(index='cohort_month', columns='month_number', values='active_users')
        retention_matrix = pivot_df.divide(pivot_df.iloc[:, 0], axis=0) * 100

        fig = px.imshow(retention_matrix,
                        labels=dict(x="Monate nach Erstkauf", y="Kohorte", color="Retention %"),
                        x=retention_matrix.columns,
                        y=retention_matrix.index.astype(str),
                        color_continuous_scale='RdYlGn', text_auto='.1f', aspect="auto")
        
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bitte wählen Sie mindestens einen Monat in der Sidebar aus.")

# --- TAB 2: RFM ANALYSE ---
with tab2:
    st.header("Customer Segmentation (RFM)")
    st.markdown("Identifikation von Top-Kunden und Abwanderungsrisiken per SQL.")

    # RFM Berechnung per SQL
    # Heute wird simuliert als letztes Datum im Datensatz + 1 Tag
    rfm_data = duckdb.query("""
        SELECT 
            user_id,
            DATEDIFF('day', MAX(purchase_date), (SELECT MAX(purchase_date) FROM df) + INTERVAL 1 DAY) as recency,
            COUNT(DISTINCT purchase_date) as frequency,
            SUM(revenue) as monetary
        FROM df
        GROUP BY 1
    """).df()

    # Einfache Segmentierung (Quick & Dirty Director Logic)
    def segment_user(row):
        if row['frequency'] >= 7 and row['recency'] < 30: return 'Champions'
        if row['frequency'] >= 4: return 'Loyal Customers'
        if row['recency'] > 90: return 'At Risk / Churned'
        return 'Standard'

    rfm_data['Segment'] = rfm_data.apply(segment_user, axis=1)

    # Visualisierung der Segmente
    segment_counts = rfm_data['Segment'].value_counts().reset_index()
    fig_rfm = px.treemap(segment_counts, path=['Segment'], values='count', 
                         color='Segment', title="Verteilung der Kundensegmente")
    st.plotly_chart(fig_rfm, use_container_width=True)

    # Vorschau der Liste
    st.write("### Kunden-Segment-Liste")
    st.dataframe(rfm_data.sort_values('monetary', ascending=False), use_container_width=True)

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio")