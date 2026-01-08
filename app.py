import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Growth Intelligence", layout="wide")
st.title("🚀 Strategic Growth & RFM Dashboard")

# 2. Synthetic Data Engine
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = []
    for i in range(1, 601): # 600 User für bessere Statistik
        start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="d")
        num_purchases = np.random.randint(1, 15)
        for _ in range(num_purchases):
            purchase_date = start_date + pd.to_timedelta(np.random.randint(0, 200), unit="d")
            data.append([f"User_{i:03d}", purchase_date, np.random.uniform(25, 300)])
    return pd.DataFrame(data, columns=["user_id", "purchase_date", "revenue"])

df = generate_data()

# Tabs
tab1, tab2 = st.tabs(["📉 Kohorten-Analyse (Retention)", "👤 RFM Segmentierung & Vergleich"])

# --- TAB 1: KOHORTEN-ANALYSE ---
with tab1:
    st.header("Retention Heatmap")
    
    # SQL mit sauberer Datumsformatierung für die Lesbarkeit
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
            strftime(cohort_month, '%b %Y') as cohort_name, -- Kurzform: "Jan 2024"
            cohort_month, 
            month_number, 
            COUNT(DISTINCT user_id) AS active_users
        FROM order_activities
        GROUP BY 1, 2, 3 ORDER BY 2, 3
    """).df()

    # Filter in der Sidebar
    all_months = sorted(cohort_raw['cohort_month'].unique())
    selected_months_raw = st.sidebar.multiselect(
        "Kohorten auswählen:", options=all_months, default=all_months,
        format_func=lambda x: x.strftime('%B %Y')
    )

    filtered_cohorts = cohort_raw[cohort_raw['cohort_month'].isin(selected_months_raw)]

    if not filtered_cohorts.empty:
        pivot_df = filtered_cohorts.pivot(index='cohort_name', columns='month_number', values='active_users')
        # Sortierung sicherstellen (Jan -> Feb -> Mar)
        pivot_df = pivot_df.reindex(filtered_cohorts['cohort_name'].unique())
        retention_matrix = pivot_df.divide(pivot_df.iloc[:, 0], axis=0) * 100

        fig = px.imshow(retention_matrix,
                        labels=dict(x="Monate seit Erstkauf", y="Start-Kohorte", color="Retention %"),
                        color_continuous_scale='RdYlGn', text_auto='.1f', aspect="auto")
        
        # FIX: Lesbarkeit der Achsen
        fig.update_layout(
            xaxis=dict(tickmode='linear', dtick=1, side='top', tickfont=dict(size=12, color='white')),
            yaxis=dict(tickfont=dict(size=12, color='white')),
            margin=dict(l=100, r=20, t=100, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bitte Kohorten in der Sidebar wählen.")

# --- TAB 2: RFM ANALYSE & GEGENÜBERSTELLUNG ---
with tab2:
    st.header("RFM Performance & Cohort Comparison")

    # RFM Kern-Daten per SQL
    rfm_table = duckdb.query("""
        WITH user_first_mon AS (
            SELECT user_id, strftime(MIN(DATE_TRUNC('month', purchase_date)), '%b %Y') as joined_month
            FROM df GROUP BY 1
        ),
        rfm_metrics AS (
            SELECT 
                user_id,
                DATEDIFF('day', MAX(purchase_date), (SELECT MAX(purchase_date) FROM df) + INTERVAL 1 DAY) as recency,
                COUNT(DISTINCT purchase_date) as frequency,
                SUM(revenue) as monetary
            FROM df GROUP BY 1
        )
        SELECT r.*, m.joined_month 
        FROM rfm_metrics r JOIN user_first_mon m ON r.user_id = m.user_id
    """).df()

    def segment_user(row):
        if row['frequency'] >= 8 and row['recency'] < 40: return 'Champions'
        if row['frequency'] >= 4: return 'Loyal'
        if row['recency'] > 100: return 'At Risk'
        return 'Standard'

    rfm_table['Segment'] = rfm_table.apply(segment_user, axis=1)

    # GEGENÜBERSTELLUNG: Segmente pro Monat
    st.subheader("Welche Kohorten bringen die besten Kundensegmente?")
    
    # Aggregation für den Vergleich
    comparison = rfm_table.groupby(['joined_month', 'Segment']).size().reset_index(name='count')
    
    fig_comp = px.bar(comparison, x='joined_month', y='count', color='Segment',
                      title="Kunden-Zusammensetzung nach Akquise-Monat",
                      barmode='stack', template="plotly_dark")
    
    fig_comp.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title="Akquise Monat", yaxis_title="Anzahl Kunden")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Details
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Top Segmente (Anzahl)")
        st.bar_chart(rfm_table['Segment'].value_counts())
    with col2:
        st.write("### Umsatz pro Segment")
        monetary_seg = rfm_table.groupby('Segment')['monetary'].sum()
        st.bar_chart(monetary_seg)

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio")