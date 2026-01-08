import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")

# CSS für bessere Lesbarkeit
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Strategic Growth & RFM Dashboard")
st.markdown("---")

# 2. SYNTHETIC DATA ENGINE
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = []
    for i in range(1, 601):
        start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="d")
        num_purchases = np.random.randint(1, 15)
        for _ in range(num_purchases):
            purchase_date = start_date + pd.to_timedelta(np.random.randint(0, 200), unit="d")
            data.append([f"User_{i:03d}", purchase_date, np.random.uniform(25, 300)])
    return pd.DataFrame(data, columns=["user_id", "purchase_date", "revenue"])

df = generate_data()

# TABS INITIALISIEREN
tab1, tab2 = st.tabs(["📉 Kohorten-Analyse (Retention)", "👤 RFM Segmentierung & Vergleich"])

# --- TAB 1: KOHORTEN-ANALYSE ---
with tab1:
    st.header("Monthly Retention Analysis")
    
    # SQL LOGIK: Achte darauf, dass der Block mit """) endet
    query_cohort = """
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
        FROM order_activities
        GROUP BY 1, 2, 3 ORDER BY 2, 3
    """
    cohort_raw = duckdb.query(query_cohort).df()

    # FILTER IN DER SIDEBAR
    all_months = sorted(cohort_raw['cohort_month'].unique())
    selected_months_raw = st.sidebar.multiselect(
        "Start-Kohorten filtern:", 
        options=all_months, 
        default=all_months,
        format_func=lambda x: x.strftime('%B %Y')
    )

    filtered_cohorts = cohort_raw[cohort_raw['cohort_month'].isin(selected_months_raw)]

    if not filtered_cohorts.empty:
        pivot_df = filtered_cohorts.pivot(index='cohort_name', columns='month_number', values='active_users')
        pivot_df = pivot_df.reindex(filtered_cohorts['cohort_name'].unique()) 
        retention_matrix = pivot_df.divide(pivot_df.iloc[:, 0], axis=0) * 100

        fig = px.imshow(retention_matrix,
                        labels=dict(x="Monate nach Erstkauf", y="Start-Kohorte", color="Retention %"),
                        color_continuous_scale='RdYlGn', text_auto='.1f', aspect="auto", template="plotly_dark")
        
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1, side='top'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bitte Kohorten wählen.")

# --- TAB 2: RFM ANALYSE ---
with tab2:
    st.header("Deep-Dive: Kohorten-Vergleich im RFM-Modell")

    query_rfm = """
        WITH user_first_mon AS (
            SELECT user_id, strftime(MIN(DATE_TRUNC('month', purchase_date)), '%b %Y') as joined_month,
                   MIN(DATE_TRUNC('month', purchase_date)) as joined_month_date
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
        SELECT r.*, m.joined_month, m.joined_month_date 
        FROM rfm_metrics r JOIN user_first_mon m ON r.user_id = m.user_id
    """
    rfm_table = duckdb.query(query_rfm).df()

    def segment_user(row):
        if row['frequency'] >= 8 and row['recency'] < 45: return 'Champions'
        if row['frequency'] >= 4: return 'Loyal Customers'
        if row['recency'] > 120: return 'At Risk'
        return 'Standard'
    
    rfm_table['Segment'] = rfm_table.apply(segment_user, axis=1)

    st.markdown("---")
    col_sel1, col_sel2 = st.columns(2)
    
    # Korrekte Sortierung der Monate für die Auswahl
    available_months_sorted = sorted(rfm_table['joined_month_date'].unique())
    month_options = [pd.to_datetime(x).strftime('%b %Y') for x in available_months_sorted]

    with col_sel1:
        month_a = st.selectbox("Basis-Monat (A):", options=month_options, index=0)
    with col_sel2:
        month_b = st.selectbox("Vergleichs-Monat (B):", options=month_options, index=min(1, len(month_options)-1))

    comparison_df = rfm_table[rfm_table['joined_month'].isin([month_a, month_b])]
    comp_viz = comparison_df.groupby(['joined_month', 'Segment']).size().reset_index(name='Anzahl')

    fig_compare = px.bar(comp_viz, x='Segment', y='Anzahl', color='joined_month',
                         barmode='group', template="plotly_dark",
                         title=f"Segment-Verteilung: {month_a} vs. {month_b}",
                         color_discrete_sequence=['#636EFA', '#EF553B'])
    st.plotly_chart(fig_compare, use_container_width=True)

    # METRIKEN
    data_a = rfm_table[rfm_table['joined_month'] == month_a]
    data_b = rfm_table[rfm_table['joined_month'] == month_b]
    m1, m2, m3 = st.columns(3)
    
    ch_a = len(data_a[data_a['Segment'] == 'Champions'])
    ch_b = len(data_b[data_b['Segment'] == 'Champions'])
    m1.metric("Champions", ch_b, delta=int(ch_b - ch_a))

    rev_a = data_a['monetary'].mean() if not data_a.empty else 0
    rev_b = data_b['monetary'].mean() if not data_b.empty else 0
    m2.metric("Ø Umsatz", f"{rev_b:.2f}€", delta=f"{(rev_b - rev_a):.2f}€")
    m3.metric("Kunden", len(data_b), delta=int(len(data_b) - len(data_a)))

st.markdown("---")
st.caption("Growth Intelligence Dashboard | Case Study | Volker Schulz")