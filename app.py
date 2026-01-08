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

# --- NEW: CHECKBOX SIDEBAR FILTER ---
st.sidebar.write("---")
st.sidebar.subheader("Filter Acquisition Cohorts")

df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# "Select All" Helper
select_all = st.sidebar.checkbox("Select All Months", value=True)

selected_months = []
for month in all_months:
    label = month.strftime('%B %Y')
    # If "Select All" is checked, the boxes are checked. Otherwise user decides.
    if st.sidebar.checkbox(label, value=select_all, key=f"cb_{label}"):
        selected_months.append(month)

filtered_users = df_users[df_users['cohort'].isin(selected_months)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.title("🚀 Product Growth Intelligence")
st.markdown(f"**Current Status:** {len(filtered_users)} Users in focus.")

tab1, tab2 = st.tabs(["📉 Usage Retention Matrix", "👤 Behavioral RFM Deep-Dive"])

# --- TAB 1: RETENTION MATRIX (CHRONOLOGICAL) ---
with tab1:
    st.subheader("Monthly Retention Matrix (%)")
    if not df_filtered_events.empty:
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
                WHERE e.type = 'visit'
            )
            SELECT 
                strftime(cohort_month, '%Y-%m') as sort_key,
                strftime(cohort_month, '%b %Y') as cohort_name, 
                month_number, 
                COUNT(DISTINCT user_id) AS active_users
            FROM activities WHERE month_number >= 0 GROUP BY 1, 2, 3 ORDER BY 1, 3
        """
        cohort_data = duckdb.query(cohort_query).df()
        pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
        
        # Enforce chronological sort
        pivot.index = pd.Categorical(pivot.index, categories=cohort_data.sort_values('sort_key')['cohort_name'].unique(), ordered=True)
        pivot = pivot.sort_index()
        
        retention = pivot.divide(pivot.iloc[:, 0], axis=0) * 100
        retention.columns = [f"Month {int(c)}" for c in retention.columns]

        fig_heat = px.imshow(retention, color_continuous_scale='Viridis', text_auto='.1f', aspect="auto", template=chart_template)
        fig_heat.update_layout(xaxis=dict(tickmode='linear', dtick=1, side='top'), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Please select at least one cohort.")

# --- TAB 2: RFM DEEP-DIVE ---
with tab2:
    st.subheader("Engagement vs. Monetary Value")
    rfm = duckdb.query("""
        SELECT u.user_id, strftime(u.acq_date, '%b %Y') as joined_month,
               COALESCE(date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00'), 9999) as r_hours,
               COALESCE(COUNT(CASE WHEN e.type = 'visit' THEN 1 END), 0) as f_visits,
               COALESCE(SUM(e.revenue), 0) as m_revenue
        FROM filtered_users u LEFT JOIN df_filtered_events e ON u.user_id = e.user_id GROUP BY 1, 2
    """).df()

    def segment_behavior(row):
        r = 'A. < 1h' if row['r_hours'] < 1 else 'B. 1-24h' if row['r_hours'] <= 24 else 'C. > 24h'
        f = 'A. Power User' if row['f_visits'] > 25 else 'B. Regular' if row['f_visits'] > 8 else 'C. Occasional'
        m = 'D. Not Converted' if row['m_revenue'] == 0 else 'C. Starter' if row['m_revenue'] <= 300 else 'B. Core' if row['m_revenue'] <= 1000 else 'A. High Value'
        return pd.Series([r, f, m])

    rfm[['Recency', 'Frequency', 'Monetary']] = rfm.apply(segment_behavior, axis=1)

    # COHORT A/B COMPARISON
    st.write("### Cohort A/B Comparison")
    available_comp = sorted(rfm['joined_month'].unique(), key=lambda x: datetime.strptime(x, '%b %Y'))
    if len(available_comp) >= 2:
        c1, c2 = st.columns(2)
        m_a = c1.selectbox("Cohort A:", available_comp, index=0, key="sa")
        m_b = c2.selectbox("Cohort B:", available_comp, index=min(1, len(available_comp)-1), key="sb")
        comp_df = rfm[rfm['joined_month'].isin([m_a, m_b])]
        fig_comp = px.histogram(comp_df, x="Monetary", color="joined_month", barmode="group", template=chart_template)
        st.plotly_chart(fig_comp, use_container_width=True)

    # CHARTS & TABLE
    st.write("---")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.bar(rfm['Recency'].value_counts().sort_index(), template=chart_template, title="Recency (Last Visit)"), use_container_width=True)
    col2_fig = px.bar(rfm['Frequency'].value_counts().sort_index(), template=chart_template, title="Frequency (Usage)")
    c2.plotly_chart(col2_fig, use_container_width=True)
    c3.plotly_chart(px.bar(rfm['Monetary'].value_counts().sort_index(), template=chart_template, title="Monetary (Revenue)"), use_container_width=True)

    st.subheader("Detailed Performance Data")
    st.dataframe(rfm[['user_id', 'joined_month', 'Recency', 'Frequency', 'Monetary', 'm_revenue']].sort_values('m_revenue', ascending=False), use_container_width=True)

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio | 2026")