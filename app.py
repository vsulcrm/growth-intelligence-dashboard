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

# Dynamic CSS and Plotly Template based on Theme Selection
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
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE (Synthetic Engagement & Value)
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    # Current simulation time
    now = datetime(2026, 1, 8, 12, 0)
    
    for i in range(1, 1001):
        uid = f"User_{i:04d}"
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 360))
        users.append([uid, acq])
        
        # Engagement Events (Visits)
        for _ in range(np.random.randint(1, 40)):
            v_date = acq + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
            if v_date < now:
                events.append([uid, v_date, "visit", 0])
            
        # Value Events (Purchases)
        if np.random.random() > 0.5:
            for _ in range(np.random.randint(1, 6)):
                p_date = acq + timedelta(days=np.random.randint(0, 180))
                if p_date < now:
                    events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
                
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()

# --- GLOBAL FILTERS ---
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

selected_cohorts = st.sidebar.multiselect(
    "Select Acquisition Cohorts:", 
    options=all_months, 
    default=all_months,
    format_func=lambda x: x.strftime('%b %Y')
)

filtered_users = df_users[df_users['cohort'].isin(selected_cohorts)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.title("🚀 Product Growth Intelligence")
st.markdown(f"**Analyzing {len(filtered_users)} Acquisition Leads/Users**")

tab1, tab2 = st.tabs(["📉 Usage Retention", "👤 Engagement & RFM Deep-Dive"])

# --- TAB 1: RETENTION MATRIX ---
with tab1:
    st.subheader("Monthly Retention Matrix (%)")
    if not df_filtered_events.empty:
        # SQL Logic for Cohorts
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
        pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
        retention = pivot.divide(pivot.iloc[:, 0], axis=0) * 100
        
        fig_heat = px.imshow(retention, 
                             color_continuous_scale='Viridis', 
                             text_auto='.1f', 
                             aspect="auto", 
                             template=chart_template,
                             labels=dict(x="Months Since Start", y="Cohort", color="Retention %"))
        fig_heat.update_layout(xaxis=dict(tickmode='linear', dtick=1, side='top'))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Please filter cohorts in the sidebar.")

# --- TAB 2: RFM DEEP-DIVE ---
with tab2:
    st.subheader("Behavioral Analysis: Engagement vs. Value")
    
    rfm = duckdb.query("""
        SELECT u.user_id, strftime(u.acq_date, '%b %Y') as joined_month,
               date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_hours,
               COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_visits,
               COALESCE(SUM(e.revenue), 0) as m_revenue
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        GROUP BY 1, 2
    """).df()

    def segment_behavior(row):
        # Recency (Visit-based)
        r = 'A. < 1h' if row['r_hours'] < 1 else 'B. 1-24h' if row['r_hours'] <= 24 else 'C. > 24h'
        # Frequency (Usage-based)
        f = 'A. Power User' if row['f_visits'] > 25 else 'B. Regular' if row['f_visits'] > 8 else 'C. Occasional'
        # Monetary (Revenue-based)
        m = 'D. Not Converted' if row['m_revenue'] == 0 else 'C. Starter (< 30$)' if row['m_revenue'] <= 300 else 'B. Core' if row['m_revenue'] <= 1000 else 'A. High Value'
        return pd.Series([r, f, m])

    rfm[['Recency', 'Frequency', 'Monetary']] = rfm.apply(segment_behavior, axis=1)

    # 1. COHORT COMPARISON
    st.write("### Cohort A/B Comparison")
    available = sorted(rfm['joined_month'].unique(), key=lambda x: datetime.strptime(x, '%b %Y'))
    
    if len(available) >= 2:
        c1, c2 = st.columns(2)
        m_a = c1.selectbox("Cohort A:", options=available, index=0, key="sel_a")
        m_b = c2.selectbox("Cohort B:", options=available, index=min(1, len(available)-1), key="sel_b")
        
        comp_df = rfm[rfm['joined_month'].isin([m_a, m_b])]
        fig_comp = px.histogram(comp_df, x="Monetary", color="joined_month", barmode="group", 
                                template=chart_template, color_discrete_sequence=['#00CC96', '#636EFA'])
        st.plotly_chart(fig_comp, use_container_width=True)

    # 2. KEY METRICS & DISTRIBUTION
    st.write("---")
    st.write("### General Behavioral Distribution")
    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(px.bar(rfm['Recency'].value_counts().sort_index(), template=chart_template, title="Recency (Visits)"), use_container_width=True)
    col2.plotly_chart(px.bar(rfm['Frequency'].value_counts().sort_index(), template=chart_template, title="Frequency (Usage)"), use_container_width=True)
    col3.plotly_chart(px.bar(rfm['Monetary'].value_counts().sort_index(), template=chart_template, title="Monetary (Revenue)"), use_container_width=True)

    # 3. DETAILED TABLE
    st.subheader("Raw Performance Data")
    st.dataframe(rfm[['user_id', 'joined_month', 'Recency', 'Frequency', 'Monetary', 'm_revenue']].sort_values('m_revenue', ascending=False), use_container_width=True)

st.caption("Developed by Volker Schulz | Growth Intelligence Portfolio | 2026")