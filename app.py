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
chart_template = "plotly_dark" if is_dark else "plotly_white"

# --- 2. DATA ENGINE (Realistische Simulation) ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    now = datetime(2026, 1, 8, 12, 0)
    for i in range(1, 1501):
        uid = f"User_{i:04d}"
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365))
        users.append([uid, acq])
        
        # Churn-Logik für Retention
        retention_prob = 0.8  
        for m in range(13):
            if np.random.random() < (retention_prob ** (m + 1)):
                v_date = acq + timedelta(days=m*30 + np.random.randint(0, 28))
                if v_date < now:
                    events.append([uid, v_date, "visit", 0])
                    if np.random.random() > 0.7:
                        events.append([uid, v_date, "purchase", np.random.uniform(20, 400)])
    
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- 3. HEADER BEREICH (Sidebar & Filter) ---
st.title("🚀 Product Growth Intelligence")

# View Selector
view_mode = st.radio("Analyseschwerpunkt:", ["📉 Retention Matrix", "👤 RFM Score Modell"], horizontal=True)

st.sidebar.subheader("Kohorten-Filter")

# Master-Checkbox Logik (Select/Deselect All)
if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

def toggle_all():
    if st.session_state.master_cb:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

st.sidebar.checkbox("Alle Monate an/abwählen", value=len(st.session_state.selected_months) == len(all_months), 
                    key="master_cb", on_change=toggle_all)

current_selection = []
for m in all_months:
    m_str = m.strftime('%Y-%m')
    m_label = m.strftime('%B %Y')
    is_on = m in st.session_state.selected_months
    if st.sidebar.checkbox(m_label, value=is_on, key=f"cb_{m_str}"):
        current_selection.append(m)
st.session_state.selected_months = current_selection

if not current_selection:
    st.warning("Bitte wählen Sie mindestens eine Kohorte aus.")
    st.stop()

filtered_users = df_users[df_users['cohort'].isin(current_selection)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.markdown("---")

# --- 4. BODY BEREICH ---

# --- TAB: RETENTION ---
if view_mode == "📉 Retention Matrix":
    st.header("Kohorten-Retention (12-Monats-Sicht)")
    
    ret_query = """
        WITH uc AS (SELECT user_id, DATE_TRUNC('month', acq_date) as c_month FROM filtered_users),
        act AS (
            SELECT e.user_id, uc.c_month,
            (EXTRACT(year FROM e.event_date) - EXTRACT(year FROM uc.c_month)) * 12 +
            (EXTRACT(month FROM e.event_date) - EXTRACT(month FROM uc.c_month)) as m_num
            FROM df_filtered_events e JOIN uc ON e.user_id = uc.user_id
            WHERE e.type = 'visit'
        )
        SELECT strftime(c_month, '%Y-%m') as sort_key, strftime(c_month, '%b %Y') as cohort, 
               m_num, COUNT(DISTINCT user_id) as active_users
        FROM act WHERE m_num BETWEEN 0 AND 12 GROUP BY 1, 2, 3 ORDER BY 1, 3
    """
    cohort_data = duckdb.query(ret_query).df()
    
    if not cohort_data.empty:
        pivot = cohort_data.pivot(index='cohort', columns='m_num', values='active_users').fillna(0)
        ordered_labels = cohort_