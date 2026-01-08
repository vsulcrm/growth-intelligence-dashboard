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

# --- 2. DATA ENGINE ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    now = datetime(2026, 1, 8, 12, 0)
    for i in range(1, 1200):
        uid = f"User_{i:04d}"
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 360))
        users.append([uid, acq])
        for _ in range(np.random.randint(1, 50)):
            v_date = acq + timedelta(days=np.random.randint(0, 300))
            if v_date < now: events.append([uid, v_date, "visit", 0])
        if np.random.random() > 0.4:
            for _ in range(np.random.randint(1, 10)):
                p_date = acq + timedelta(days=np.random.randint(0, 300))
                if p_date < now: events.append([uid, p_date, "purchase", np.random.uniform(5, 400)])
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- 3. HEADER & SIDEBAR LOGIK (Select All / Deselect All) ---
st.title("🚀 Product Growth Intelligence")

# View Selector
view_mode = st.radio("Modus wählen:", ["📉 Retention Analysis", "👤 RFM Score Comparison"], horizontal=True)

# Sidebar: Filter mit Session State Synchronisation
st.sidebar.subheader("Kohorten-Steuerung")

if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

# Logik für Alles an / Alles aus
def toggle_all():
    if st.session_state.master_toggle:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

st.sidebar.checkbox("Select All / Deselect All", value=len(st.session_state.selected_months) == len(all_months), 
                    key="master_toggle", on_change=toggle_all)

final_selected = []
for m in all_months:
    m_str = m.strftime('%B %Y')
    is_on = m in st.session_state.selected_months
    # Jede Checkbox wird an den State gebunden
    if st.sidebar.checkbox(m_str, value=is_on, key=f"cb_{m_str}"):
        final_selected.append(m)
st.session_state.selected_months = final_selected

if not final_selected:
    st.warning("Bitte Kohorten in der Sidebar auswählen.")
    st.stop()

filtered_users = df_users[df_users['cohort'].isin(final_selected)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.markdown("---")

# --- 4. BODY: RETENTION ---
if view_mode == "📉 Retention Analysis":
    st.header("Monthly Retention (Standard 12-Month View)")
    
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
        SELECT strftime(cohort_month, '%b %Y') as cohort_name, month_number, COUNT(DISTINCT user_id) AS active_users
        FROM activity WHERE month_number >= 0 AND month_number <= 12 GROUP BY 1, 2 ORDER BY 2
    """
    cohort_data = duckdb.query(retention_query).df()
    
    if not cohort_data.empty:
        pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users').fillna(0)
        retention_matrix = (pivot.divide(pivot.iloc[:, 0], axis=0) * 100).clip(upper=100.0)
        
        fig_heat = px.imshow(retention_matrix, text_auto='.1f', color_continuous_scale='RdYlGn', template=chart_template, aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

        # Milestone KPIs
        st.subheader("Milestone Metrics (M1, M3, M6, M12)")
        sel_c = st.selectbox("Kohorte wählen:", retention_matrix.index)
        kpi = retention_matrix.loc[sel_c]
        c1, c2, c3, c4 = st.columns(4)
        for i, m in enumerate([1, 3, 6, 12]):
            val = kpi[m] if m in kpi else 0
            [c1, c2, c3, c4][i].metric(f"Month {m}", f"{val:.1f}%")

# --- 4. BODY: RFM MODEL ---
else: 
    st.header("RFM Score Comparison (1-3 Scoring)")
    
    # 1. RFM Rohdaten berechnen
    rfm_raw = duckdb.query("""
        SELECT 
            u.user_id, 
            strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
            date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_raw,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_raw,
            COALESCE(SUM(e.revenue), 0) as m_raw
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        GROUP BY 1, 2
    """).df()

    # 2. Scoring Logik (1-3)
    # R: Niedrige Tage = Score 3, Hohe Tage = Score 1
    rfm_raw['R'] = pd.qcut(rfm_raw['r_raw'], 3, labels=["3", "2", "1"]).astype(str)
    # F: Hohe Anzahl = Score 3
    rfm_raw['F'] = pd.qcut(rfm_raw['f_raw'].rank(method='first'), 3, labels=["1", "2", "3"]).astype(str)
    # M: Hoher Umsatz = Score 3
    rfm_raw['M'] = pd.qcut(rfm_raw['m_raw'].rank(method='first'), 3, labels=["1", "2", "3"]).astype(str)
    
    # Gruppen kombinieren (z.B. "3-3-3")
    rfm_raw['RFM_Group'] = rfm_raw['R'] + "-" + rfm_raw['F'] + "-" + rfm_raw['M']

    # 3. Vergleichs-UI
    if len(final_selected) >= 2:
        opts = sorted(rfm_raw['joined_month'].unique())
        colA, colB = st.columns(2)
        mA = colA.selectbox("Vergleichsmonat A:", opts, index=0)
        mB = colB.selectbox("Vergleichsmonat B:", opts, index=1)
        
        comp_df = rfm_raw[rfm_raw['joined_month'].isin([mA, mB])]
        
        # Aggregation für den Gruppenvergleich
        group_counts = comp_df.groupby(['joined_month', 'RFM_Group']).size().reset_index(name='count')
        
        st.subheader("Vergleich der RFM-Gruppen (Verteilung)")
        fig_groups = px.bar(group_counts, x="RFM_Group", y="count", color="joined_month", 
                            barmode="group", title="Häufigkeit der RFM-Kombinationen (z.B. 3-3-3 = Champions)",
                            template=chart_template)
        st.plotly_chart(fig_groups, use_container_width=True)

        # Durchschnittswerte pro Score-Klasse
        st.subheader("Analyse der R-F-M Einzelscores")
        m_cols = st.columns(3)
        for i, score_type in enumerate(['R', 'F', 'M']):
            sub_stats = comp_df.groupby(['joined_month', score_type])['m_raw'].mean().reset_index()
            fig = px.bar(sub_stats, x=score_type, y="m_raw", color="joined_month", barmode="group",
                         title=f"Ø Umsatz nach {score_type}-Score", template=chart_template)
            m_cols[i].plotly_chart(fig, use_container_width=True)
    else:
        st.info("Bitte zwei Monate für den RFM-Vergleich wählen.")

# --- 5. FOOTER ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | AC Compliant RFM Scoring (1-3) | Volker Schulz")