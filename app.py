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
if is_dark:
    st.markdown("<style>.stApp { background-color: #0e1117; color: #ffffff; }</style>", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    now = datetime(2026, 1, 8, 12, 0)
    for i in range(1, 1001):
        uid = f"User_{i:04d}"
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 360))
        users.append([uid, acq])
        # Engagement (Visits)
        for _ in range(np.random.randint(1, 40)):
            v_date = acq + timedelta(days=np.random.randint(0, 360))
            if v_date < now: events.append([uid, v_date, "visit", 0])
        # Revenue (Purchases)
        if np.random.random() > 0.5:
            for _ in range(np.random.randint(1, 6)):
                p_date = acq + timedelta(days=np.random.randint(0, 360))
                if p_date < now: events.append([uid, p_date, "purchase", np.random.uniform(10, 500)])
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- 3. HEADER & SIDEBAR (STRICT SELECTION LOGIC) ---
st.title("🚀 Product Growth Intelligence")

# View Selector
view_mode = st.radio("Ansicht:", ["📉 Retention Analysis", "👤 RFM Model Comparison"], horizontal=True)

# SIDEBAR FILTER LOGIC
st.sidebar.subheader("Kohorten-Filter")

# State-Synchronisation für Checkboxen
if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

def sync_checkboxes():
    if st.session_state.master_cb:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

# Master Toggle
st.sidebar.checkbox("Select All Months", value=len(st.session_state.selected_months) == len(all_months), 
                    key="master_cb", on_change=sync_checkboxes)

# Einzelne Checkboxen
final_selected = []
for m in all_months:
    m_str = m.strftime('%B %Y')
    # Wir setzen den Wert direkt aus dem session_state
    is_on = m in st.session_state.selected_months
    if st.sidebar.checkbox(m_str, value=is_on, key=f"cb_{m_str}"):
        final_selected.append(m)
st.session_state.selected_months = final_selected

if not final_selected:
    st.warning("Bitte wähle mindestens einen Monat aus.")
    st.stop()

# Filterung der Daten
filtered_users = df_users[df_users['cohort'].isin(final_selected)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.markdown("---")

# --- 4. BODY BEREICH ---

if view_mode == "📉 Retention Analysis":
    st.header("Monthly Retention (Cohort View)")
    
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
        SELECT 
            strftime(cohort_month, '%Y-%m') as sort_key,
            strftime(cohort_month, '%b %Y') as cohort_name, 
            month_number, 
            COUNT(DISTINCT user_id) AS active_users
        FROM activity
        WHERE month_number >= 0 AND month_number <= 12
        GROUP BY 1, 2, 3 ORDER BY 1, 3
    """
    cohort_data = duckdb.query(retention_query).df()
    
    if not cohort_data.empty:
        pivot = cohort_data.pivot(index='cohort_name', columns='month_number', values='active_users')
        pivot.index = pd.Categorical(pivot.index, categories=cohort_data.sort_values('sort_key')['cohort_name'].unique(), ordered=True)
        pivot = pivot.sort_index().fillna(0)
        
        # Normierung: Monat 0 ist 100%, Folgemonate können nicht > 100% sein
        retention_matrix = pivot.divide(pivot.iloc[:, 0], axis=0) * 100
        retention_matrix = retention_matrix.clip(upper=100.0) # Cap bei 100%
        
        fig_heat = px.imshow(retention_matrix, text_auto='.1f', color_continuous_scale='RdYlGn',
                             labels=dict(x="Monate seit Start", y="Kohorte", color="Retention %"),
                             template=chart_template, aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

        # --- KPI SECTION (AC: M1, M3, M6, M12) ---
        st.subheader("Retention Milestones (Selected Cohort)")
        selected_cohort_kpi = st.selectbox("Wähle Kohorte für KPI Check:", retention_matrix.index)
        
        kpi_row = retention_matrix.loc[selected_cohort_kpi]
        cols = st.columns(4)
        for i, m in enumerate([1, 3, 6, 12]):
            val = kpi_row[m] if m in kpi_row else 0
            cols[i].metric(f"Retention Month {m}", f"{val:.1f}%")

else: # RFM Model Comparison
    st.header("Behavioral RFM Model Comparison")
    
    rfm_raw = duckdb.query("""
        SELECT 
            u.user_id, 
            strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
            date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_days,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_visits,
            COALESCE(SUM(e.revenue), 0) as m_revenue
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        GROUP BY 1, 2
    """).df()

    if len(final_selected) >= 2:
        c1, c2 = st.columns(2)
        opts = sorted(rfm_raw['joined_month'].unique())
        mA = c1.selectbox("Kohorte A (Modell):", opts, index=0)
        mB = c2.selectbox("Kohorte B (Modell):", opts, index=1)
        
        # Scatter Model Comparison
        comp_df = rfm_raw[rfm_raw['joined_month'].isin([mA, mB])]
        
        st.subheader("RFM Bubble Model: Recency vs Frequency (Size = Monetary)")
        fig_model = px.scatter(comp_df, x="r_days", y="f_visits", size="m_revenue", color="joined_month",
                               hover_name="user_id", log_x=False, size_max=40,
                               labels={"r_days": "Recency (Tage seit Besuch)", "f_visits": "Frequency (Anzahl Besuche)"},
                               template=chart_template, color_discrete_sequence=['#00CC96', '#636EFA'])
        # Optimierung: X-Achse invertieren (kleine Werte = besser)
        fig_model.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_model, use_container_width=True)

        # Bar Charts für harte Fakten
        st.markdown("---")
        stats = comp_df.groupby('joined_month').agg({'m_revenue': 'mean', 'f_visits': 'mean', 'r_days': 'mean'}).reset_index()
        b1, b2, b3 = st.columns(3)
        b1.plotly_chart(px.bar(stats, x='joined_month', y='m_revenue', color='joined_month', title="Avg Monetary Value (€)", template=chart_template), use_container_width=True)
        b2.plotly_chart(px.bar(stats, x='joined_month', y='f_visits', color='joined_month', title="Avg Frequency (Usage)", template=chart_template), use_container_width=True)
        b3.plotly_chart(px.bar(stats, x='joined_month', y='r_days', color='joined_month', title="Avg Recency (Days)", template=chart_template), use_container_width=True)
    else:
        st.info("Bitte wähle zwei Monate für das Vergleichsmodell aus.")

# --- 5. FOOTER ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | Optimized RFM & Retention Engine")