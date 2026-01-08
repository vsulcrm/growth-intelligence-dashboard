import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Growth Intelligence", layout="wide")

st.sidebar.title("App Settings")
chart_template = "plotly_dark"

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
        
        # Churn-Logik für Realistic SaaS Retention (Shifted Hyperbolic / Power Law)
        # 1 / (1 + alpha * t) mimics "early drop, then flatten"
        # t is month index (0 to 12)
        dates = pd.date_range(acq, periods=13, freq='30D')
        for m, v_date in enumerate(dates):
            # Retention Curve: High drop month 1-3, stabilizes
            # Prob(alive) at month m. If alive, generate visit.
            
            # Simple approach: P(Active at m) = 1.0 / (1.0 + 0.5 * m)
            # m=0: 100%, m=1: 66%, m=2: 50%, m=12: ~14%
            # Modified to be a bit stickier: 1.0 / (1.0 + 0.2 * m) -> m12=29%
            prob_active = 1.0 / (1.0 + 0.25 * m)
            
            if np.random.random() < prob_active:
                v_date_jitter = v_date + timedelta(days=np.random.randint(0, 10))
                if v_date_jitter < now:
                    events.append([uid, v_date_jitter, "visit", 0])
                    if np.random.random() > 0.8: # Purchase prob if active
                        events.append([uid, v_date_jitter, "purchase", np.random.uniform(20, 500)])
    
    return pd.DataFrame(users, columns=["user_id", "acq_date"]), pd.DataFrame(events, columns=["user_id", "event_date", "type", "revenue"])

df_users, df_events = generate_data()
df_users['cohort'] = df_users['acq_date'].dt.to_period('M').dt.to_timestamp()
all_months = sorted(df_users['cohort'].unique())

# --- 3. HEADER BEREICH (Logik & Sidebar) ---
st.title("🚀 Product Growth Intelligence")

# View Selector
view_mode = st.radio("Analyseschwerpunkt wählen:", ["📉 Retention Matrix", "👤 RFM Score Modell"], horizontal=True)

st.sidebar.subheader("Kohorten-Filter")

# Session State für Checkboxen
if 'selected_months' not in st.session_state:
    st.session_state.selected_months = [m for m in all_months]

def toggle_all():
    new_state = st.session_state.master_cb
    for m in all_months:
        m_str = m.strftime('%Y-%m')
        st.session_state[f"cb_{m_str}"] = new_state
    
    if new_state:
        st.session_state.selected_months = [m for m in all_months]
    else:
        st.session_state.selected_months = []

# Master Checkbox (Select/Deselect All)
st.sidebar.checkbox("Alle Monate an/abwählen", value=True, 
                    key="master_cb", on_change=toggle_all)

current_selection = []
for m in all_months:
    m_str = m.strftime('%Y-%m')
    m_label = m.strftime('%B %Y')
    
    # Init default state
    if f"cb_{m_str}" not in st.session_state:
        st.session_state[f"cb_{m_str}"] = True
        
    if st.sidebar.checkbox(m_label, key=f"cb_{m_str}"):
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
    st.header("Kohorten-Retention (Chronologische 12-Monats-Sicht)")
    
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
        # Fix NameError: Chronologische Sortierung sicherstellen
        ordered_labels = cohort_data.sort_values('sort_key')['cohort'].unique()
        pivot = pivot.reindex(ordered_labels)
        
        # Normalisierung (Monat 0 = 100%)
        retention_matrix = (pivot.divide(pivot.iloc[:, 0], axis=0) * 100).clip(upper=100.0)
        
        fig_heat = px.imshow(retention_matrix, text_auto='.1f', color_continuous_scale='RdYlGn',
                             labels=dict(x="Monate nach Akquise", y="Kohorte", color="Retention %"),
                             aspect="auto", template=chart_template)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Retention Milestones (M1, M3, M6, M12)")
        sel_c = st.selectbox("Kohorte für KPI-Check:", options=retention_matrix.index)
        kpi = retention_matrix.loc[sel_c]
        c1, c2, c3, c4 = st.columns(4)
        for i, m in enumerate([1, 3, 6, 12]):
            val = kpi[m] if m in kpi else 0
            [c1, c2, c3, c4][i].metric(f"Monat {m}", f"{val:.1f}%")

# --- TAB: RFM SCORE MODELL ---
else: 
    st.header("RFM Score Modell Vergleich (1-3 Tertile)")
    
    rfm_query = """
        SELECT 
            u.user_id, 
            strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
            strftime(DATE_TRUNC('month', u.acq_date), '%Y-%m') as sort_key,
            date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_raw,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_raw,
            COALESCE(SUM(e.revenue), 0) as m_raw
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        GROUP BY 1, 2, 3
    """
    rfm_df = duckdb.query(rfm_query).df()

    # Scoring Logik (1-3 Tertile)
    def calculate_rfm_scores(df):
        df = df.copy()
        df['R'] = pd.qcut(df['r_raw'], 3, labels=["3", "2", "1"]).astype(str)
        df['F'] = pd.qcut(df['f_raw'].rank(method='first'), 3, labels=["1", "2", "3"]).astype(str)
        df['M'] = pd.qcut(df['m_raw'].rank(method='first'), 3, labels=["1", "2", "3"]).astype(str)
        df['RFM_Group'] = df['R'] + "-" + df['F'] + "-" + df['M']
        return df

    rfm_scored = calculate_rfm_scores(rfm_df)
    
    # --- RFM HEATMAP (ZUSATZ-WUNSCH) ---
    st.subheader("RFM Group Heatmap: Frequency vs Recency (Color = Avg Revenue)")
    
    # Allow specific cohort selection based on current global filter
    available = rfm_scored.sort_values('sort_key')['joined_month'].unique()
    # Which ones are currently "active" in the main filter?
    preselected = [c for c in available if pd.to_datetime(c).strftime('%Y-%m') in [d.strftime('%Y-%m') for d in current_selection]]
    
    selected_heatmap_cohorts = st.multiselect("Kohorten für Heatmap auswählen (Teilmenge der globalen Auswahl):", 
                                              options=available, 
                                              default=preselected)
    
    if not selected_heatmap_cohorts:
        st.warning("Keine Kohorten für Heatmap ausgewählt.")
    else:
        # Aggregation für Heatmap
        heatmap_data = rfm_scored[rfm_scored['joined_month'].isin(selected_heatmap_cohorts)].groupby(['F', 'R'])['m_raw'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='F', columns='R', values='m_raw')
        
        # Sort index/columns to ensure order 3-2-1 if necessary, though 1-2-3 is sorted string
        # Recency: 3=Best/Newest, 1=Oldest.
        # Frequency: 3=High, 1=Low.
        # We want X=Recency, Y=Frequency.
        # Order should be R: 3, 2, 1 (New -> Old) ? Or 1, 2, 3? 
        # Usually R score 3 is "Recent" (Good).
        
        fig_rfm_heat = px.imshow(heatmap_pivot, text_auto='.0f', color_continuous_scale='Viridis',
                                 labels=dict(x="Recency Score (3=Frisch)", y="Frequency Score (3=Oft)", color="Ø Umsatz €"),
                                 template=chart_template)
        st.plotly_chart(fig_rfm_heat, use_container_width=True)

    # Gruppen-Vergleich (Bar)
    st.subheader("Vergleich der Top RFM-Kombinationen")
    
    comp_df = rfm_scored[rfm_scored['joined_month'].isin(selected_heatmap_cohorts)]
    group_stats = comp_df.groupby(['joined_month', 'RFM_Group']).size().reset_index(name='count')
    
    # Sortierung: Immer absteigend nach Anzahl (Gesamt oder pro Gruppe)
    # Hier simple sort by count descending across all rows
    group_stats = group_stats.sort_values(by="count", ascending=False)
    
    fig_groups = px.bar(group_stats, x="RFM_Group", y="count", color="joined_month", barmode="group",
                        template=chart_template)
    st.plotly_chart(fig_groups, use_container_width=True)

# --- 5. FOOTER BEREICH ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | RFM 1-3 Model & Retention Heatmap")