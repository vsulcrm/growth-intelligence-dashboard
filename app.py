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

# --- 2. DATA ENGINE (Realistic Simulation) ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    now = datetime(2026, 1, 8, 12, 0)
    for i in range(1, 1501):
        uid = f"User_{i:04d}"
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365))
        users.append([uid, acq])
        
        # Churn Logic for Realistic SaaS Retention (Shifted Hyperbolic / Power Law)
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

# --- 3. HEADER AREA (Logic & Sidebar) ---
st.title("🚀 Product Growth Intelligence")

# View Selector
view_mode = st.radio("Select Analysis Focus:", ["📉 Retention Matrix", "👤 RFM Score Model"], horizontal=True)

st.sidebar.subheader("Cohort Filter")

# Session State for Checkboxes
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
st.sidebar.checkbox("Select/Deselect All Months", value=True, 
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
    st.warning("Please select at least one cohort.")
    st.stop()

filtered_users = df_users[df_users['cohort'].isin(current_selection)]
df_filtered_events = df_events[df_events['user_id'].isin(filtered_users['user_id'])]

st.markdown("---")

# --- 4. BODY AREA ---

# --- TAB: RETENTION ---
if view_mode == "📉 Retention Matrix":
    st.header("Cohort Retention (Chronological 12-Month View)")
    
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
        
        # Normalization (Month 0 = 100%)
        retention_matrix = (pivot.divide(pivot.iloc[:, 0], axis=0) * 100).clip(upper=100.0)
        
        fig_heat = px.imshow(retention_matrix, text_auto='.1f', color_continuous_scale='RdYlGn',
                             labels=dict(x="Months Since Acquisition", y="Cohort", color="Retention %"),
                             aspect="auto", template=chart_template)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Retention Milestones (M1, M3, M6, M12)")
        sel_c = st.selectbox("Cohort for KPI Check:", options=retention_matrix.index)
        kpi = retention_matrix.loc[sel_c]
        c1, c2, c3, c4 = st.columns(4)
        for i, m in enumerate([1, 3, 6, 12]):
            val = kpi[m] if m in kpi else 0
            [c1, c2, c3, c4][i].metric(f"Month {m}", f"{val:.1f}%")

# --- TAB: RFM SCORE MODEL ---
else: 
    st.header("RFM Score Model Comparison (1-3 Tertiles)")
    
    # --- 1. LOGIC: SNAPSHOT CALCULATION ---
    # function to calculate RFM table at a specific reference date
    def get_rfm_at_date(ref_date):
        # Filter events up to ref_date
        # We also need to respect the "Cohort Filter" (filtered_users is already filtered by cohort)
        
        # We need a fresh query for the specific date cutoff
        # ref_date needs to be string for SQL
        ref_date_str = ref_date.strftime('%Y-%m-%d %H:%M:%S')
        
        q = f"""
            SELECT 
                u.user_id, 
                strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
                strftime(DATE_TRUNC('month', u.acq_date), '%Y-%m') as sort_key,
                date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '{ref_date_str}') as r_raw,
                COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_raw,
                COALESCE(SUM(e.revenue), 0) as m_raw
            FROM filtered_users u 
            LEFT JOIN df_filtered_events e ON u.user_id = e.user_id AND e.event_date <= '{ref_date_str}'
            GROUP BY 1, 2, 3
        """
        df = duckdb.query(q).df()
        return df

    # Scoring Logic (1-3 Tertiles) - Shared thresholds or recalculate per snapshot?
    # Usually standard is to recalibrate, but for MoM comparison, keeping thresholds constant OR recalibrating
    # is a choice. Let's recalibrate each time to see relative movement, OR stick to fixed.
    # For simplicity/robustness: Recalculate tertiles on the *current* dataset, then apply to past?
    # Actually, simpler: Just calculate scores independently for Now and Prev to see "Relative Standing" shifts.
    
    def calculate_rfm_scores(df):
        if df.empty: return df
        df = df.copy()
        # Handle NA for users with no events (Recency usually NaN) -> Fill with max? 
        # For simplicity, if r_raw is NaN (no visits), set to large number
        df['r_raw'] = df['r_raw'].fillna(9999)
        
        # 3 buckets
        try:
            df['R'] = pd.qcut(df['r_raw'], 3, labels=["3", "2", "1"], duplicates='drop').astype(str) # 3 is best (lowest recency)
            df['F'] = pd.qcut(df['f_raw'].rank(method='first'), 3, labels=["1", "2", "3"]).astype(str)
            df['M'] = pd.qcut(df['m_raw'].rank(method='first'), 3, labels=["1", "2", "3"]).astype(str)
        except ValueError:
            # Fallback if too little data
            df['R'] = "1"
            df['F'] = "1"
            df['M'] = "1"
            
        df['RFM_Group'] = df['R'] + "-" + df['F'] + "-" + df['M']
        return df

    # Data Points
    now_date = datetime(2026, 1, 8, 12, 0)
    prev_date = now_date - timedelta(days=30)
    
    rfm_now_raw = get_rfm_at_date(now_date)
    rfm_prev_raw = get_rfm_at_date(prev_date)
    
    rfm_now = calculate_rfm_scores(rfm_now_raw)
    rfm_prev = calculate_rfm_scores(rfm_prev_raw)
    
    # --- 2. LAYOUT & FILTERS ---
    
    col_filter, col_content = st.columns([1, 3])
    
    with col_filter:
        st.subheader("Filter Settings")
        
        # A. Cohort Filter (Already handled by global, but we show it or allow subset)
        # B. RFM Group Filter
        all_groups = sorted(rfm_now['RFM_Group'].unique()) if not rfm_now.empty else []
        selected_groups = st.multiselect("Filter RFM Groups:", options=all_groups, default=[])
        
        # C. Cohort Subset (for Heatmap specifically)
        available_cohorts = rfm_now.sort_values('sort_key')['joined_month'].unique()
        preselected = [c for c in available_cohorts if pd.to_datetime(c).strftime('%Y-%m') in [d.strftime('%Y-%m') for d in current_selection]]
        
        # Note: If no RFM group selected, assume ALL
        mask_now = pd.Series(True, index=rfm_now.index)
        mask_prev = pd.Series(True, index=rfm_prev.index)
        
        if selected_groups:
            mask_now = mask_now & rfm_now['RFM_Group'].isin(selected_groups)
            mask_prev = mask_prev & rfm_prev['RFM_Group'].isin(selected_groups)
            
        filtered_now = rfm_now[mask_now]
        filtered_prev = rfm_prev[mask_prev]

    # --- 3. TOP SUMMARY (MoM Flow) ---
    with col_content:
        st.markdown("### User Flow Summary (MoM Change)")
        
        # Counts
        count_now = filtered_now['user_id'].nunique()
        count_prev = filtered_prev['user_id'].nunique()
        diff = count_now - count_prev
        
        c1, c_arr, c2 = st.columns([2,1,2])
        
        with c1:
            st.metric(label=f"Users ({prev_date.strftime('%b %d')})", value=count_prev)
            
        with c_arr:
             st.markdown(f"<h2 style='text-align: center; color: {'green' if diff >= 0 else 'red'};'>{'➜' if diff == 0 else ('↗' if diff > 0 else '↘')}</h2>", unsafe_allow_html=True)
        
        with c2:
            st.metric(label=f"Users ({now_date.strftime('%b %d')})", value=count_now, delta=int(diff))
            
        st.markdown("---")

        # --- 4. DATA PREP FOR HEATMAP (DELTA) ---
        # We need counts per (F, R) bucket for Now and Prev, then subtract
        
        # Full aggregation (independent of RFM Group filter for the heatmap grid? 
        # User said: "IN RFM Group Heatmap möchte ich im allgemeinen die Veränderung sehen... Die Heatmap zeigt den aktuellen Monat und soll mit + oder - Werten Zeigen"
        # Usually Heatmap shows the whole landscape. If I filter to "1-1-1", the heatmap would only show one cell.
        # User requirement: "Filter applicable to... Comparison of RFM combination should reflect selected groups". 
        # It seems the Heatmap should probably show the *General* change to provide context, or be filtered?
        # "Filter which group I want to select... User flow summary should show... In RFM Group Heatmap I want to see the change IN GENERAL"
        # -> inferred: Heatmap shows ALL groups (Unfiltered by RFM Group), but Filter applies to Summary & Bar Chart.
        
        # However, clarity: "RFM Group Heatmap möchte ich im allgemeinen die Veränderung sehen" -> "In General".
        # So I will use the Full Data (filtered by Cohort, but NOT by RFM Group selector) for Heatmap.
        
        hm_now = rfm_now.groupby(['F', 'R']).size().reset_index(name='count')
        hm_prev = rfm_prev.groupby(['F', 'R']).size().reset_index(name='count')
        
        hm_merged = pd.merge(hm_now, hm_prev, on=['F', 'R'], how='outer', suffixes=('_now', '_prev')).fillna(0)
        hm_merged['delta'] = hm_merged['count_now'] - hm_merged['count_prev']
        
        # Pivot for Heatmap
        # Axes: Y = Recency (3-2-1), X = Frequency (1-2-3)
        heatmap_pivot = hm_merged.pivot(index='R', columns='F', values='delta').fillna(0)
        
        # Sorting Index (Rows: R) -> 3, 2, 1 (Top to Bottom)
        # Sorting Cols (Cols: F) -> 1, 2, 3 (Left to Right)
        heatmap_pivot = heatmap_pivot.reindex(index=["3", "2", "1"], columns=["1", "2", "3"])
        
        st.subheader("RFM Delta Heatmap (MoM Change in User Count)")
        fig_rfm_heat = px.imshow(heatmap_pivot, text_auto='+d', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                                 labels=dict(x="Frequency (1=Low, 3=High)", y="Recency (3=Recent, 1=Old)", color="Change"),
                                 template=chart_template)
        st.plotly_chart(fig_rfm_heat, use_container_width=True)

        # --- 5. GROUP COMPARISON (BAR CHART) ---
        st.subheader("Comparison of Selected RFM Groups")
        
        # This IS affected by the filter
        if filtered_now.empty:
            st.info("No users in selected groups.")
        else:
            group_stats = filtered_now.groupby(['joined_month', 'RFM_Group']).size().reset_index(name='count')
            group_stats = group_stats.sort_values(by="count", ascending=False)
            
            fig_groups = px.bar(group_stats, x="RFM_Group", y="count", color="joined_month", barmode="group",
                                template=chart_template, title=f"Composition of Selected Groups ({now_date.strftime('%Y-%m-%d')})")
            st.plotly_chart(fig_groups, use_container_width=True)

# --- 5. FOOTER AREA ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | RFM MoM Analysis")