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

st.sidebar.subheader("Filter: User Cohorts (Join Date)")

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
    # Dynamic "Reference Month" Selector
    # Create list of available months from data (up to last month to allow +1 month projection)
    # Use cohort months as proxies for "months available in data"
    # End date of simulation is 2026-01-08.
    
    # Generate list of First Days of months available in the dataset range
    # Min Date: 2025-01-01, Max Date: 2026-01-08
    # We want to select "Base Month" (Left). Right will be Base + 1 Month.
    # So Base Month can be up to Dec 2025 (so next is Jan 2026).
    
    sim_start = datetime(2025, 1, 1)
    sim_end = datetime(2026, 1, 1) # Last full month start
    available_months_dt = pd.date_range(sim_start, sim_end, freq='MS')
    available_months_str = [d.strftime('%b %Y') for d in available_months_dt]
    
    # --- 2. LAYOUT & FILTERS ---
    
    col_filter, col_content = st.columns([1, 3])
    
    with col_filter:
        st.subheader("Filter Settings")
        
        # Reference Month Selector
        # Default to 2nd to last (Nov 2025) so Dec 2025 is next, or similar.
        default_idx = len(available_months_str) - 2 if len(available_months_str) > 1 else 0
        selected_month_str = st.selectbox("Select Analysis Month (Timeframe):", options=available_months_str, index=default_idx)
        
        # Calculate Dates
        base_date = datetime.strptime(selected_month_str, '%b %Y')
        next_date = base_date + pd.DateOffset(months=1)
        
        # For RFM snapshot calculation, we usually take the END of that month?
        # User says "Month that is old" (Left) vs "Month that follows" (Right).
        # Let's say we snapshot at the End of the selected months? OR Beginning?
        # Usually RFM is "As of".
        # Let's use End of Month for robustness, or +30 days. 
        # Simpler: "As of 1st of Next Month" == Status of Previous Month?
        # Let's set snapshot dates to the 1st of the displayed months to catch "State at that time".
        
        # Actually user wants "Month Left" -> "Month Right".
        # Left: RFM State at end of Base Month?
        # Right: RFM State at end of Next Month?
        # Let's use:
        # Prev Date = Base Month + 1 Month (Start) - 1 Day (End of Base)? 
        # Let's stick to: Snapshot at precise points. 
        # Snapshot 1 (Left): base_date + 30 days (approx end of month)
        # Snapshot 2 (Right): next_date + 30 days
        
        # Let's just use the 1st of the Next Month as the cutoff for the "Month" status.
        # e.g. Status of "Jan" is captured at Feb 1st. 
        
        date_left = base_date + pd.DateOffset(months=1) # Proxy for End of Base Month
        date_right = next_date + pd.DateOffset(months=1) # Proxy for End of Next Month
        
        # A. Cohort Filter (Already handled by global)
        
        # Recalculate RFM based on dynamic dates
        rfm_left_raw = get_rfm_at_date(date_left)
        rfm_right_raw = get_rfm_at_date(date_right)
        
        rfm_left = calculate_rfm_scores(rfm_left_raw)
        rfm_right = calculate_rfm_scores(rfm_right_raw)
        
        # B. RFM Group Filter - Single Select
        all_groups = sorted(rfm_left['RFM_Group'].unique()) if not rfm_left.empty else []
        if all_groups:
             all_groups.insert(0, "All")
        
        selected_group = st.selectbox("Filter RFM Group:", options=all_groups, index=0)
        
        # C. Cohort Subset (for Heatmap)
        available_cohorts = rfm_left.sort_values('sort_key')['joined_month'].unique()
        preselected = [c for c in available_cohorts if pd.to_datetime(c).strftime('%Y-%m') in [d.strftime('%Y-%m') for d in current_selection]]
        
        # Filter Logic
        mask_left = pd.Series(True, index=rfm_left.index)
        mask_right = pd.Series(True, index=rfm_right.index)
        
        if selected_group != "All":
            mask_left = mask_left & (rfm_left['RFM_Group'] == selected_group)
            mask_right = mask_right & (rfm_right['RFM_Group'] == selected_group)
            
        filtered_left = rfm_left[mask_left]
        filtered_right = rfm_right[mask_right]

    # --- 3. TOP SUMMARY (MoM Flow) ---
    with col_content:
        st.markdown("### Month-Over-Month Flow")
        
        # Counts
        count_left = filtered_left['user_id'].nunique()
        count_right = filtered_right['user_id'].nunique()
        diff = count_right - count_left
        
        c1, c_arr, c2 = st.columns([2,1,2])
        
        label_left = base_date.strftime('%B %Y')
        label_right = next_date.strftime('%B %Y')
        
        with c1:
            st.metric(label=f"Users ({label_left})", value=count_left)
            
        with c_arr:
             st.markdown(f"<h2 style='text-align: center; color: {'green' if diff >= 0 else 'red'};'>{'➜' if diff == 0 else ('↗' if diff > 0 else '↘')}</h2>", unsafe_allow_html=True)
        
        with c2:
            st.metric(label=f"Users ({label_right})", value=count_right, delta=int(diff))
            
        st.markdown("---")

        # --- 4. DATA PREP FOR HEATMAP (DELTA) ---
        # Heatmap shows GENERAL change (All Groups), but filtered by Cohort?
        # User: "IN RFM Group Heatmap möchte ich im allgemeinen die Veränderung sehen"
        # BUT: "Comparison of RFM combination should reflect the selected groups" -> This applies to the Bar Chart.
        # Implication: Heatmap stays BROAD (General), Bar Chart becomes SPECIFIC (Filtered).
        
        # Recalculate "General" snapshot for Heatmap (ignoring RFM Group Filter, but respecting Cohort Filter via get_rfm_at_date)
        # Actually... get_rfm_at_date relies on `filtered_users` which is controlled by sidebar. 
        # So rfm_left/rfm_right ARE the general population for the selected cohorts.
        
        hm_left = rfm_left.groupby(['F', 'R']).size().reset_index(name='count')
        hm_right = rfm_right.groupby(['F', 'R']).size().reset_index(name='count')
        
        hm_merged = pd.merge(hm_left, hm_right, on=['F', 'R'], how='outer', suffixes=('_left', '_right')).fillna(0)
        hm_merged['delta'] = hm_merged['count_right'] - hm_merged['count_left']
        
        # Pivot for Heatmap
        heatmap_pivot = hm_merged.pivot(index='R', columns='F', values='delta').fillna(0)
        
        # Sorting
        heatmap_pivot = heatmap_pivot.reindex(index=["3", "2", "1"], columns=["1", "2", "3"])
        
        st.subheader(f"RFM Delta Heatmap ({label_left} ➜ {label_right})")
        fig_rfm_heat = px.imshow(heatmap_pivot, text_auto='+d', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                                 labels=dict(x="Frequency (1=Low, 3=High)", y="Recency (3=Recent, 1=Old)", color="Change"),
                                 template=chart_template)
        st.plotly_chart(fig_rfm_heat, use_container_width=True)

        # --- 5. GROUP COMPARISON (BAR CHART) ---
        st.subheader("Comparison of Selected RFM Groups")
        
        # Filtered Data (Respects Single Group Selection)
        if filtered_left.empty:
            st.info("No users in selected groups.")
        else:
            # Show ONLY the selected group if specific, or all if "All"
            # Bar chart likely expects to compare Segments. 
            # If "All" selected -> Show all groups.
            # If specific group selected -> Show just that group? Or maybe breakdown by Cohort?
            # User: "Comparison of RFM combination should reflect the selected groups"
            
            group_stats = filtered_left.groupby(['joined_month', 'RFM_Group']).size().reset_index(name='count')
            group_stats = group_stats.sort_values(by="count", ascending=False)
            
            fig_groups = px.bar(group_stats, x="RFM_Group", y="count", color="joined_month", barmode="group",
                                template=chart_template, title=f"Composition ({label_left})")
            st.plotly_chart(fig_groups, use_container_width=True)

# --- 5. FOOTER AREA ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | RFM MoM Analysis")