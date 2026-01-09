import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Growth Intelligence", layout="wide")

st.sidebar.title("App Settings")
# Dark mode is default now, no toggle needed
chart_template = "plotly_dark"

# --- 2. DATA ENGINE (Realistic SaaS Simulation) ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    users, events = [], []
    now = datetime(2026, 1, 8, 12, 0)
    for i in range(1, 1501):
        uid = f"User_{i:04d}"
        # Random acquisition date in 2025
        acq = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365))
        users.append([uid, acq])
        
        # Churn Logic: Realistic SaaS Retention (Shifted Hyperbolic)
        dates = pd.date_range(acq, periods=13, freq='30D')
        for m, v_date in enumerate(dates):
            # Prob(Active) drops then stabilizes
            # m=0: 100%, m=12: ~29%
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

# --- 3. HEADER & SIDEBAR ---
st.title("🚀 Product Growth Intelligence")

# View Selector
view_mode = st.radio("Select Analysis Mode:", ["📉 Retention Matrix", "👤 RFM Score Model"], horizontal=True)

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

# Master Checkbox
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

# --- 4. BODY CONTENT ---

# --- TAB: RETENTION (CALENDAR VIEW) ---
if view_mode == "📉 Retention Matrix":
    st.header("Cohort Retention (Calendar Month View)")
    
    # Logic: Pivot on Calendar Month (YYYY-MM) instead of m_num
    ret_query = """
        WITH uc AS (SELECT user_id, DATE_TRUNC('month', acq_date) as c_month FROM filtered_users),
        act AS (
            SELECT e.user_id, uc.c_month, DATE_TRUNC('month', e.event_date) as event_month
            FROM df_filtered_events e JOIN uc ON e.user_id = uc.user_id
            WHERE e.type = 'visit'
        )
        SELECT 
            strftime(c_month, '%Y-%m') as cohort_key,
            strftime(c_month, '%b %Y') as cohort_label, 
            strftime(event_month, '%Y-%m') as event_key,
            strftime(event_month, '%b %Y') as event_label,
            COUNT(DISTINCT user_id) as active_users
        FROM act 
        GROUP BY 1, 2, 3, 4 
        ORDER BY 1, 3
    """
    cohort_df = duckdb.query(ret_query).df()
    
    if not cohort_df.empty:
        # Get cohort sizes (M0) to normalize
        # M0 is when event_month == c_month
        cohort_sizes = cohort_df[cohort_df['cohort_key'] == cohort_df['event_key']].set_index('cohort_key')['active_users']
        
        # Pivot: Index=Cohort, Columns=EventMonth
        pivot = cohort_df.pivot(index='cohort_label', columns='event_label', values='active_users')
        
        # Sort rows/cols
        sorted_cohorts = cohort_df.sort_values('cohort_key')['cohort_label'].unique()
        sorted_events = cohort_df.sort_values('event_key')['event_label'].unique()
        
        pivot = pivot.reindex(index=sorted_cohorts, columns=sorted_events)
        
        # Masking N/A for Logic: 
        # Row: Cohort Start X. Column: Event Month Y.
        # If Y < X, it should be NaN (impossible).
        # We can iterate to strictly enforce this or rely on nulls if no data existed.
        # But if no data exist, it's NaN anyway.
        # We need to calculate percentages.
        
        # DataFrame for Percentages
        retention_pct = pd.DataFrame(index=pivot.index, columns=pivot.columns)
        
        for c_label in pivot.index:
            # Find the cohort key for this label to get size
            # (Assuming unique mapping label->key, strict one-to-one is safe here)
            # Use safe lookup
            matches = cohort_df[cohort_df['cohort_label'] == c_label]['cohort_key']
            if matches.empty: continue
            c_key = matches.iloc[0]
            
            size = cohort_sizes.get(c_key, 0)
            
            start_date = pd.to_datetime(c_key)
            
            if size > 0:
                for e_label in pivot.columns:
                    # Get event date from label (need mapping map back or parse)
                    # Easier: iterate stored keys.
                    # Let's direct parse labels if standard
                    try:
                        e_date = datetime.strptime(e_label, '%b %Y')
                    except:
                        continue
                        
                    # Logic Check:
                    # e_date is always 1st of month. start_date is 1st of month.
                    # If e_date < start_date, impossible
                    if e_date >= start_date:
                        val = pivot.loc[c_label, e_label]
                        if pd.notna(val):
                            retention_pct.loc[c_label, e_label] = (val / size) * 100.0
                        else:
                            # Valid month but no data -> 0%
                            retention_pct.loc[c_label, e_label] = 0.0
                    else:
                        # Impossible month -> leave NaN
                        retention_pct.loc[c_label, e_label] = np.nan
                        
        # Heatmap
        fig_heat = px.imshow(retention_pct.astype(float), 
                             text_auto='.1f', 
                             color_continuous_scale='RdYlGn',
                             labels=dict(x="Event Month", y="Cohort", color="Retention %"),
                             aspect="auto", 
                             template=chart_template)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for selection.")

# --- TAB: RFM SCORE MODELL ---
else: 
    st.header("RFM Score Model (Comparative View)")
    
    # 1. Calculate RFM Scores Global
    # 1. Calculate RFM Scores Global
    # Changed: r_raw is now in hours, f_visits, m_revenue
    rfm_query = """
        SELECT 
            u.user_id, 
            strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
            strftime(DATE_TRUNC('month', u.acq_date), '%Y-%m') as sort_key,
            date_diff('hour', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '2026-01-08 12:00:00') as r_hours,
            COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_visits,
            COALESCE(SUM(e.revenue), 0) as m_revenue
        FROM filtered_users u 
        LEFT JOIN df_filtered_events e ON u.user_id = e.user_id 
        GROUP BY 1, 2, 3
    """
    rfm_df = duckdb.query(rfm_query).df()

    def segment_behavior(row):
        # 1. RECENCY (Hours since last visit)
        # Handle NA: If no visits, r_hours is NA/NaN.
        if pd.isna(row['r_hours']):
            r = 'D. Inactive (No Visits)'
        elif row['r_hours'] < 1:
            r = 'A. < 1h'
        elif row['r_hours'] < 24:
            r = 'B. 1-24h'
        else:
            r = 'C. > 24h'
            
        # 2. FREQUENCY (Visits)
        if row['f_visits'] > 25:
            f = 'A. Power User'
        elif row['f_visits'] > 5:
            f = 'B. Regular'
        else:
            f = 'C. Casual'
            
        # 3. MONETARY (Revenue)
        if row['m_revenue'] > 200:
            m = 'A. Big Spender'
        elif row['m_revenue'] > 0:
            m = 'B. Customer'
        else:
            if r == 'D. Inactive (No Visits)':
                m = 'D. Non-Converted' # Consistent labeling
            else:
                m = 'C. Window Shopper' # Active but no spend
                
        return pd.Series({'Recency': r, 'Frequency': f, 'Monetary': m})

    # Apply the segmentation
    rfm_df[['Recency', 'Frequency', 'Monetary']] = rfm_df.apply(segment_behavior, axis=1)
    rfm_df['RFM_Group'] = rfm_df['Recency'] + " | " + rfm_df['Frequency'] + " | " + rfm_df['Monetary']
    
    rfm_scored = rfm_df.copy()
    
    # Sort cohorts
    sorted_cohorts_df = rfm_scored[['sort_key', 'joined_month']].drop_duplicates().sort_values('sort_key')
    available_cohorts = sorted_cohorts_df['joined_month'].tolist()

    # View Controls
    col_mode, col_space = st.columns([1, 3])
    compare_mode = col_mode.toggle("Enable Comparison Mode", value=False)
    
    # Function to generate Heatmap Figure
    def make_rfm_heatmap(df_subset, title):
        if df_subset.empty:
            return None
        # Use new columns: Frequency, Recency, m_revenue
        hm_data = df_subset.groupby(['Frequency', 'Recency'])['m_revenue'].mean().reset_index()
        hm_pivot = hm_data.pivot(index='Frequency', columns='Recency', values='m_revenue')
        
        # Ensure all keys exist for consistent shape if possible, or just let it be dynamic
        # The labels are A, B, C, D so automatic alphabetical sorting works well (A=Best/Recent)
        # We want Best Top-Left?
        # A is Best (Power User / Recent)
        # So we want A -> D
        
        # Sort index/cols
        hm_pivot = hm_pivot.sort_index(axis=0).sort_index(axis=1)
        
        fig = px.imshow(hm_pivot, text_auto='.0f', color_continuous_scale='Viridis',
                        labels=dict(x="Recency", y="Frequency", color="Avg Rev €"),
                        title=title, template=chart_template)
        return fig

    # --- HEATMAP SECTION ---
    st.subheader("RFM Frequency/Recency Heatmap")
    
    if compare_mode:
        c1, c2 = st.columns(2)
        with c1:
            sel_a = st.selectbox("Cohort A", available_cohorts, index=0)
            df_a = rfm_scored[rfm_scored['joined_month'] == sel_a]
            fig_a = make_rfm_heatmap(df_a, f"Cohort: {sel_a}")
            if fig_a: st.plotly_chart(fig_a, use_container_width=True)
            
        with c2:
            sel_b = st.selectbox("Cohort B", available_cohorts, index=min(1, len(available_cohorts)-1))
            df_b = rfm_scored[rfm_scored['joined_month'] == sel_b]
            fig_b = make_rfm_heatmap(df_b, f"Cohort: {sel_b}")
            if fig_b: st.plotly_chart(fig_b, use_container_width=True)
            
    else:
        # Global or Multi-select View
        # Default intersection with master filter
        preselected_cohorts = [c for c in available_cohorts if c in [d.strftime('%b %Y') for d in current_selection]]
        
        sel_cohorts = st.multiselect("Select Cohorts for Heatmap:", available_cohorts, default=preselected_cohorts)
        
        if sel_cohorts:
            df_sub = rfm_scored[rfm_scored['joined_month'].isin(sel_cohorts)]
            fig_main = make_rfm_heatmap(df_sub, "Combined Selection Heatmap")
            if fig_main: st.plotly_chart(fig_main, use_container_width=True)

    # --- BAR CHART SECTION ---
    st.subheader("Top RFM Combinations")
    
    # RFM Group Selector
    all_groups = sorted(rfm_scored['RFM_Group'].unique())
    sel_groups = st.multiselect("Filter by RFM Group:", all_groups, default=[])
    
    # Filter Data for Chart
    chart_df = rfm_scored.copy()
    if compare_mode:
        # restrict to A/B
        chart_df = chart_df[chart_df['joined_month'].isin([sel_a, sel_b])]
    else:
        # restrict to main selection
        if 'sel_cohorts' in locals() and sel_cohorts:
             chart_df = chart_df[chart_df['joined_month'].isin(sel_cohorts)]
    
    if sel_groups:
        chart_df = chart_df[chart_df['RFM_Group'].isin(sel_groups)]
        
    if not chart_df.empty:
        group_stats = chart_df.groupby(['joined_month', 'RFM_Group']).size().reset_index(name='count')
        group_stats = group_stats.sort_values(by="count", ascending=False)
        
        fig_groups = px.bar(group_stats, x="RFM_Group", y="count", color="joined_month", barmode="group",
                            template=chart_template)
        st.plotly_chart(fig_groups, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

# --- 5. FOOTER ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | Comparison Mode & Calendar View")