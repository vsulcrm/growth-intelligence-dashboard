import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Growth Intelligence Dashboard", layout="wide")

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
st.title("Growth Intelligence Dashboard")

# View Selector
view_mode = st.radio("Select Analysis Focus:", ["📉 Retention Matrix", "👤 RFM Score Model", "💰 LTV Prediction"], horizontal=True)

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
        # Logic to mask FUTURE months as NaN
        # Current Real Time: 2026-01-08 (Simulation "Now")
        # For each cohort, calculate max reachable month index
        sim_now = datetime(2026, 1, 8, 12, 0)
        
        retention_matrix_pct = (pivot.divide(pivot.iloc[:, 0], axis=0) * 100).clip(upper=100.0)
        
        # Mask future cells
        # Cohort Date + Month Index * 30 Days > sim_now? -> NaN
        # Simpler: Iterate and mask
        for cohort_name in retention_matrix_pct.index:
            # Parse cohort date from index (e.g. "Jan 2025")
            c_date = datetime.strptime(cohort_name, '%b %Y')
            for col_m in retention_matrix_pct.columns:
                # Approx month date
                m_date = c_date + pd.DateOffset(months=int(col_m))
                if m_date > sim_now:
                     retention_matrix_pct.loc[cohort_name, col_m] = None

        fig_heat = px.imshow(retention_matrix_pct, text_auto='.1f', color_continuous_scale='RdYlGn',
                             labels=dict(x="Months Since Acquisition", y="Cohort", color="Retention %"),
                             aspect="auto", template=chart_template)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Retention Milestones (M1, M3, M6, M12)")
        sel_c = st.selectbox("Cohort for KPI Check:", options=retention_matrix_pct.index)
        kpi = retention_matrix_pct.loc[sel_c]
        c1, c2, c3, c4 = st.columns(4)
        for i, m in enumerate([1, 3, 6, 12]):
            val = kpi[m] if (m in kpi and pd.notna(kpi[m])) else 0
            # If NaN (Future), show N/A
            disp_val = f"{val:.1f}%" if (m in kpi and pd.notna(kpi[m])) else "N/A"
            [c1, c2, c3, c4][i].metric(f"Month {m}", disp_val)

# --- TAB: LTV PREDICTION ---
elif view_mode == "💰 LTV Prediction":
    st.header("LTV Prediction (Linear Projection)")
    st.caption("Projected Lifetime Value based on cumulative revenue per cohort.")
    
    # 1. Calc Revenue per Cohort per Month
    ltv_query = """
        WITH uc AS (SELECT user_id, DATE_TRUNC('month', acq_date) as c_month FROM filtered_users),
        rev AS (
            SELECT e.user_id, uc.c_month,
            (EXTRACT(year FROM e.event_date) - EXTRACT(year FROM uc.c_month)) * 12 +
            (EXTRACT(month FROM e.event_date) - EXTRACT(month FROM uc.c_month)) as m_num,
            e.revenue
            FROM df_filtered_events e JOIN uc ON e.user_id = uc.user_id
            WHERE e.type = 'purchase'
        )
        SELECT strftime(c_month, '%b %Y') as cohort,
               c_month,
               m_num, 
               SUM(revenue) as total_rev
        FROM rev
        WHERE m_num BETWEEN 0 AND 24 
        GROUP BY 1, 2, 3
    """
    # Note: Need total USERS per cohort to normalize ARPU
    cohort_sizes = df_users.groupby('cohort').size().reset_index(name='size')
    cohort_sizes['cohort_label'] = cohort_sizes['cohort'].dt.strftime('%b %Y')
    
    rev_data = duckdb.query(ltv_query).df()
    
    if rev_data.empty:
        st.warning("No revenue data available.")
    else: 
        # Calculate Cumulative ARPU
        rev_pivot = rev_data.pivot(index='cohort', columns='m_num', values='total_rev').fillna(0)
        # Ensure correct order
        rev_pivot = rev_pivot.reindex([d.strftime('%b %Y') for d in sorted(pd.to_datetime(rev_pivot.index, format='%b %Y'))])
        
        cum_rev = rev_pivot.cumsum(axis=1)
        
        # Divide by cohort size
        arpu_df = cum_rev.copy()
        
        for c in arpu_df.index:
            size_row = cohort_sizes[cohort_sizes['cohort_label'] == c]
            if not size_row.empty:
                size = size_row.iloc[0]['size']
                arpu_df.loc[c] = arpu_df.loc[c] / size
        
        # Projection Logic (Simple Linear Regression per cohort)
        # Using numpy polyfit
        
        sim_now = datetime(2026, 1, 8, 12, 0)
        
        # 1. First Pass: Calculate Slopes for cohorts with enough data
        slopes = []
        cohort_models = {}
        
        for cohort in arpu_df.index:
            row = arpu_df.loc[cohort]
            c_date = datetime.strptime(cohort, '%b %Y')
            
            x_known = []
            y_known = []
            
            for m_idx in row.index:
                 if c_date + pd.DateOffset(months=int(m_idx)) <= sim_now:
                     valid_val = row[m_idx]
                     x_known.append(int(m_idx))
                     y_known.append(valid_val)
            
            if len(x_known) > 1:
                coef = np.polyfit(x_known, y_known, 1) # [slope, intercept]
                slopes.append(coef[0])
                cohort_models[cohort] = {'m': coef[0], 'c': coef[1], 'x_known': x_known, 'y_known': y_known}
            else:
                cohort_models[cohort] = {'m': None, 'c': None, 'x_known': x_known, 'y_known': y_known}

        # Avg Slope Fallback
        avg_slope = np.mean(slopes) if slopes else 0
        
        # 2. Build LTV Matrix (Cohorts x Month 0-12)
        # Rows: All Cohorts
        # Cols: 0 to 12
        ltv_matrix_data = [] # List of dicts
        
        for cohort in arpu_df.index:
            model = cohort_models[cohort]
            
            # Determine Coefs to use
            if model['m'] is not None:
                m = model['m']
                c = model['c']
                formula_str = f"LTV = {m:.2f} * Month + {c:.2f}"
            else:
                m = avg_slope
                if model['x_known']:
                    last_x = model['x_known'][-1]
                    last_y = model['y_known'][-1]
                    c = last_y - (m * last_x)
                else:
                    c = 0 
                formula_str = f"LTV = {m:.2f} * Month + {c:.2f} (Avg Slope Fallback)" 
            
            # Generate row for 0 to 12
            row_dict = {'Cohort': cohort}
            c_date = datetime.strptime(cohort, '%b %Y')
            
            for m_idx in range(13):
                # Is this Past (Actual) or Future (Predicted)?
                check_date = c_date + pd.DateOffset(months=m_idx)
                
                val = None
                
                # Check for Actual Value first
                if check_date <= sim_now:
                    # Try to fetch from Dataframe
                    if m_idx in arpu_df.columns:
                        try:
                            raw_val = arpu_df.at[cohort, m_idx]
                            # Check strictly for NaN (pandas nullable usually)
                            if pd.notna(raw_val):
                                val = float(raw_val)
                        except:
                            pass
                
                # If no actual value (future OR missing actual), use Model
                if val is None:
                    val = m * m_idx + c
                
                # Final Safety Check
                if pd.isna(val):
                    val = 0.0
                
                # Ensure non-negative
                val = max(0.0, float(val))
                row_dict[m_idx] = val
            
            # Add Formula Column
            row_dict['LTV Model (Cumulative)'] = f"{formula_str}"
            
            ltv_matrix_data.append(row_dict)
            
        ltv_df = pd.DataFrame(ltv_matrix_data).set_index("Cohort")
        # Ensure standard sort order
        ordered_ltv_labels = sorted(ltv_df.index, key=lambda x: datetime.strptime(x, '%b %Y'))
        ltv_df = ltv_df.reindex(ordered_ltv_labels)
        
        # Display as Table (Simple)
        st.subheader("LTV Matrix (Actuals vs Predictions)")
        st.caption("Actuals are standard text. Predictions are **blue** (Styling temporarily disabled).")

        # Basic Format
        numeric_cols = [c for c in ltv_df.columns if isinstance(c, int)]
        format_dict = {c: "€{:.2f}" for c in numeric_cols}

        # Use Standard Dataframe (No Styler) ensures compatibility
        # We apply formatting to the data itself or use st.column_config if needed, 
        # but to be safe we just format the values in a copy for display.
        
        display_df = ltv_df.copy()
        for c in numeric_cols:
             display_df[c] = display_df[c].apply(lambda x: f"€{x:.2f}")

        st.dataframe(display_df, use_container_width=True)

# --- TAB: RFM SCORE MODEL ---
else: 
    st.header("RFM Score Model Comparison")
    st.caption("Segments based on: **Recency** (Tertiles), **Frequency** (Custom Bins), **Monetary** (Spending Tier).")
    
    # Legend
    with st.expander("ℹ️  RFM Logic & Definitions", expanded=False):
        st.markdown("""
        **Recency (R)**: 
        - **3**: Recent (Top 33%)
        - **2**: Middle (Mid 33%)
        - **1**: Old (Bottom 33%) (Note: Based on relative distribution)

        **Frequency (F)**: 
        - **3**: High (Daily+, ≥ 20 visits)
        - **2**: Medium (Weekly, 4-19 visits)
        - **1**: Low (Monthly, 1-3 visits)

        **Monetary (M)**: 
        - **3**: High (> 100€)
        - **2**: Medium (1 - 100€)
        - **1**: Small (0.01 - 1€)
        - **0**: Non-Paying (0€)
        """)

    # --- 1. LOGIC: SNAPSHOT CALCULATION ---
    # function to calculate RFM table at a specific reference date
    def get_rfm_at_date(ref_date):
        # User Request: "Month Selection should be used for total users not the app setings"
        # We switch to using the FULL datasets (df_users, df_events) instead of filtered versions.
        # This ignores the Sidebar Cohort Filter and Sidebar Date Slider.
        
        ref_date_str = ref_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # We join df_users (All) with df_events (All, filtered by date <= ref_date)
        q = f"""
            SELECT 
                u.user_id, 
                strftime(DATE_TRUNC('month', u.acq_date), '%b %Y') as joined_month,
                strftime(DATE_TRUNC('month', u.acq_date), '%Y-%m') as sort_key,
                date_diff('day', MAX(CASE WHEN e.type = 'visit' THEN e.event_date END), timestamp '{ref_date_str}') as r_raw,
                COUNT(CASE WHEN e.type = 'visit' THEN 1 END) as f_raw,
                COALESCE(SUM(e.revenue), 0) as m_raw
            FROM df_users u 
            LEFT JOIN df_events e ON u.user_id = e.user_id AND e.event_date <= '{ref_date_str}'
            GROUP BY 1, 2, 3
        """
        df = duckdb.query(q).df()
        return df

    # Scoring Logic (Custom Bins)
    def calculate_rfm_scores(df):
        if df.empty: return df
        df = df.copy()
        # Handle NA for users with no events (Recency usually NaN) -> Fill with max? 
        # For simplicity, if r_raw is NaN (no visits), set to large number
        df['r_raw'] = df['r_raw'].fillna(9999)
        
        # R: Tertiles (Relative) - Staying with 3-2-1 logic
        try:
             # Invert QCut labels for Recency so 3 is "Low Recency Value" (Recent)
             # But pd.qcut sorts values. Low raw value = Recent.
             # qcut labels are applied to bins in increasing order. 
             # Bin 1 (Low values, Recent) -> Label "3"
             # Bin 3 (High values, Old) -> Label "1"
             df['R'] = pd.qcut(df['r_raw'], 3, labels=["3", "2", "1"], duplicates='drop').astype(str)
        except ValueError:
            df['R'] = "1"

        # F: Custom Bins (1-3 visits=1, 4-19=2, 20+=3)
        # Bins: [-1 (include 0?), 0.9, 3, 19, 99999]
        # Wait, user said "Monthly (1), Weekly (4), Daily".
        # Let's map: 0=0, 1-3=1 (Low), 4-19=2 (Med), 20+=3 (High)
        # Actually User said: F (hourly, daily, weekly, monthly). 
        # Impl: 1=Low, 2=Med, 3=High.
        f_bins = [-1, 0, 3, 19, 999999]
        f_labels = ["0", "1", "2", "3"] # 0 for no visits? (Though usually active have visits)
        df['F'] = pd.cut(df['f_raw'], bins=f_bins, labels=f_labels).astype(str)
        
        # M: Value Bins (0, 0.01-1, 1-100, >100)
        # Bins: [-1, 0, 1, 100, 999999]
        # Labels: 0 (0), 1 (0-1), 2 (1-100), 3 (>100)
        # Note: 0.0 value falls in [-1, 0]. 0.01 falls in (0, 1].
        m_bins = [-1, 0, 1, 100, 999999]
        m_labels = ["0", "1", "2", "3"]
        df['M'] = pd.cut(df['m_raw'], bins=m_bins, labels=m_labels).astype(str)
            
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
    st.markdown("### Month Selection")
    
    # Move Reference Month Selector to Top 
    # Default to last available month
    default_idx = len(available_months_str) - 1
    selected_month_str = st.selectbox("Select Analysis Month (Current):", options=available_months_str, index=default_idx)
    
    # Calculate Dates
    # Selected = Right Side (Current Month Status)
    # Snapshot for Right: End of Selected Month (or 1st of Next Month)
    current_month_date = datetime.strptime(selected_month_str, '%b %Y') 
    
    # Left Side = Previous Month 
    prev_month_date = current_month_date - pd.DateOffset(months=1)
    
    # Snapshots (Proxy: 1st of Month After)
    # Status of "May" is known at "June 1st"
    date_right = current_month_date + pd.DateOffset(months=1) # End of Current (Selected) Month
    date_left = current_month_date # End of Previous Month ( == Start of Current Month)
    
    # Recalculate RFM based on dynamic dates
    rfm_left_raw = get_rfm_at_date(date_left)
    rfm_right_raw = get_rfm_at_date(date_right)
    
    # --- STRICT MOM FLOW LOGIC ---
    valid_users = rfm_left_raw['user_id'].unique()
    rfm_right_raw = rfm_right_raw[rfm_right_raw['user_id'].isin(valid_users)]
    
    rfm_left = calculate_rfm_scores(rfm_left_raw)
    rfm_right = calculate_rfm_scores(rfm_right_raw)
    
    # --- 3. TOP SUMMARY ---
    # User Request: "Zeige bei RFM nur eine Zahl an (die Gesamtnutzer für den start monat) und den Filter."
       
    st.markdown("### Analysis Group Summary")
    
    all_groups = sorted(rfm_right['RFM_Group'].unique()) if not rfm_right.empty else []
    if all_groups:
         all_groups.insert(0, "All")
    
    c_sum, c_filt = st.columns([1, 1])
    
    with c_filt:
         selected_group = st.selectbox("Filter RFM Group:", options=all_groups, index=0)
    
    # Filter Logic
    mask_left = pd.Series(True, index=rfm_left.index)
    mask_right = pd.Series(True, index=rfm_right.index)
    
    if selected_group != "All":
        mask_left = mask_left & (rfm_left['RFM_Group'] == selected_group)
        mask_right = mask_right & (rfm_right['RFM_Group'] == selected_group)
        
    filtered_left = rfm_left[mask_left]
    filtered_right = rfm_right[mask_right]
    
    with c_sum:
        # User Request: Only 1 Number (Base Month)
        count_left = filtered_left['user_id'].nunique()
        label_left = prev_month_date.strftime('%B %Y')
        label_right = current_month_date.strftime('%B %Y') # Restore for later use
        st.metric(label=f"Total Users ({label_left})", value=count_left)
            
    st.markdown("---")

    # --- 4. DATA PREP FOR HEATMAP (DELTA) ---
    st.subheader(f"RFM Delta Heatmap ({label_left} ➜ {label_right})")
    
    # Heatmap shows GENERAL change (All Groups) for the selected Global Cohorts
    # User requested to remove the "Heatmap Filter" and use the data from above.
    
    # Calculate counts on the global set (rfm_left/rfm_right from Sidebar selection)
    hm_left = rfm_left.groupby(['F', 'R']).size().reset_index(name='count')
    hm_right = rfm_right.groupby(['F', 'R']).size().reset_index(name='count')
    
    hm_merged = pd.merge(hm_left, hm_right, on=['F', 'R'], how='outer', suffixes=('_left', '_right')).fillna(0)
    hm_merged['delta'] = hm_merged['count_right'] - hm_merged['count_left']
    
    # Pivot for Heatmap
    heatmap_pivot = hm_merged.pivot(index='R', columns='F', values='delta').fillna(0)
    
    # Sorting
    # Ensure all bins (0, 1, 2, 3) are present in columns
    heatmap_pivot = heatmap_pivot.reindex(index=["3", "2", "1"], columns=["0", "1", "2", "3"]).fillna(0)
    
    fig_rfm_heat = px.imshow(heatmap_pivot, text_auto='+d', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                             labels=dict(x="Frequency (0=Zero, 1=Low, 3=High)", y="Recency (3=Recent, 1=Old)", color="Change"),
                             template=chart_template)
    st.plotly_chart(fig_rfm_heat, use_container_width=True)

    # --- 5. GROUP COMPARISON (BAR CHART) ---
    st.subheader("Comparison of Selected RFM Groups")
    
    # Recalculate filtered set for Bar Chart based on Cohort Filter + RFM Group Filter
    # "Comparison of RFM combination should reflect the selected groups" -> Yes, Group Filter applies.
    # Note: rfm_right is already filtered by the GLOBAL Sidebar Cohort Filter.
    # We just need to apply the mask_right (which contains the RFM Group Filter logic).
    
    final_mask_right = mask_right 
    bar_data = rfm_right[final_mask_right]
    
    if bar_data.empty:
        st.info("No users in selected groups.")
    else:
        group_stats = bar_data.groupby(['joined_month', 'RFM_Group']).size().reset_index(name='count')
        group_stats = group_stats.sort_values(by="count", ascending=False)
        
        fig_groups = px.bar(group_stats, x="RFM_Group", y="count", color="joined_month", barmode="group",
                            template=chart_template, title=f"Composition ({label_right})")
        st.plotly_chart(fig_groups, use_container_width=True)

# --- 5. FOOTER AREA ---
st.markdown("---")
st.caption("Growth Intelligence Dashboard | © 2026 Volker Schulz | RFM MoM Analysis")