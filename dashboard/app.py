"""
Merchant Risk Command Center
Streamlit dashboard for fraud detection monitoring
"""
import sys
import os  
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Get the directory of the current file (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (where 'src' lives)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the root directory to Python's search path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.monitoring.case_manager import CaseManager, AlertMonitor

# Page configuration
st.set_page_config(
    page_title="Merchant Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    .risk-high {
        color: #dc2626;
        font-weight: bold;
    }
    .risk-medium {
        color: #d97706;
        font-weight: bold;
    }
    .risk-low {
        color: #059669;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data"""

    # LOAD RAW DATA
    merchants = pd.read_csv("data/raw/merchants.csv")
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=['timestamp'])

    # Ensure merchant_id is a string 
    merchants['merchant_id'] = merchants['merchant_id'].astype(str)
    
    # Load processed risk scores
    if os.path.exists("data/processed/merchant_risk_scores.csv"):
        risk_scores = pd.read_csv("data/processed/merchant_risk_scores.csv")
        risk_scores['merchant_id'] = risk_scores['merchant_id'].astype(str)

    # Remove 'decision' or 'risk_score' if they already exist in merchants.csv
    # This prevents the "decision_x / decision_y" naming conflict
        cols_to_drop = [c for c in ['decision', 'risk_score', 'risk_factors'] if c in merchants.columns]
        if cols_to_drop:
            merchants = merchants.drop(columns=cols_to_drop)

    # Merge only the specific columns we need to avoid clutter/conflicts
    # We use 'left' merge to keep all merchants even if they don't have a score yet
        merchants = merchants.merge(
            risk_scores[['merchant_id', 'risk_score', 'decision', 'risk_factors']], 
            on='merchant_id', 
            how='left'
        )

    # Safety check: if still missing (due to file not existing), then fallback
    if 'decision' not in merchants.columns:
        merchants['decision'] = 'REVIEW'
    
    # Fill actual empty rows resulting from the merge
    merchants['decision'] = merchants['decision'].fillna('REVIEW')
    merchants['risk_score'] = merchants['risk_score'].fillna(50.0)

    #  LOAD ML PREDICTIONS
    if os.path.exists("data/processed/ml_predictions.csv"):
        ml_preds = pd.read_csv("data/processed/ml_predictions.csv")
        
        # We only want the anomaly score from the ML step
        if 'ml_anomaly_score' in ml_preds.columns:
            merchants = merchants.merge(
                ml_preds[['merchant_id', 'ml_anomaly_score']], 
                on='merchant_id', 
                how='left'
            )
    
    # If any merge failed or columns were missing, we fill them here
    if 'decision' not in merchants.columns:
        merchants['decision'] = 'REVIEW'
    
    if 'risk_score' not in merchants.columns:
        merchants['risk_score'] = 0.0

    if 'risk_factors' not in merchants.columns:
        merchants['risk_factors'] = "No factors identified"

    if 'ml_anomaly_score' not in merchants.columns:
        # Default to risk_score if ML score isn't available yet
        merchants['ml_anomaly_score'] = merchants['risk_score']

    # Fill NaN values that occur when a merchant exists in raw but not in processed
    merchants['decision'] = merchants['decision'].fillna('REVIEW')
    merchants['risk_score'] = merchants['risk_score'].fillna(0.0)
    merchants['ml_anomaly_score'] = merchants['ml_anomaly_score'].fillna(0.0)

    return merchants, transactions

def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("🛡️ Merchant Risk Command Center")
        st.markdown("*Real-time fraud detection and risk monitoring*")

    with col2:
        st.metric(
            "System Status",
            "🟢 OPERATIONAL",
            delta="Last update: " + datetime.now().strftime("%H:%M:%S")
        )

    with col3:
        st.metric(
            "Data Freshness",
            "Live",
            delta="Processing 10K+ merchants"
        )

def render_kpis(merchants_df, transactions_df):
    """Render KPI metrics"""
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_merchants = len(merchants_df)
        st.metric(
            "Total Merchants",
            f"{total_merchants:,}",
            delta="+12 today"
        )

    with col2:
        fraud_rate = merchants_df['is_fraud'].mean()
        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2%}",
            delta="-0.3% vs yesterday",
            delta_color="inverse"
        )

    with col3:
        avg_risk = merchants_df['risk_score'].mean()
        st.metric(
            "Avg Risk Score",
            f"{avg_risk:.1f}",
            delta="Stable"
        )

    with col4:
        decline_rate = (merchants_df['decision'] == 'DECLINE').mean()
        st.metric(
            "Decline Rate",
            f"{decline_rate:.1%}",
            delta="+2.1%"
        )

    with col5:
        open_cases = len(merchants_df[merchants_df['decision'] == 'REVIEW'])
        st.metric(
            "Cases Pending",
            f"{open_cases}",
            delta="5 critical",
            delta_color="off"
        )

def render_risk_distribution(merchants_df):
    """Render risk score distribution"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")

        fig = px.histogram(
            merchants_df,
            x='risk_score',
            color='decision',
            nbins=50,
            title="Merchant Risk Score Distribution",
            labels={'risk_score': 'Risk Score', 'count': 'Number of Merchants'},
            color_discrete_map={
                'APPROVE': '#10b981',
                'REVIEW': '#f59e0b',
                'DECLINE': '#ef4444'
            }
        )

        # Add threshold lines
        fig.add_vline(x=30, line_dash="dash", line_color="green", 
                     annotation_text="Approve Threshold")
        fig.add_vline(x=60, line_dash="dash", line_color="orange",
                     annotation_text="Review Threshold")

        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Decision Breakdown")

        decision_counts = merchants_df['decision'].value_counts()

        fig = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="Merchant Decisions",
            color=decision_counts.index,
            color_discrete_map={
                'APPROVE': '#10b981',
                'REVIEW': '#f59e0b',
                'DECLINE': '#ef4444'
            }
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

def render_fraud_analysis(merchants_df, transactions_df):
    """Render fraud analysis section"""
    st.markdown("---")
    st.subheader("🔍 Fraud Detection Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Fraud by industry
        fraud_by_industry = merchants_df.groupby('industry').agg({
            'is_fraud': ['sum', 'count', 'mean']
        }).reset_index()
        fraud_by_industry.columns = ['industry', 'fraud_count', 'total', 'fraud_rate']
        fraud_by_industry = fraud_by_industry.sort_values('fraud_rate', ascending=True)

        fig = px.bar(
            fraud_by_industry,
            x='fraud_rate',
            y='industry',
            orientation='h',
            title="Fraud Rate by Industry",
            labels={'fraud_rate': 'Fraud Rate', 'industry': 'Industry'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch')

    with col2:
        # ML Model Performance
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(merchants_df['is_fraud'], merchants_df['ml_anomaly_score'])
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            mode='lines',
            line=dict(color='#3b82f6', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title="ML Model ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350
        )
        st.plotly_chart(fig, width='stretch')

    with col3:
        # Risk factors breakdown
        all_factors = []
        for factors in merchants_df['risk_factors'].dropna():
            all_factors.extend([f.strip() for f in factors.split(';')])

        factor_counts = pd.Series(all_factors).value_counts().head(8)

        fig = px.bar(
            x=factor_counts.values,
            y=factor_counts.index,
            orientation='h',
            title="Top Risk Factors",
            labels={'x': 'Frequency', 'y': 'Risk Factor'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch')

def render_transaction_monitoring(transactions_df):
    """Render real-time transaction monitoring"""
    st.markdown("---")
    st.subheader("📊 Transaction Monitoring")

    # Time series of transactions
    transactions_df['hour'] = transactions_df['timestamp'].dt.floor('h')
    hourly_stats = transactions_df.groupby('hour').agg({
        'amount': ['count', 'sum'],
        'is_fraud': 'sum'
    }).reset_index()
    hourly_stats.columns = ['hour', 'txn_count', 'txn_volume', 'fraud_count']
    hourly_stats['fraud_rate'] = hourly_stats['fraud_count'] / hourly_stats['txn_count']

    col1, col2 = st.columns(2)

    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['txn_count'],
                name="Transaction Count",
                mode='lines',
                line=dict(color='#3b82f6')
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['fraud_rate'] * 100,
                name="Fraud Rate (%)",
                mode='lines',
                line=dict(color='#ef4444')
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Transaction Volume & Fraud Rate (Last 24h)",
            height=400
        )
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)

        st.plotly_chart(fig, width='stretch')

    with col2:
        # Amount distribution
        fig = px.box(
            transactions_df,
            x='is_fraud',
            y='amount',
            title="Transaction Amount by Fraud Status",
            labels={'is_fraud': 'Fraudulent', 'amount': 'Amount ($)'},
            color='is_fraud',
            color_discrete_map={False: '#10b981', True: '#ef4444'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')

def render_merchant_queue(merchants_df, monitor, transactions, cm):
    st.markdown("---")  

    # PREPARE DATA (Crucial to avoid NameError)
    open_count = len(merchants_df[merchants_df['decision'] == 'REVIEW'])
    all_alerts = monitor.check_fraud_rate(transactions) 
    unread_alerts = [a for a in all_alerts if st.session_state.get('last_cleared_time') is None 
                     or a['timestamp'] > st.session_state['last_cleared_time']]
    unread_count = len(unread_alerts)
    # Header Layout: Title gets most space, then two slots for the indicators
    head_col, case_col, alert_col = st.columns([3, 0.8, 1])
    with head_col:
        st.subheader("⚠️ Priority Review Queue")   
    with case_col:
            # Added padding-top to align it vertically with the subheader text.
            st.markdown(f"<div style='padding-top: 12px; font-size: 16px; font-weight: 500;'>📂 Open Cases: {open_count}</div>", unsafe_allow_html=True)    
    with alert_col:
        # We can use a button here so it's clickable right next to the header
        if st.button(f"🔔 {unread_count} Alerts", help="View System Spikes"):
            show_alert_modal(unread_alerts)

    # Filter high-risk merchants
    review_queue = merchants_df[
        (merchants_df['decision'].isin(['REVIEW', 'DECLINE'])) |
        (merchants_df['risk_score'] >= 70)
    ].sort_values('risk_score', ascending=False).head(20)

    # Format for display
    display_df = review_queue[[
        'merchant_id', 'business_name', 'industry', 'risk_score', 
        'decision', 'ml_anomaly_score', 'fraud_type'
    ]].copy()

    display_df['risk_score'] = display_df['risk_score'].round(1)
    display_df['ml_anomaly_score'] = display_df['ml_anomaly_score'].round(1)

    # Color code risk scores
    def color_risk(val):
        if val >= 75:
            return 'background-color: #fecaca; color: #991b1b'
        elif val >= 60:
            return 'background-color: #fed7aa; color: #9a3412'
        else:
            return 'background-color: #bbf7d0; color: #166534'

    styled_df = display_df.style.applymap(
        color_risk, subset=['risk_score', 'ml_anomaly_score']
    )

    st.dataframe(styled_df, width='stretch', height=400)

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Review Top Case"):
            #st.info("Opening case management interface...")
            # Set a flag to tell the app to show a case view
            st.session_state['view_case'] = True
            # Grab the ID of the highest risk merchant
            top_id = review_queue.iloc[0]['merchant_id']
            st.session_state['selected_merchant'] = top_id
   
    with col2:
        if st.button("✅ Bulk Approve Low Risk"):
            # Identify low-risk merchants to approve
            to_approve = merchants_df[merchants_df['decision'] == 'APPROVE']
            
            if not to_approve.empty:
                open_cases = cm.cases.values() 
                
                for _, merchant in to_approve.iterrows():
                    m_id = merchant['merchant_id']
                    # 2. Find the Case ID associated with this Merchant
                    case_to_close = next((c['case_id'] for c in open_cases if c['merchant_id'] == m_id and c['status'] == 'OPEN'), None)
                    
                    if case_to_close:
                        # 3. Use the CORRECT method name: resolve_case
                        cm.resolve_case(case_id=case_to_close, resolution='APPROVED', analyst_id='SYSTEM_AUTO')
                
                #st.success(f"Approved {len(to_approve)} merchants")
            
            # Use the session_state version specifically
            # df = st.session_state.merchants
            # to_approve_ids = df[df['decision'] == 'APPROVE']['merchant_id'].tolist()
            
            # if to_approve_ids:
            #     # 1. Update the Data (This makes 'Cases Pending' go down)
            #     st.session_state.merchants.loc[
            #         st.session_state.merchants['merchant_id'].isin(to_approve_ids), 'decision'
            #     ] = 'APPROVED_DONE'
                
            #     # 2. Update the CaseManager (This makes 'Resolution Time' update)
            #     for m_id in to_approve_ids:
            #         case_id = next((k for k, v in cm.cases.items() if v['merchant_id'] == m_id), None)
            #         if case_id:
            #             cm.resolve_case(case_id=case_id, resolution='APPROVED', analyst_id='SYSTEM')
                
            #     st.success(f"Approved {len(to_approve_ids)} merchants.")
            #     st.rerun()

                # Save the list to session state
                st.session_state['approved_list'] = to_approve[['merchant_id', 'business_name', 'risk_score']].to_dict('records')
                st.session_state['show_approved_view'] = True
                st.success(f"Successfully auto-approved {len(to_approve)} low-risk merchants")
                
            else:
                st.warning("No merchants currently meet the 'APPROVE' criteria.")
  
    with col3:
        if st.button("🚫 Block High Risk"):
            # Identify high-risk merchants to block
            to_block = merchants_df[merchants_df['decision'] == 'DECLINE']
            
            if not to_block.empty:
                open_cases = cm.cases.values()
                for _, merchant in to_block.iterrows():
                    m_id = merchant['merchant_id']
                    case_to_close = next((c['case_id'] for c in open_cases if c['merchant_id'] == m_id and c['status'] == 'OPEN'), None)
                    
                    if case_to_close:
                        cm.resolve_case(case_id=case_to_close, resolution='BLOCKED', analyst_id='SYSTEM_AUTO')
                
                #st.error(f"Blocked {len(to_block)} merchants")
                #st.rerun()
                st.session_state['blocked_list'] = to_block[['merchant_id', 'business_name', 'risk_score']].to_dict('records')
                st.session_state['show_blocked_view'] = True
                st.error(f"Successfully blocked {len(to_block)} high-risk merchants")
            else:
                st.warning("No merchants currently meet the 'DECLINE' criteria.")

def render_model_performance(merchants_df):
    """Render model performance metrics"""
    st.markdown("---")
    st.subheader("🤖 Model Performance")

    from sklearn.metrics import classification_report, confusion_matrix

    col1, col2, col3 = st.columns(3)

    with col1:
        # Confusion matrix
        y_true = merchants_df['is_fraud']
        y_pred = merchants_df['ml_anomaly_score'] > 70  # Threshold

        cm = confusion_matrix(y_true, y_pred)

        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            title="Confusion Matrix",
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Metrics
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'Specificity', 'F1-Score'],
            'Value': [precision, recall, specificity, f1],
            'Target': [0.80, 0.85, 0.90, 0.82]
        })

        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title="Model Metrics vs Targets",
            labels={'Value': 'Score'},
            color='Value',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Target")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

    with col3:
        # Score calibration
        score_bins = pd.cut(merchants_df['ml_anomaly_score'], bins=10)
        calibration = merchants_df.groupby(score_bins)['is_fraud'].agg(['mean', 'count'])
        calibration['bin_midpoint'] = [interval.mid for interval in calibration.index]

        fig = px.scatter(
            calibration,
            x='bin_midpoint',
            y='mean',
            size='count',
            title="Model Calibration",
            labels={
                'bin_midpoint': 'Predicted Score',
                'mean': 'Actual Fraud Rate'
            }
        )
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Perfect Calibration'
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

@st.dialog("🔔 System Alerts & Notifications")
def show_alert_modal(unread_alerts):
    """This creates the popup window"""
    if not unread_alerts:
        st.success("All caught up! No unread alerts.")
    else:
        st.write(f"You have **{len(unread_alerts)}** unread system spikes.")
        
        # Action Buttons inside the popup
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Mark All as Read"):
                st.session_state['last_cleared_time'] = datetime.now()
                st.rerun()
        # with c2:
        #     st.button("Close")
        st.divider()
        
        # Scrollable log inside the popup
        with st.container(height=300):
            log_html = ""
            for alert in reversed(unread_alerts):
                time = alert['timestamp'].strftime('%H:%M')
                log_html += f"**{time}** | {alert['message']} (`{alert['severity']}`)\n\n"
            st.markdown(log_html)

def main():
    """Main dashboard function"""

    if 'alerts_cleared' not in st.session_state:
        st.session_state['alerts_cleared'] = False

    if 'last_cleared_time' not in st.session_state:
        st.session_state['last_cleared_time'] = None

    if 'show_log' not in st.session_state:
        st.session_state['show_log'] = True

    if 'view_case' not in st.session_state:
        st.session_state['view_case'] = False
    
    # Load data
    try:
        merchants, transactions = load_data()

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the data generation and model training scripts first.")
        return
    
    # Initialize the Engines
    cm = CaseManager()
    monitor = AlertMonitor()

    # if 'merchants' not in st.session_state:
    #     st.session_state.merchants = merchants.copy()
    #     # This is also where you'd link initial cases to the CaseManager
    #     cm.auto_create_cases(st.session_state.merchants)
        
    # --- SIDEBAR: SYSTEM HEALTH & DRIFT ---
    with st.sidebar:
        st.header("🛠️ System Health")
        
        # Calculate Model Drift (PSI)
        # We compare original risk scores vs. current ones
        psi_value = monitor.calculate_psi(merchants['risk_score'], merchants['risk_score'] * 1.05) 
        
        st.metric("Model Drift (PSI)", f"{psi_value:.3f}", 
                  delta="Stable" if psi_value < 0.1 else "Drifting", 
                  delta_color="normal" if psi_value < 0.1 else "inverse")
        
        # Performance Metrics from CaseManager
        # metrics = cm.get_metrics()
        # st.write("---")
        # st.subheader("Efficiency")
        # st.metric("Avg Resolution Time", f"{metrics.get('avg_resolution_time', 0):.1f}s")

    # --- DATA DEBUGGER SIDEBAR ---
    #with st.sidebar:
        st.write("---")
        st.header("🛠️ Data Debugger")
        if st.checkbox("Show Data Integrity Check"):
            st.write("**Columns Found:**")
            st.write(list(merchants.columns))
            
            # Check for critical columns
            crit_cols = ['risk_score', 'decision', 'ml_anomaly_score', 'is_fraud']
            for col in crit_cols:
                if col in merchants.columns:
                    st.success(f"✅ {col}")
                else:
                    st.error(f"❌ {col} MISSING")
            
            st.write("**Data Preview:**")
            st.dataframe(merchants.head(5))
    
    # Render sections
    render_header()  
    render_kpis(merchants, transactions)
    render_risk_distribution(merchants)
    render_fraud_analysis(merchants, transactions)
    render_transaction_monitoring(transactions)
    render_merchant_queue(merchants, monitor, transactions, cm)

    # REVIEW TOP CASE VIEW 
    # This checks if the button was clicked and shows details at the bottom
    if st.session_state.get('view_case'):

        st.markdown("---")
        st.header(f"🕵️ Investigation: {st.session_state.get('selected_merchant', 'Unknown')}")

        merchant_id = st.session_state['selected_merchant']
        score = merchants[merchants['merchant_id'] == merchant_id]['risk_score'].values[0]
        
        # Use the CaseManager's priority logic
        priority = cm._calculate_priority(score)
        
        st.subheader(f"🕵️ Investigation: {merchant_id}")
        
        # Display Priority Badge
        if priority == 'CRITICAL':
            st.error(f"PRIORITY: {priority}")
        elif priority == 'HIGH':
            st.warning(f"PRIORITY: {priority}")
        else:
            st.info(f"PRIORITY: {priority}")

        # --- ANALYST ASSIGNMENT ---
        col_a, col_b = st.columns(2)
        with col_a:
            analyst = st.selectbox("Assign to Analyst", ["Unassigned", "Irene", "Senior Investigator"])
        with col_b:
            resolution = st.selectbox("Proposed Action", ["No Action", "Flag for Follow-up", "Verify Identity"])
        
        if st.button("Update Case File"):
            # This calls your assign_case function!
            cm.assign_case(merchant_id, analyst)
            st.success(f"Case updated and assigned to {analyst}")
      
        # Pull data for just this one merchant
        merchant_id = st.session_state['selected_merchant']
        details = merchants[merchants['merchant_id'] == merchant_id].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Business:** {details['business_name']}")
            st.write(f"**Industry:** {details['industry']}")
            st.write(f"**Risk Score:** {details['risk_score']}")
        with col2:
            st.write(f"**Decision:** {details['decision']}")
            st.write(f"**Factors:** {details['risk_factors']}")
            
        if st.button("Close Case View"):
            st.session_state['view_case'] = False
            st.rerun()

    # PPROVED MERCHANTS VIEW 
    if st.session_state.get('show_approved_view'):
        st.markdown("---")
        with st.container():
            st.subheader("✅ Bulk Approval Log")
            
            approved_df = pd.DataFrame(st.session_state['approved_list'])
            st.dataframe(approved_df, width='stretch', hide_index=True)
            
            # --- ADD DOWNLOAD BUTTON HERE ---
            csv = approved_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Approval List (CSV)",
                data=csv,
                file_name=f"approved_merchants_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            if st.button("Dismiss Approval Log"):
                st.session_state['show_approved_view'] = False
                st.rerun()

    if st.session_state.get('show_blocked_view'):
        st.markdown("---")
        with st.container():
            st.subheader("🚫 Blocked Merchants Log")
            
            blocked_df = pd.DataFrame(st.session_state['blocked_list'])
            st.table(blocked_df)
            
            # --- ADD DOWNLOAD BUTTON HERE ---
            csv_blocked = blocked_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Block List (CSV)",
                data=csv_blocked,
                file_name=f"blocked_merchants_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            # --------------------------------

            if st.button("Dismiss Log"):
                st.session_state['show_blocked_view'] = False
                st.rerun()

    # Render performance section
    render_model_performance(merchants)

    # Footer
    st.markdown("---")
    st.caption("Merchant Fraud Detection Simulator v1.0 | Built by Irene A. Gyebi")

if __name__ == "__main__":
    main()
