import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ===============================================
# PAGE CONFIG & TACTICAL UI
# ===============================================
st.set_page_config(
    page_title="VANGUARD | Military Health System",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Tactical CSS
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Tactical Header */
    .main-header {
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #1f4788 0%, #000 100%);
        padding: 15px;
        border-left: 5px solid #00d4ff;
        border-radius: 5px;
        margin-bottom: 25px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }

    /* Status Cards */
    .status-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        background: rgba(255, 255, 255, 0.05);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .fit-glow { color: #00ff88; text-shadow: 0 0 10px #00ff88; font-weight: bold; font-size: 24px; }
    .monitor-glow { color: #ffcc00; text-shadow: 0 0 10px #ffcc00; font-weight: bold; font-size: 24px; }
    .risk-glow { color: #ff4b4b; text-shadow: 0 0 10px #ff4b4b; font-weight: bold; font-size: 24px; }

    /* Metric Styling */
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; color: #00d4ff; }
    
    /* Alert Box */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
        background: rgba(255, 255, 255, 0.05);
    }
    
    .alert-critical { border-color: #ff4b4b; }
    .alert-warning { border-color: #ffcc00; }
    .alert-normal { border-color: #00ff88; }
</style>
""", unsafe_allow_html=True)

# ===============================================
# DATA & MODEL ENGINE
# ===============================================

@st.cache_data
def load_and_train_models():
    # Load data
    try:
        df = pd.read_csv("military_wearable_synthetic_500_rows.csv")
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'military_wearable_synthetic_500_rows.csv' exists.")
        st.stop()
    
    # Normalize fatigue and handle encoding
    if df['fatigue_index'].max() > 1:
        df['fatigue_index'] = df['fatigue_index'] / 100.0
    
    phase_le = LabelEncoder()
    df['operational_phase_encoded'] = phase_le.fit_transform(df['operational_phase'].str.strip().str.lower())
    
    # Logic-based Risk Labelling (Ground Truth for training)
    def get_risk_label(row):
        score = 100 - (row['fatigue_index'] * 40) - (row['stress_level_0_100'] * 0.3)
        score -= row['sleep_debt_hours'] * 1.5
        score -= max(0, row['core_body_temp_c'] - 37.5) * 10
        if score >= 70: return "FIT"
        elif score >= 40: return "MONITOR"
        else: return "HIGH_RISK"
    
    df['risk_status'] = df.apply(get_risk_label, axis=1)
    risk_le = LabelEncoder()
    df['risk_label'] = risk_le.fit_transform(df['risk_status'])
    
    features = [
        'operational_phase_encoded', 'heart_rate_bpm', 'core_body_temp_c', 
        'hydration_percent', 'fatigue_index', 'spo2', 'respiration_rate', 
        'movement_intensity_g', 'blood_pressure_sys_mmHg', 
        'blood_pressure_dia_mmHg', 'stress_level_0_100', 
        'energy_expenditure_kcal_hr', 'sleep_debt_hours', 'ambient_temp_c'
    ]
    
    X = df[features]
    y = df['risk_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    return df, rf_model, gb_model, scaler, phase_le, risk_le, features, X_test_scaled, y_test

def detect_anomalies(df):
    """Flags physiological outliers (vitals 2SD from mean)"""
    hr_mean, hr_std = df['heart_rate_bpm'].mean(), df['heart_rate_bpm'].std()
    df['is_anomaly'] = (df['heart_rate_bpm'] > hr_mean + 2*hr_std) | (df['core_body_temp_c'] > 38.8)
    return df

def predict_recovery_time(risk_status, fatigue, stress, sleep_debt, heart_rate):
    base_recovery = {"FIT": 2, "MONITOR": 12, "HIGH_RISK": 48}
    recovery_hours = base_recovery[risk_status] + (fatigue * 10) + (stress / 10) + (sleep_debt * 2)
    return int(recovery_hours)

def calculate_readiness_score(row):
    """Calculate overall readiness score (0-100)"""
    score = 100
    score -= row['fatigue_index'] * 30
    score -= (row['stress_level_0_100'] / 100) * 20
    score -= row['sleep_debt_hours'] * 2
    score -= max(0, row['core_body_temp_c'] - 37.5) * 8
    score -= max(0, 100 - row['hydration_percent']) * 0.3
    score -= max(0, row['heart_rate_bpm'] - 100) * 0.2
    return max(0, min(100, score))

# ===============================================
# INITIALIZATION
# ===============================================

df, rf_model, gb_model, scaler, phase_le, risk_le, features, X_test_scaled, y_test = load_and_train_models()
df = detect_anomalies(df)
df['readiness_score'] = df.apply(calculate_readiness_score, axis=1)

# ===============================================
# SIDEBAR
# ===============================================

st.sidebar.markdown("<h2 style='text-align: center; color: #00d4ff;'>⚡ VANGUARD C2</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("COMMAND MENU", [
    "🎯 DASHBOARD", 
    "🔬 STATUS PREDICTOR", 
    "📊 UNIT DATABASE", 
    "📈 INTEL & ANALYTICS",
    "⚠️ THREAT MONITOR",
    "🌡️ ENVIRONMENTAL"
])
st.sidebar.markdown("---")

# Real-time stats
st.sidebar.success("📡 BIOMETRIC FEED: ACTIVE")
st.sidebar.info(f"👥 UNIT SIZE: {len(df)} PAX")
st.sidebar.metric("⚡ Avg Readiness", f"{df['readiness_score'].mean():.1f}%")
st.sidebar.metric("🔴 Critical Alerts", len(df[df['risk_status'] == 'HIGH_RISK']))

# Time simulation
st.sidebar.markdown("---")
st.sidebar.markdown("**⏱️ MISSION CLOCK**")
mission_time = st.sidebar.slider("Hours Elapsed", 0, 72, 24)

# ===============================================
# PAGE 1: ENHANCED DASHBOARD
# ===============================================

if page == "🎯 DASHBOARD":
    st.markdown('<div class="main-header"><h1>⚔️ UNIT READINESS HUD</h1></div>', unsafe_allow_html=True)
    
    # KPI Metrics Row 1
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("TOTAL STRENGTH", f"{len(df)}", "ACTIVE")
    with m2:
        fit_p = (len(df[df['risk_status'] == 'FIT']) / len(df)) * 100
        st.metric("COMBAT READY", f"{fit_p:.1f}%", f"{fit_p-50:.1f}%")
    with m3:
        anomalies = df['is_anomaly'].sum()
        st.metric("PHYSIO ANOMALIES", f"{anomalies}", delta_color="inverse")
    with m4:
        avg_fatigue = df['fatigue_index'].mean()
        st.metric("AVG UNIT FATIGUE", f"{avg_fatigue:.2f}", f"{0.3-avg_fatigue:.2f}")
    with m5:
        avg_readiness = df['readiness_score'].mean()
        st.metric("READINESS INDEX", f"{avg_readiness:.1f}", f"{avg_readiness-70:.1f}")

    st.markdown("---")
    
    # Main visualization area
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("🛰️ 3D Bio-Cluster Analysis")
        fig_3d = px.scatter_3d(df, x='heart_rate_bpm', y='stress_level_0_100', z='fatigue_index',
                               color='risk_status', template="plotly_dark",
                               color_discrete_map={'FIT': '#00ff88', 'MONITOR': '#ffcc00', 'HIGH_RISK': '#ff4b4b'},
                               opacity=0.8, size='readiness_score', hover_data=['hydration_percent', 'sleep_debt_hours'])
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=450)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_r:
        st.subheader("🚨 Priority Alerts")
        criticals = df[df['risk_status'] == 'HIGH_RISK'].head(5)
        if not criticals.empty:
            for idx, row in criticals.iterrows():
                st.markdown(f"""
                <div class="alert-box alert-critical">
                    <strong>PAX ID {idx}: CRITICAL</strong><br>
                    HR: {row['heart_rate_bpm']:.0f} bpm | Temp: {row['core_body_temp_c']:.1f}°C<br>
                    Fatigue: {row['fatigue_index']:.2f} | Readiness: {row['readiness_score']:.0f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No High-Risk alerts in current cycle.")
    
    st.markdown("---")
    
    # Distribution Charts
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("📊 Risk Distribution")
        risk_counts = df['risk_status'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=['#00ff88', '#ffcc00', '#ff4b4b']),
            hole=0.4
        )])
        fig_pie.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        st.subheader("💓 Heart Rate Distribution")
        fig_hr = px.histogram(df, x='heart_rate_bpm', nbins=30, template="plotly_dark",
                             color='risk_status', 
                             color_discrete_map={'FIT': '#00ff88', 'MONITOR': '#ffcc00', 'HIGH_RISK': '#ff4b4b'})
        fig_hr.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_hr, use_container_width=True)
    
    with c3:
        st.subheader("😴 Sleep Debt Analysis")
        fig_sleep = px.box(df, y='sleep_debt_hours', x='risk_status', template="plotly_dark",
                          color='risk_status',
                          color_discrete_map={'FIT': '#00ff88', 'MONITOR': '#ffcc00', 'HIGH_RISK': '#ff4b4b'})
        fig_sleep.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_sleep, use_container_width=True)
    
    st.markdown("---")
    
    # Operational Phase Analysis
    st.subheader("🎖️ Readiness by Operational Phase")
    phase_analysis = df.groupby('operational_phase').agg({
        'readiness_score': 'mean',
        'heart_rate_bpm': 'mean',
        'fatigue_index': 'mean',
        'stress_level_0_100': 'mean'
    }).reset_index()
    
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Bar(name='Readiness Score', x=phase_analysis['operational_phase'], 
                               y=phase_analysis['readiness_score'], marker_color='#00d4ff'))
    fig_phase.add_trace(go.Scatter(name='Avg Heart Rate', x=phase_analysis['operational_phase'], 
                                   y=phase_analysis['heart_rate_bpm'], yaxis='y2', marker_color='#ff4b4b'))
    fig_phase.update_layout(
        template="plotly_dark",
        yaxis=dict(title='Readiness Score'),
        yaxis2=dict(title='Heart Rate (bpm)', overlaying='y', side='right'),
        height=400
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# ===============================================
# PAGE 2: ENHANCED STATUS PREDICTOR
# ===============================================

elif page == "🔬 STATUS PREDICTOR":
    st.markdown('<div class="main-header"><h1>🔬 SOLDIER DIAGNOSTICS & PREDICTION</h1></div>', unsafe_allow_html=True)
    
    with st.form("diagnostic_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("💓 Vitals")
            hr = st.slider("HR (BPM)", 40, 180, 70)
            temp = st.slider("Core Temp (°C)", 35.0, 42.0, 37.0)
            spo2 = st.slider("SpO2 %", 85, 100, 98)
            resp = st.slider("Respiration Rate", 10, 40, 16)
            sys_bp = st.slider("Systolic BP", 80, 200, 120)
            dia_bp = st.slider("Diastolic BP", 50, 130, 80)
            
        with c2:
            st.subheader("⚙️ Operational")
            phase = st.selectbox("Phase", ["combat", "training", "recovery", "patrol", "rest"])
            fatigue = st.slider("Fatigue Index", 0.0, 1.0, 0.2)
            stress = st.slider("Stress (0-100)", 0, 100, 20)
            movement = st.slider("Movement Intensity (g)", 0.0, 3.0, 1.0)
            energy = st.slider("Energy Expenditure (kcal/hr)", 50, 800, 150)
            
        with c3:
            st.subheader("🎒 Logistics")
            hydra = st.slider("Hydration %", 0, 100, 80)
            sleep = st.slider("Sleep Debt (Hrs)", 0.0, 15.0, 2.0)
            ambient = st.number_input("Ambient Temp (°C)", value=25)
            
        submit = st.form_submit_button("🔍 RUN DIAGNOSTIC", use_container_width=True)

    if submit:
        # Predict
        phase_enc = phase_le.transform([phase.lower()])[0]
        input_data = np.array([[phase_enc, hr, temp, hydra, fatigue, spo2, resp, movement, 
                               sys_bp, dia_bp, stress, energy, sleep, ambient]])
        input_scaled = scaler.transform(input_data)
        
        res_idx = rf_model.predict(input_scaled)[0]
        res_label = risk_le.inverse_transform([res_idx])[0]
        proba = rf_model.predict_proba(input_scaled)[0]
        
        # Calculate readiness
        readiness = 100 - (fatigue * 30) - (stress / 5) - (sleep * 2) - max(0, temp - 37.5) * 8
        readiness = max(0, min(100, readiness))
        
        # UI Output
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if res_label == "FIT":
                st.markdown(f'<div class="status-card"><span class="fit-glow">STATUS: {res_label}</span></div>', unsafe_allow_html=True)
            elif res_label == "MONITOR":
                st.markdown(f'<div class="status-card"><span class="monitor-glow">STATUS: {res_label}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-card"><span class="risk-glow">STATUS: {res_label}</span></div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Readiness Score", f"{readiness:.1f}%", f"{readiness-70:.1f}%")
            
        with col3:
            recovery = predict_recovery_time(res_label, fatigue, stress, sleep, hr)
            st.metric("Est. Recovery Time", f"{recovery}h", delta_color="inverse")
        
        # Prediction Confidence
        st.subheader("📊 Model Confidence")
        conf_df = pd.DataFrame({
            'Status': risk_le.classes_,
            'Probability': proba
        })
        fig_conf = px.bar(conf_df, x='Status', y='Probability', template="plotly_dark",
                         color='Probability', color_continuous_scale='RdYlGn')
        fig_conf.update_layout(height=300)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Gauges for key metrics
        st.subheader("🎯 Vital Signs Dashboard")
        gauge_col1, gauge_col2, gauge_col3, gauge_col4 = st.columns(4)
        
        with gauge_col1:
            fig_hr_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hr,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Rate"},
                gauge={'axis': {'range': [40, 180]},
                      'bar': {'color': "#00d4ff"},
                      'steps': [{'range': [40, 100], 'color': '#00ff88'},
                               {'range': [100, 140], 'color': '#ffcc00'},
                               {'range': [140, 180], 'color': '#ff4b4b'}]}
            ))
            fig_hr_gauge.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_hr_gauge, use_container_width=True)
        
        with gauge_col2:
            fig_temp_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=temp,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Temperature °C"},
                gauge={'axis': {'range': [35, 42]},
                      'bar': {'color': "#00d4ff"},
                      'steps': [{'range': [35, 37.5], 'color': '#00ff88'},
                               {'range': [37.5, 38.5], 'color': '#ffcc00'},
                               {'range': [38.5, 42], 'color': '#ff4b4b'}]}
            ))
            fig_temp_gauge.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_temp_gauge, use_container_width=True)
        
        with gauge_col3:
            fig_hydra_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hydra,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Hydration %"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#00d4ff"},
                      'steps': [{'range': [0, 60], 'color': '#ff4b4b'},
                               {'range': [60, 80], 'color': '#ffcc00'},
                               {'range': [80, 100], 'color': '#00ff88'}]}
            ))
            fig_hydra_gauge.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_hydra_gauge, use_container_width=True)
        
        with gauge_col4:
            fig_stress_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Stress Level"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#00d4ff"},
                      'steps': [{'range': [0, 40], 'color': '#00ff88'},
                               {'range': [40, 70], 'color': '#ffcc00'},
                               {'range': [70, 100], 'color': '#ff4b4b'}]}
            ))
            fig_stress_gauge.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_stress_gauge, use_container_width=True)
        
        # Mission Simulation
        st.markdown("---")
        st.subheader("🛠️ Mission Duration Simulator")
        sim_col1, sim_col2 = st.columns([3, 1])
        
        with sim_col1:
            extra_fatigue = st.slider("Mission Time (Hours) - Projects Fatigue Increase", 0, 24, 0)
        with sim_col2:
            sim_button = st.button("Run Simulation")
            
        if extra_fatigue > 0 or sim_button:
            fatigue_increase = extra_fatigue * 0.02
            stress_increase = extra_fatigue * 1.5
            sleep_increase = extra_fatigue * 0.5
            
            sim_data = input_data.copy()
            sim_data[0][4] = min(1.0, sim_data[0][4] + fatigue_increase)
            sim_data[0][10] = min(100, sim_data[0][10] + stress_increase)
            sim_data[0][12] = min(15, sim_data[0][12] + sleep_increase)
            
            sim_res_idx = rf_model.predict(scaler.transform(sim_data))[0]
            sim_res = risk_le.inverse_transform([sim_res_idx])[0]
            
            sim_readiness = 100 - (sim_data[0][4] * 30) - (sim_data[0][10] / 5) - (sim_data[0][12] * 2)
            sim_readiness = max(0, min(100, sim_readiness))
            
            st.warning(f"⚠️ **After {extra_fatigue}h mission:** Status = **{sim_res}** | Readiness = **{sim_readiness:.1f}%**")
            
            # Comparison chart
            comparison = pd.DataFrame({
                'Metric': ['Fatigue', 'Stress', 'Sleep Debt'],
                'Current': [fatigue, stress/100, sleep],
                'Projected': [sim_data[0][4], sim_data[0][10]/100, sim_data[0][12]]
            })
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Current', x=comparison['Metric'], y=comparison['Current'], marker_color='#00d4ff'))
            fig_comp.add_trace(go.Bar(name='Projected', x=comparison['Metric'], y=comparison['Projected'], marker_color='#ff4b4b'))
            fig_comp.update_layout(template="plotly_dark", height=300, barmode='group')
            st.plotly_chart(fig_comp, use_container_width=True)

# ===============================================
# PAGE 3: ENHANCED UNIT DATABASE
# ===============================================

elif page == "📊 UNIT DATABASE":
    st.markdown('<div class="main-header"><h1>📋 UNIT ROSTER & ANALYTICS</h1></div>', unsafe_allow_html=True)
    
    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        f_risk = st.multiselect("Filter Risk Status", ["FIT", "MONITOR", "HIGH_RISK"], 
                               default=["FIT", "MONITOR", "HIGH_RISK"])
    with f_col2:
        f_phase = st.multiselect("Filter Operational Phase", 
                                df['operational_phase'].unique().tolist(),
                                default=df['operational_phase'].unique().tolist())
    with f_col3:
        readiness_threshold = st.slider("Min Readiness Score", 0, 100, 0)
    
    filtered = df[(df['risk_status'].isin(f_risk)) & 
                  (df['operational_phase'].isin(f_phase)) &
                  (df['readiness_score'] >= readiness_threshold)]
    
    st.info(f"📊 Showing {len(filtered)} / {len(df)} personnel")
    
    # Summary stats for filtered data
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Avg Readiness", f"{filtered['readiness_score'].mean():.1f}%")
    with stat_col2:
        st.metric("Avg Heart Rate", f"{filtered['heart_rate_bpm'].mean():.0f} bpm")
    with stat_col3:
        st.metric("Avg Fatigue", f"{filtered['fatigue_index'].mean():.2f}")
    with stat_col4:
        st.metric("Anomalies", f"{filtered['is_anomaly'].sum()}")
    
    # Data table with styling
    display_cols = ['operational_phase', 'heart_rate_bpm', 'core_body_temp_c', 'hydration_percent',
                   'fatigue_index', 'stress_level_0_100', 'sleep_debt_hours', 'readiness_score', 'risk_status']
    
    st.dataframe(
        filtered[display_cols].style.background_gradient(cmap='RdYlGn', subset=['readiness_score'])
                                    .background_gradient(cmap='RdYlGn_r', subset=['fatigue_index'])
                                    .format({'readiness_score': '{:.1f}', 'fatigue_index': '{:.2f}',
                                           'core_body_temp_c': '{:.1f}', 'heart_rate_bpm': '{:.0f}'}),
        use_container_width=True,
        height=400
    )
    
    # Scatter matrix for detailed analysis
    st.subheader("🔍 Multi-Variable Correlation Analysis")
    scatter_vars = st.multiselect("Select variables for scatter matrix", 
                                  ['heart_rate_bpm', 'core_body_temp_c', 'fatigue_index', 
                                   'stress_level_0_100', 'readiness_score', 'hydration_percent'],
                                  default=['heart_rate_bpm', 'fatigue_index', 'readiness_score'])
    
    if len(scatter_vars) >= 2:
        fig_scatter = px.scatter_matrix(filtered, dimensions=scatter_vars, color='risk_status',
                                       template="plotly_dark",
                                       color_discrete_map={'FIT': '#00ff88', 'MONITOR': '#ffcc00', 'HIGH_RISK': '#ff4b4b'})
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Export options
    st.markdown("---")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button("📥 Export Filtered Data (CSV)", filtered.to_csv(index=False), 
                          "unit_report.csv", "text/csv", use_container_width=True)
    with export_col2:
        st.download_button("📥 Export Full Database (CSV)", df.to_csv(index=False), 
                          "full_unit_data.csv", "text/csv", use_container_width=True)

# ===============================================
# PAGE 4: ENHANCED INTEL & ANALYTICS
# ===============================================

elif page == "📈 INTEL & ANALYTICS":
    st.markdown('<div class="main-header"><h1>📊 MODEL INTELLIGENCE & ANALYTICS</h1></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🎯 Model Performance Metrics")
        
        # RF Model accuracy
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
        gb_acc = accuracy_score(y_test, gb_model.predict(X_test_scaled))
        
        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            st.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
        with acc_col2:
            st.metric("Gradient Boosting Accuracy", f"{gb_acc*100:.2f}%")
        
        # Confusion Matrix
        st.subheader("📋 Confusion Matrix (RF Model)")
        cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
        fig_cm = px.imshow(cm, text_auto=True, template="plotly_dark", 
                          color_continuous_scale='Blues',
                          labels=dict(x="Predicted", y="Actual"),
                          x=risk_le.classes_, y=risk_le.classes_)
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with c2:
        st.subheader("🔬 Feature Importance Analysis")
        
        # Feature Importance
        imp = pd.DataFrame({
            'Feature': features, 
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', 
                        template="plotly_dark", title="Primary Risk Drivers",
                        color='Importance', color_continuous_scale='Viridis')
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Top 5 features
        st.info(f"**Top Risk Indicators:** {', '.join(imp.head(5)['Feature'].tolist())}")
    
    st.markdown("---")
    
    # Second row of analytics
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🌡️ Vital Signs Correlation Heatmap")
        corr_vars = ['heart_rate_bpm', 'core_body_temp_c', 'fatigue_index', 
                     'stress_level_0_100', 'sleep_debt_hours', 'hydration_percent', 'spo2']
        corr = df[corr_vars].corr()
        
        fig_corr = px.imshow(corr, text_auto='.2f', template="plotly_dark", 
                            color_continuous_scale='RdBu_r', aspect='auto')
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col4:
        st.subheader("📊 Risk Status by Operational Phase")
        phase_risk = pd.crosstab(df['operational_phase'], df['risk_status'])
        
        fig_phase_risk = go.Figure()
        for status in ['FIT', 'MONITOR', 'HIGH_RISK']:
            if status in phase_risk.columns:
                color = {'FIT': '#00ff88', 'MONITOR': '#ffcc00', 'HIGH_RISK': '#ff4b4b'}[status]
                fig_phase_risk.add_trace(go.Bar(
                    name=status, 
                    x=phase_risk.index, 
                    y=phase_risk[status],
                    marker_color=color
                ))
        
        fig_phase_risk.update_layout(
            template="plotly_dark", 
            barmode='stack',
            height=500,
            xaxis_title="Operational Phase",
            yaxis_title="Personnel Count"
        )
        st.plotly_chart(fig_phase_risk, use_container_width=True)
    
    st.markdown("---")
    
    # ROC Curves
    st.subheader("📈 ROC Curves - Multi-Class Performance")
    
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    y_score = rf_model.predict_proba(X_test_scaled)
    
    fig_roc = go.Figure()
    colors = cycle(['#00ff88', '#ffcc00', '#ff4b4b'])
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{risk_le.classes_[i]} (AUC = {roc_auc:.2f})',
            mode='lines',
            line=dict(color=color, width=2)
        ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig_roc.update_layout(
        template="plotly_dark",
        title="ROC Curve - All Classes",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Classification Report
    st.subheader("📄 Detailed Classification Report")
    y_pred = rf_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=risk_le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score'])
                               .format(precision=2), use_container_width=True)

# ===============================================
# PAGE 5: THREAT MONITOR
# ===============================================

elif page == "⚠️ THREAT MONITOR":
    st.markdown('<div class="main-header"><h1>⚠️ REAL-TIME THREAT ASSESSMENT</h1></div>', unsafe_allow_html=True)
    
    # Threat Level Metrics
    threat_col1, threat_col2, threat_col3, threat_col4 = st.columns(4)
    
    critical_count = len(df[df['risk_status'] == 'HIGH_RISK'])
    warning_count = len(df[df['risk_status'] == 'MONITOR'])
    temp_critical = len(df[df['core_body_temp_c'] > 38.5])
    hr_critical = len(df[df['heart_rate_bpm'] > 140])
    
    with threat_col1:
        st.metric("🔴 CRITICAL STATUS", critical_count, f"{(critical_count/len(df)*100):.1f}%", delta_color="inverse")
    with threat_col2:
        st.metric("🟡 WARNING STATUS", warning_count, f"{(warning_count/len(df)*100):.1f}%", delta_color="inverse")
    with threat_col3:
        st.metric("🌡️ TEMP ALERTS", temp_critical, delta_color="inverse")
    with threat_col4:
        st.metric("💓 HR ALERTS", hr_critical, delta_color="inverse")
    
    st.markdown("---")
    
    # Real-time threat timeline
    threat_left, threat_right = st.columns([2, 1])
    
    with threat_left:
        st.subheader("📡 Physiological Threat Timeline")
        
        # Create threat events based on anomalies
        threat_events = df[df['is_anomaly'] == True].copy()
        threat_events['threat_level'] = threat_events.apply(
            lambda x: 'CRITICAL' if x['risk_status'] == 'HIGH_RISK' else 'WARNING', axis=1
        )
        
        if len(threat_events) > 0:
            fig_timeline = px.scatter(threat_events.head(50), 
                                     x=threat_events.head(50).index, 
                                     y='heart_rate_bpm',
                                     color='threat_level',
                                     size='core_body_temp_c',
                                     hover_data=['fatigue_index', 'stress_level_0_100'],
                                     template="plotly_dark",
                                     color_discrete_map={'CRITICAL': '#ff4b4b', 'WARNING': '#ffcc00'})
            fig_timeline.update_layout(height=400, xaxis_title="Personnel ID", yaxis_title="Heart Rate (bpm)")
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.success("✅ No anomalies detected in current monitoring cycle")
    
    with threat_right:
        st.subheader("🚨 Active Threat List")
        
        threat_personnel = df[df['is_anomaly'] == True].head(10)
        if not threat_personnel.empty:
            for idx, row in threat_personnel.iterrows():
                threat_class = "alert-critical" if row['risk_status'] == 'HIGH_RISK' else "alert-warning"
                st.markdown(f"""
                <div class="alert-box {threat_class}">
                    <strong>ID-{idx}</strong> | {row['operational_phase'].upper()}<br>
                    HR: {row['heart_rate_bpm']:.0f} | Temp: {row['core_body_temp_c']:.1f}°C<br>
                    Status: <strong>{row['risk_status']}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ All personnel within normal parameters")
    
    st.markdown("---")
    
    # Threat distribution analysis
    st.subheader("📊 Threat Distribution Analysis")
    
    threat_anal1, threat_anal2, threat_anal3 = st.columns(3)
    
    with threat_anal1:
        st.markdown("**High Heart Rate Distribution**")
        hr_threat = df[df['heart_rate_bpm'] > 120]
        fig_hr_threat = px.histogram(hr_threat, x='heart_rate_bpm', nbins=20,
                                     template="plotly_dark", color_discrete_sequence=['#ff4b4b'])
        fig_hr_threat.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_hr_threat, use_container_width=True)
        st.metric("Personnel Affected", len(hr_threat))
    
    with threat_anal2:
        st.markdown("**Elevated Temperature Distribution**")
        temp_threat = df[df['core_body_temp_c'] > 37.8]
        fig_temp_threat = px.histogram(temp_threat, x='core_body_temp_c', nbins=20,
                                       template="plotly_dark", color_discrete_sequence=['#ff4b4b'])
        fig_temp_threat.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_temp_threat, use_container_width=True)
        st.metric("Personnel Affected", len(temp_threat))
    
    with threat_anal3:
        st.markdown("**Critical Fatigue Distribution**")
        fatigue_threat = df[df['fatigue_index'] > 0.6]
        fig_fatigue_threat = px.histogram(fatigue_threat, x='fatigue_index', nbins=20,
                                          template="plotly_dark", color_discrete_sequence=['#ff4b4b'])
        fig_fatigue_threat.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_fatigue_threat, use_container_width=True)
        st.metric("Personnel Affected", len(fatigue_threat))
    
    # Threat prediction over time
    st.markdown("---")
    st.subheader("🔮 Projected Threat Escalation")
    
    hours_ahead = st.slider("Project threats ahead (hours)", 1, 48, 12)
    
    # Simulate threat escalation
    escalation_data = []
    for h in range(0, hours_ahead + 1, 2):
        fatigue_mult = 1 + (h * 0.03)
        stress_mult = 1 + (h * 0.02)
        
        projected_critical = len(df[
            (df['fatigue_index'] * fatigue_mult > 0.7) | 
            (df['stress_level_0_100'] * stress_mult > 80)
        ])
        
        escalation_data.append({
            'Hours': h,
            'Projected Critical': projected_critical,
            'Current': critical_count
        })
    
    escalation_df = pd.DataFrame(escalation_data)
    
    fig_escalation = go.Figure()
    fig_escalation.add_trace(go.Scatter(
        x=escalation_df['Hours'], 
        y=escalation_df['Projected Critical'],
        mode='lines+markers',
        name='Projected',
        line=dict(color='#ff4b4b', width=3),
        fill='tozeroy'
    ))
    fig_escalation.add_trace(go.Scatter(
        x=escalation_df['Hours'], 
        y=escalation_df['Current'],
        mode='lines',
        name='Current Baseline',
        line=dict(color='#00d4ff', width=2, dash='dash')
    ))
    
    fig_escalation.update_layout(
        template="plotly_dark",
        title=f"Threat Escalation Forecast ({hours_ahead}h)",
        xaxis_title="Hours from Now",
        yaxis_title="Critical Personnel Count",
        height=400
    )
    st.plotly_chart(fig_escalation, use_container_width=True)
    
    if escalation_df.iloc[-1]['Projected Critical'] > critical_count * 1.5:
        st.error(f"⚠️ **WARNING**: Projected critical personnel may increase by {((escalation_df.iloc[-1]['Projected Critical']/critical_count - 1) * 100):.0f}% in {hours_ahead} hours")
    else:
        st.info(f"ℹ️ Threat levels projected to remain stable over next {hours_ahead} hours")

# ===============================================
# PAGE 6: ENVIRONMENTAL ANALYSIS
# ===============================================

elif page == "🌡️ ENVIRONMENTAL":
    st.markdown('<div class="main-header"><h1>🌡️ ENVIRONMENTAL IMPACT ANALYSIS</h1></div>', unsafe_allow_html=True)
    
    # Environmental metrics
    env_col1, env_col2, env_col3, env_col4 = st.columns(4)
    
    avg_ambient = df['ambient_temp_c'].mean()
    heat_stress = len(df[df['ambient_temp_c'] > 30])
    cold_exposure = len(df[df['ambient_temp_c'] < 10])
    optimal_conditions = len(df[(df['ambient_temp_c'] >= 15) & (df['ambient_temp_c'] <= 25)])
    
    with env_col1:
        st.metric("Avg Ambient Temp", f"{avg_ambient:.1f}°C")
    with env_col2:
        st.metric("Heat Stress Risk", heat_stress, delta_color="inverse")
    with env_col3:
        st.metric("Cold Exposure Risk", cold_exposure, delta_color="inverse")
    with env_col4:
        st.metric("Optimal Conditions", optimal_conditions)
    
    st.markdown("---")
    
    # Environmental impact visualization
    env_left, env_right = st.columns([3, 2])
    
    with env_left:
        st.subheader("🌡️ Temperature Impact on Performance")
        
        # Create temperature bins
        df['temp_category'] = pd.cut(df['ambient_temp_c'], 
                                     bins=[-np.inf, 10, 20, 30, np.inf],
                                     labels=['Cold (<10°C)', 'Cool (10-20°C)', 
                                            'Moderate (20-30°C)', 'Hot (>30°C)'])
        
        temp_performance = df.groupby('temp_category').agg({
            'readiness_score': 'mean',
            'fatigue_index': 'mean',
            'heart_rate_bpm': 'mean',
            'core_body_temp_c': 'mean'
        }).reset_index()
        
        fig_temp_perf = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_temp_perf.add_trace(
            go.Bar(name='Readiness Score', x=temp_performance['temp_category'], 
                   y=temp_performance['readiness_score'], marker_color='#00d4ff'),
            secondary_y=False
        )
        
        fig_temp_perf.add_trace(
            go.Scatter(name='Avg Heart Rate', x=temp_performance['temp_category'], 
                      y=temp_performance['heart_rate_bpm'], 
                      mode='lines+markers', marker_color='#ff4b4b', line=dict(width=3)),
            secondary_y=True
        )
        
        fig_temp_perf.update_layout(template="plotly_dark", height=400)
        fig_temp_perf.update_xaxes(title_text="Temperature Category")
        fig_temp_perf.update_yaxes(title_text="Readiness Score", secondary_y=False)
        fig_temp_perf.update_yaxes(title_text="Heart Rate (bpm)", secondary_y=True)
        
        st.plotly_chart(fig_temp_perf, use_container_width=True)
    
    with env_right:
        st.subheader("📊 Environmental Risk Matrix")
        
        risk_by_temp = pd.crosstab(df['temp_category'], df['risk_status'])
        
        fig_risk_temp = go.Figure()
        for status in ['FIT', 'MONITOR', 'HIGH_RISK']:
            if status in risk_by_temp.columns:
                color = {'FIT': '#00ff88', 'MONITOR': '#ffcc00', 'HIGH_RISK': '#ff4b4b'}[status]
                fig_risk_temp.add_trace(go.Bar(
                    name=status,
                    x=risk_by_temp.index,
                    y=risk_by_temp[status],
                    marker_color=color
                ))
        
        fig_risk_temp.update_layout(
            template="plotly_dark",
            barmode='stack',
            height=400,
            xaxis_title="Temperature Category",
            yaxis_title="Personnel Count"
        )
        st.plotly_chart(fig_risk_temp, use_container_width=True)
    
    st.markdown("---")
    
    # Hydration correlation with environment
    st.subheader("💧 Hydration Status vs Environmental Conditions")
    
    hydro_col1, hydro_col2 = st.columns(2)
    
    with hydro_col1:
        fig_hydro_scatter = px.scatter(df, x='ambient_temp_c', y='hydration_percent',
                                      color='risk_status', template="plotly_dark",
                                      color_discrete_map={'FIT': '#00ff88', 'MONITOR': '#ffcc00', 
                                                         'HIGH_RISK': '#ff4b4b'},
                                      trendline="lowess",
                                      title="Hydration vs Ambient Temperature")
        fig_hydro_scatter.update_layout(height=400)
        st.plotly_chart(fig_hydro_scatter, use_container_width=True)
    
    with hydro_col2:
        # Dehydration risk by temperature
        df['dehydration_risk'] = df['hydration_percent'] < 70
        dehydration_by_temp = df.groupby('temp_category')['dehydration_risk'].agg(['sum', 'count'])
        dehydration_by_temp['percentage'] = (dehydration_by_temp['sum'] / dehydration_by_temp['count'] * 100)
        
        fig_dehydro = px.bar(dehydration_by_temp.reset_index(), 
                            x='temp_category', y='percentage',
                            template="plotly_dark",
                            color='percentage',
                            color_continuous_scale='Reds',
                            title="Dehydration Risk by Temperature")
        fig_dehydro.update_layout(height=400, yaxis_title="% at Risk")
        st.plotly_chart(fig_dehydro, use_container_width=True)
    
    st.markdown("---")
    
    # Heat/Cold injury risk calculator
    st.subheader("🎯 Environmental Injury Risk Calculator")
    
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        calc_ambient = st.slider("Ambient Temperature (°C)", -10, 50, 25)
        calc_humidity = st.slider("Humidity (%)", 0, 100, 50)
        calc_wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
    
    with calc_col2:
        calc_activity = st.selectbox("Activity Level", ["Rest", "Light", "Moderate", "Heavy", "Extreme"])
        calc_duration = st.slider("Exposure Duration (hours)", 0.5, 12.0, 2.0)
        calc_gear = st.selectbox("Gear Load", ["Light (<10kg)", "Standard (10-25kg)", "Heavy (>25kg)"])
    
    with calc_col3:
        # Calculate heat index
        if calc_ambient > 25:
            heat_index = calc_ambient + (0.5 * calc_humidity / 10)
            risk_score = heat_index + (calc_duration * 5)
            
            if calc_activity == "Extreme":
                risk_score *= 1.5
            elif calc_activity == "Heavy":
                risk_score *= 1.3
            
            st.metric("Heat Stress Index", f"{heat_index:.1f}°C")
            st.metric("Injury Risk Score", f"{risk_score:.0f}")
            
            if risk_score > 60:
                st.error("🔴 EXTREME RISK - Immediate action required")
            elif risk_score > 40:
                st.warning("🟡 HIGH RISK - Limit exposure")
            else:
                st.success("🟢 MODERATE RISK - Monitor conditions")
        else:
            # Wind chill for cold
            wind_chill = calc_ambient - (calc_wind * 0.5)
            st.metric("Wind Chill", f"{wind_chill:.1f}°C")
            st.metric("Cold Exposure Time", f"{calc_duration:.1f}h")
            
            if wind_chill < -10:
                st.error("🔴 EXTREME COLD - Frostbite risk high")
            elif wind_chill < 0:
                st.warning("🟡 COLD STRESS - Hypothermia possible")
            else:
                st.success("🟢 SAFE - Standard cold weather precautions")
    
    # Environmental recommendations
    st.markdown("---")
    st.subheader("📋 Environmental Mitigation Recommendations")
    
    recommendations = []
    
    if heat_stress > len(df) * 0.3:
        recommendations.append("⚠️ High heat stress affecting >30% of personnel - Increase hydration protocols")
    if cold_exposure > len(df) * 0.2:
        recommendations.append("❄️ Cold exposure risk detected - Ensure proper thermal gear distribution")
    if df['hydration_percent'].mean() < 75:
        recommendations.append("💧 Unit average hydration below optimal - Enforce hydration schedule")
    if avg_ambient > 30:
        recommendations.append("🌡️ Extreme heat conditions - Consider operational timing adjustments")
    elif avg_ambient < 5:
        recommendations.append("🥶 Extreme cold conditions - Limit exposure duration for non-critical tasks")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ Environmental conditions within acceptable parameters")

# ===============================================
# FOOTER
# ===============================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<div style='text-align: center; color: gray;'>VANGUARD SYSTEM v2.0</div>", unsafe_allow_html=True)
with footer_col2:
    st.markdown(f"<div style='text-align: center; color: gray;'>SECURE MILITARY ENCRYPTION ACTIVE</div>", unsafe_allow_html=True)
with footer_col3:
    st.markdown(f"<div style='text-align: center; color: gray;'>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)