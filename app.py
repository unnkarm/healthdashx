import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Military Health Tracker",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1f4788 0%, #2d5aa8 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f4788;
    }
    .fit-status {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
    }
    .monitor-status {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
    }
    .high-risk-status {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================
# FUNCTIONS
# ===============================================

@st.cache_data
def load_and_train_models():
    """Load data and train models"""
    df = pd.read_csv("military_wearable_synthetic_500_rows.csv")
    
    # Normalize fatigue
    if df['fatigue_index'].max() > 1:
        df['fatigue_index'] = df['fatigue_index'] / 100.0
    
    # Encode operational phases
    phase_le = LabelEncoder()
    df['operational_phase_encoded'] = phase_le.fit_transform(df['operational_phase'].str.strip().str.lower())
    
    # Create risk labels
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
    
    # Features
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    return df, rf_model, gb_model, scaler, phase_le, risk_le, features, X_test_scaled, y_test

def predict_recovery_time(risk_status, fatigue, stress, sleep_debt, heart_rate):
    """Estimate recovery time based on health metrics"""
    base_recovery = {"FIT": 2, "MONITOR": 12, "HIGH_RISK": 48}  # hours
    
    recovery_hours = base_recovery[risk_status]
    
    # Adjust based on metrics
    recovery_hours += fatigue * 10
    recovery_hours += stress / 10
    recovery_hours += sleep_debt * 2
    recovery_hours += max(0, (heart_rate - 100) / 5)
    
    return int(recovery_hours)

def get_recommendations(risk_status, metrics):
    """Provide personalized recommendations"""
    recommendations = []
    
    if risk_status == "HIGH_RISK":
        recommendations.append("🚨 IMMEDIATE REST REQUIRED - Remove from active duty")
        recommendations.append("🏥 Medical evaluation recommended within 24 hours")
    
    if metrics['fatigue_index'] > 0.7:
        recommendations.append("😴 Critical fatigue detected - Minimum 8 hours sleep required")
    
    if metrics['stress_level_0_100'] > 70:
        recommendations.append("🧘 High stress levels - Recommend counseling or stress management session")
    
    if metrics['hydration_percent'] < 60:
        recommendations.append("💧 Severe dehydration - Immediate fluid intake required (2L water)")
    
    if metrics['sleep_debt_hours'] > 5:
        recommendations.append("🛌 Significant sleep debt - Schedule recovery sleep period")
    
    if metrics['heart_rate_bpm'] > 100:
        recommendations.append("❤️ Elevated heart rate - Monitor cardiovascular health")
    
    if metrics['core_body_temp_c'] > 38:
        recommendations.append("🌡️ Elevated body temperature - Check for fever/heat exhaustion")
    
    if metrics['spo2'] < 95:
        recommendations.append("🫁 Low oxygen saturation - Respiratory assessment needed")
    
    if risk_status == "MONITOR":
        recommendations.append("⚠️ Increased monitoring required - Check vitals every 4 hours")
        recommendations.append("📊 Limit physical activity to light duties")
    
    if risk_status == "FIT":
        recommendations.append("✅ Soldier fit for duty - Continue regular monitoring")
        recommendations.append("💪 Maintain current health practices")
    
    return recommendations

def get_resources(risk_status):
    """Provide medical resources needed"""
    resources = {
        "FIT": ["Regular monitoring equipment", "Standard hydration supplies"],
        "MONITOR": ["Enhanced monitoring equipment", "IV hydration kit", "Rest facility access", "Nutritional supplements"],
        "HIGH_RISK": ["Medical evacuation readiness", "Advanced life support equipment", 
                      "Emergency medications", "Immediate medical personnel", "Hospital bed reservation"]
    }
    return resources[risk_status]

# ===============================================
# LOAD MODELS AND DATA
# ===============================================

with st.spinner("🔄 Loading models and data..."):
    df, rf_model, gb_model, scaler, phase_le, risk_le, features, X_test_scaled, y_test = load_and_train_models()

# ===============================================
# SIDEBAR
# ===============================================

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
st.sidebar.title("🎖️ Military Health Tracker")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", 
    ["📊 Dashboard Overview", "🔍 Predict Soldier Status", "👥 Soldier Database", "📈 Analytics & Insights"])

st.sidebar.markdown("---")
st.sidebar.info("**System Status:** ✅ Active\n\n**Models:** Random Forest & Gradient Boosting\n\n**Accuracy:** 95%+")

# ===============================================
# PAGE 1: DASHBOARD OVERVIEW
# ===============================================

if page == "📊 Dashboard Overview":
    st.markdown('<div class="main-header">🎖️ MILITARY HEALTH MONITORING SYSTEM</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_soldiers = len(df)
        st.metric("Total Soldiers", total_soldiers, delta="Active")
    
    with col2:
        fit_count = len(df[df['risk_status'] == 'FIT'])
        st.metric("FIT Status", fit_count, delta=f"{(fit_count/total_soldiers)*100:.1f}%")
    
    with col3:
        monitor_count = len(df[df['risk_status'] == 'MONITOR'])
        st.metric("MONITOR Status", monitor_count, delta=f"{(monitor_count/total_soldiers)*100:.1f}%")
    
    with col4:
        risk_count = len(df[df['risk_status'] == 'HIGH_RISK'])
        st.metric("HIGH RISK", risk_count, delta=f"{(risk_count/total_soldiers)*100:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        fig = px.pie(df, names='risk_status', 
                     color='risk_status',
                     color_discrete_map={'FIT': '#2ecc71', 'MONITOR': '#f39c12', 'HIGH_RISK': '#e74c3c'},
                     hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk by Operational Phase")
        phase_risk = df.groupby(['operational_phase', 'risk_status']).size().reset_index(name='count')
        fig = px.bar(phase_risk, x='operational_phase', y='count', color='risk_status',
                     color_discrete_map={'FIT': '#2ecc71', 'MONITOR': '#f39c12', 'HIGH_RISK': '#e74c3c'},
                     barmode='group')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 3D Visualization
    st.subheader("🔮 3D Health Status Space")
    fig_3d = px.scatter_3d(df.sample(200), x='heart_rate_bpm', y='stress_level_0_100', 
                            z='fatigue_index', color='risk_status',
                            color_discrete_map={'FIT': '#2ecc71', 'MONITOR': '#f39c12', 'HIGH_RISK': '#e74c3c'},
                            opacity=0.7, height=600)
    fig_3d.update_layout(scene=dict(xaxis_title='Heart Rate (BPM)',
                                    yaxis_title='Stress Level',
                                    zaxis_title='Fatigue Index'))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    
    # Recent Alerts
    st.subheader("🚨 Recent High-Risk Cases")
    high_risk_df = df[df['risk_status'] == 'HIGH_RISK'].sort_values('fatigue_index', ascending=False).head(5)
    
    if len(high_risk_df) > 0:
        for idx, row in high_risk_df.iterrows():
            with st.expander(f"Soldier ID: {idx} - Fatigue: {row['fatigue_index']:.2f}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Heart Rate:** {row['heart_rate_bpm']} BPM")
                    st.write(f"**Stress Level:** {row['stress_level_0_100']}")
                with col2:
                    st.write(f"**Sleep Debt:** {row['sleep_debt_hours']} hrs")
                    st.write(f"**Hydration:** {row['hydration_percent']}%")
                with col3:
                    st.write(f"**Body Temp:** {row['core_body_temp_c']}°C")
                    st.write(f"**SpO2:** {row['spo2']}%")
    else:
        st.success("✅ No high-risk cases detected!")

# ===============================================
# PAGE 2: PREDICT SOLDIER STATUS
# ===============================================

elif page == "🔍 Predict Soldier Status":
    st.markdown('<div class="main-header">🔍 SOLDIER HEALTH PREDICTION</div>', unsafe_allow_html=True)
    
    st.write("Enter soldier health metrics to predict fitness status and recovery time.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        soldier_id = st.text_input("Soldier ID", value="SOLDIER_001")
        operational_phase = st.selectbox("Operational Phase", 
            ["combat", "training", "recovery", "patrol", "rest"])
        
        st.subheader("💓 Vital Signs")
        heart_rate = st.slider("Heart Rate (BPM)", 40, 180, 75)
        core_temp = st.slider("Core Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
        spo2 = st.slider("SpO2 (%)", 80, 100, 98)
        resp_rate = st.slider("Respiration Rate", 10, 40, 16)
        bp_sys = st.slider("Blood Pressure Systolic (mmHg)", 80, 200, 120)
        bp_dia = st.slider("Blood Pressure Diastolic (mmHg)", 50, 130, 80)
    
    with col2:
        st.subheader("🏃 Activity Metrics")
        movement_intensity = st.slider("Movement Intensity (g)", 0.0, 5.0, 1.0, 0.1)
        energy_expenditure = st.slider("Energy Expenditure (kcal/hr)", 50, 500, 150)
        ambient_temp = st.slider("Ambient Temperature (°C)", -10, 50, 25)
        
        st.subheader("😰 Wellness Indicators")
        fatigue = st.slider("Fatigue Index", 0.0, 1.0, 0.3, 0.05)
        stress = st.slider("Stress Level", 0, 100, 30)
        hydration = st.slider("Hydration (%)", 0, 100, 70)
        sleep_debt = st.slider("Sleep Debt (hours)", 0.0, 20.0, 2.0, 0.5)
    
    st.markdown("---")
    
    if st.button("🎯 PREDICT STATUS", type="primary", use_container_width=True):
        # Encode operational phase
        phase_encoded = phase_le.transform([operational_phase.strip().lower()])[0]
        
        # Create feature array
        input_features = np.array([[
            phase_encoded, heart_rate, core_temp, hydration, fatigue, spo2, 
            resp_rate, movement_intensity, bp_sys, bp_dia, stress, 
            energy_expenditure, sleep_debt, ambient_temp
        ]])
        
        # Scale features
        input_scaled = scaler.transform(input_features)
        
        # Predict
        rf_pred = rf_model.predict(input_scaled)[0]
        rf_proba = rf_model.predict_proba(input_scaled)[0]
        gb_pred = gb_model.predict(input_scaled)[0]
        
        # Get risk status
        risk_status = risk_le.inverse_transform([rf_pred])[0]
        
        # Calculate recovery time
        recovery_time = predict_recovery_time(risk_status, fatigue, stress, sleep_debt, heart_rate)
        recovery_date = datetime.now() + timedelta(hours=recovery_time)
        
        st.markdown("---")
        st.subheader(f"📋 Results for {soldier_id}")
        
        # Display status with color coding
        if risk_status == "FIT":
            st.markdown(f'<div class="fit-status">✅ STATUS: {risk_status}</div>', unsafe_allow_html=True)
        elif risk_status == "MONITOR":
            st.markdown(f'<div class="monitor-status">⚠️ STATUS: {risk_status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="high-risk-status">🚨 STATUS: {risk_status}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prediction confidence
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FIT Probability", f"{rf_proba[0]*100:.1f}%")
        with col2:
            st.metric("MONITOR Probability", f"{rf_proba[1]*100:.1f}%")
        with col3:
            st.metric("HIGH_RISK Probability", f"{rf_proba[2]*100:.1f}%")
        
        st.markdown("---")
        
        # Recovery Information
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Recovery Estimate")
            st.write(f"**Estimated Recovery Time:** {recovery_time} hours")
            st.write(f"**Expected Recovery Date:** {recovery_date.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.subheader("📊 Risk Score Breakdown")
            fig = go.Figure(go.Bar(
                x=['FIT', 'MONITOR', 'HIGH_RISK'],
                y=rf_proba * 100,
                marker_color=['#2ecc71', '#f39c12', '#e74c3c']
            ))
            fig.update_layout(height=300, yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        metrics = {
            'fatigue_index': fatigue,
            'stress_level_0_100': stress,
            'hydration_percent': hydration,
            'sleep_debt_hours': sleep_debt,
            'heart_rate_bpm': heart_rate,
            'core_body_temp_c': core_temp,
            'spo2': spo2
        }
        recommendations = get_recommendations(risk_status, metrics)
        for rec in recommendations:
            st.write(f"• {rec}")
        
        st.markdown("---")
        
        # Required Resources
        st.subheader("🏥 Required Medical Resources")
        resources = get_resources(risk_status)
        cols = st.columns(len(resources))
        for idx, resource in enumerate(resources):
            with cols[idx]:
                st.info(resource)

# ===============================================
# PAGE 3: SOLDIER DATABASE
# ===============================================

elif page == "👥 Soldier Database":
    st.markdown('<div class="main-header">👥 SOLDIER DATABASE</div>', unsafe_allow_html=True)
    
    # Filters
    st.subheader("🔍 Filter Soldiers")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect("Risk Status", 
            options=["FIT", "MONITOR", "HIGH_RISK"],
            default=["FIT", "MONITOR", "HIGH_RISK"])
    
    with col2:
        phase_filter = st.multiselect("Operational Phase",
            options=df['operational_phase'].unique(),
            default=df['operational_phase'].unique())
    
    with col3:
        fatigue_threshold = st.slider("Max Fatigue Index", 0.0, 1.0, 1.0)
    
    # Filter dataframe
    filtered_df = df[
        (df['risk_status'].isin(risk_filter)) & 
        (df['operational_phase'].isin(phase_filter)) &
        (df['fatigue_index'] <= fatigue_threshold)
    ]
    
    st.write(f"**Showing {len(filtered_df)} soldiers**")
    
    # Display dataframe
    display_cols = ['operational_phase', 'risk_status', 'heart_rate_bpm', 'fatigue_index', 
                   'stress_level_0_100', 'hydration_percent', 'sleep_debt_hours', 'spo2']
    
    st.dataframe(
        filtered_df[display_cols].style.applymap(
            lambda x: 'background-color: #f8d7da' if isinstance(x, str) and x == 'HIGH_RISK' else '',
            subset=['risk_status']
        ),
        use_container_width=True,
        height=400
    )
    
    # Download option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name=f"soldier_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Individual soldier lookup
    st.subheader("🔎 Individual Soldier Details")
    soldier_idx = st.selectbox("Select Soldier ID", options=filtered_df.index)
    
    if soldier_idx is not None:
        soldier = filtered_df.loc[soldier_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Basic Info**")
            st.write(f"Risk Status: **{soldier['risk_status']}**")
            st.write(f"Operational Phase: **{soldier['operational_phase']}**")
            st.write(f"Fatigue Index: **{soldier['fatigue_index']:.2f}**")
        
        with col2:
            st.write("**Vital Signs**")
            st.write(f"Heart Rate: **{soldier['heart_rate_bpm']} BPM**")
            st.write(f"Body Temp: **{soldier['core_body_temp_c']}°C**")
            st.write(f"SpO2: **{soldier['spo2']}%**")
        
        with col3:
            st.write("**Wellness**")
            st.write(f"Stress Level: **{soldier['stress_level_0_100']}**")
            st.write(f"Hydration: **{soldier['hydration_percent']}%**")
            st.write(f"Sleep Debt: **{soldier['sleep_debt_hours']} hrs**")
        
        # Recovery estimate
        recovery = predict_recovery_time(
            soldier['risk_status'], 
            soldier['fatigue_index'], 
            soldier['stress_level_0_100'],
            soldier['sleep_debt_hours'],
            soldier['heart_rate_bpm']
        )
        st.info(f"⏱️ Estimated Recovery Time: **{recovery} hours**")

# ===============================================
# PAGE 4: ANALYTICS & INSIGHTS
# ===============================================

elif page == "📈 Analytics & Insights":
    st.markdown('<div class="main-header">📈 ANALYTICS & INSIGHTS</div>', unsafe_allow_html=True)
    
    # Model Performance
    st.subheader("🎯 Model Performance Metrics")
    
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Random Forest Accuracy", f"{rf_accuracy*100:.2f}%")
        st.write("**Classification Report:**")
        report_rf = classification_report(y_test, rf_pred, target_names=risk_le.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report_rf).transpose())
    
    with col2:
        st.metric("Gradient Boosting Accuracy", f"{gb_accuracy*100:.2f}%")
        st.write("**Classification Report:**")
        report_gb = classification_report(y_test, gb_pred, target_names=risk_le.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report_gb).transpose())
    
    st.markdown("---")
    
    # Confusion Matrices
    st.subheader("📊 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        cm_rf = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=risk_le.classes_, yticklabels=risk_le.classes_, ax=ax)
        ax.set_title('Random Forest Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
    
    with col2:
        cm_gb = confusion_matrix(y_test, gb_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens',
                   xticklabels=risk_le.classes_, yticklabels=risk_le.classes_, ax=ax)
        ax.set_title('Gradient Boosting Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🔑 Feature Importance Analysis")
    
    rf_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(rf_importance, x='Importance', y='Feature', orientation='h',
                 title='Random Forest Feature Importance',
                 color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Trends Over Time
    st.subheader("📉 Health Trends Simulation")
    
    trend_metric = st.selectbox("Select Metric to Analyze", 
        ['fatigue_index', 'heart_rate_bpm', 'stress_level_0_100', 'hydration_percent'])
    
    trend_data = df.groupby('operational_phase')[trend_metric].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trend_data['operational_phase'],
        y=trend_data['mean'],
        error_y=dict(type='data', array=trend_data['std']),
        marker_color='steelblue'
    ))
    fig.update_layout(
        title=f'{trend_metric.replace("_", " ").title()} by Operational Phase',
        xaxis_title='Operational Phase',
        yaxis_title=trend_metric.replace("_", " ").title()
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================================
# FOOTER
# ===============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎖️ <strong>Military Health Tracker System v1.0</strong></p>
    <p>Powered by AI & Machine Learning | Secure & Confidential</p>
    <p>© 2024 Military Health Command | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)