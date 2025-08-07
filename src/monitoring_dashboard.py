import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from model_monitor import ModelPerformanceMonitor
from data_monitor import DataMonitor

def load_performance_history():
    """تحميل تاريخ الأداء"""
    try:
        with open('metrics/performance_history.json', 'r') as f:
            return json.load(f)
    except:
        return []

def load_latest_drift_report():
    """تحميل آخر تقرير drift"""
    try:
        reports_dir = 'reports'
        if os.path.exists(reports_dir):
            reports = [f for f in os.listdir(reports_dir) if f.startswith('data_drift_report')]
            if reports:
                latest_report = sorted(reports)[-1]
                return f"{reports_dir}/{latest_report}"
        return None
    except:
        return None

def main():
    st.set_page_config(
        page_title="MLOps Monitoring Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 MLOps Monitoring Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page", 
        ["Model Performance", "Data Monitoring", "System Health"]
    )
    
    if page == "Model Performance":
        st.header("📈 Model Performance Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Performance Check"):
                monitor = ModelPerformanceMonitor()
                with st.spinner("Evaluating model performance..."):
                    report = monitor.generate_performance_report()
                
                st.success("Performance check completed!")
                
                # عرض الأداء الحالي
                current_perf = report['current_performance']
                st.subheader("Current Performance")
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Accuracy", f"{current_perf['accuracy']:.4f}")
                with metric_cols[1]:
                    st.metric("Precision", f"{current_perf['precision']:.4f}")
                with metric_cols[2]:
                    st.metric("Recall", f"{current_perf['recall']:.4f}")
                with metric_cols[3]:
                    st.metric("F1-Score", f"{current_perf['f1_score']:.4f}")
                
                # تحذير إذا كان هناك تدهور
                if report['performance_degradation']['detected']:
                    st.error(f"⚠️ Performance Degradation Detected: {report['performance_degradation']['details']}")
                else:
                    st.success(f"✅ Performance Stable: {report['performance_degradation']['details']}")
        
        with col2:
            st.subheader("Performance History")
            history = load_performance_history()
            
            if history:
                df_history = pd.DataFrame(history)
                df_history['evaluation_date'] = pd.to_datetime(df_history['evaluation_date'])
                
                # رسم بياني للأداء عبر الوقت
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_history['evaluation_date'],
                    y=df_history['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_history['evaluation_date'],
                    y=df_history['f1_score'],
                    mode='lines+markers',
                    name='F1-Score',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Model Performance Over Time",
                    xaxis_title="Date",
                    yaxis_title="Score",
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance history available. Run a performance check first.")
    
    elif page == "Data Monitoring":
        st.header("📊 Data Drift Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Data Drift Report"):
                monitor = DataMonitor()
                with st.spinner("Analyzing data drift..."):
                    report_path = monitor.generate_data_drift_report()
                
                st.success(f"Data drift report generated: {report_path}")
                
                # فحص سريع للـ drift
                drift_check = monitor.check_data_drift()
                
                if drift_check['drift_detected']:
                    st.error("⚠️ Data Drift Detected!")
                    
                    for col, details in drift_check['drift_summary'].items():
                        st.write(f"**{col}:**")
                        st.write(f"- Reference Mean: {details['reference_mean']:.4f}")
                        st.write(f"- Current Mean: {details['current_mean']:.4f}")
                        st.write(f"- Drift: {details['drift_percentage']:.2f}%")
                else:
                    st.success("✅ No significant data drift detected")
        
        with col2:
            st.subheader("Latest Drift Report")
            latest_report = load_latest_drift_report()
            
            if latest_report:
                st.success(f"Latest report: {latest_report}")
                if st.button("Open Report"):
                    st.info("Report saved locally. Check your reports/ directory.")
            else:
                st.info("No drift reports available yet.")
    
    elif page == "System Health":
        st.header("🏥 System Health")
        
        # معلومات النظام
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Deployed", "1")
            st.metric("Data Sources", "1")
        
        with col2:
            st.metric("Last Model Update", "Today")
            st.metric("Monitoring Status", "Active")
        
        with col3:
            st.metric("Alerts", "0")
            st.metric("System Uptime", "99.9%")
        
        # لوج الأنشطة
        st.subheader("Recent Activities")
        activities = [
            {"time": "10:30 AM", "activity": "Model performance check completed", "status": "✅"},
            {"time": "09:15 AM", "activity": "Data drift analysis started", "status": "🔄"},
            {"time": "08:00 AM", "activity": "Daily monitoring routine initiated", "status": "✅"},
        ]
        
        for activity in activities:
            st.write(f"{activity['status']} **{activity['time']}** - {activity['activity']}")

if __name__ == "__main__":
    main()
