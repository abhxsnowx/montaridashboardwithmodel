import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# OEE Calculation Functions (EXACT as per your image)
def calculate_availability(loading_time, down_time):
    """AR = (Loading time - Down time) / Loading time"""
    if loading_time == 0:
        return 0
    return (loading_time - down_time) / loading_time

def calculate_performance(agreed_cycle_time, units_produced, operating_time):
    """PR = (Specified Cycle Time X Units produced) / Operating time"""
    if operating_time == 0:
        return 0
    return (agreed_cycle_time * units_produced) / operating_time

def calculate_quality(units_produced, defective_units):
    """QR = (Units Produced - Defective units) / Unit Produced"""
    if units_produced == 0:
        return 0
    return (units_produced - defective_units) / units_produced

def calculate_oee(ar, pr, qr):
    """OEE = AR * PR * QR * 100%"""
    return ar * pr * qr * 100

# Page config
st.set_page_config(page_title="Montari Production Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .oee-section {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .formula-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .prediction-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_production_models():
    try:
        model = joblib.load(r'C:\Users\Abhilash\Downloads\streammontari\new_model\new_production_xgboost_model.pkl')
        scaler = joblib.load(r'C:\Users\Abhilash\Downloads\streammontari\new_model\new_production_scaler.pkl')
        feature_cols = joblib.load(r'C:\Users\Abhilash\Downloads\streammontari\new_model\new_feature_names.pkl')
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

model, scaler, feature_cols = load_production_models()

# Model mapping from your Excel
MODEL_MAP = {
    "4KW": "4KW",
    "7.5KW": "7.5KW", 
    "9KW": "9KW",
    "11KW": "11KW",
    "11KW6P": "11KW6P"
}

# Shift mapping
SHIFT_MAP = {
    "Full (1.0)": 1.0,
    "Half (0.5)": 0.5
}

# Header
st.markdown('<div class="main-header">Montari Production + OEE Analytics Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Production Prediction", "Mont Line OEE"])

if page == "Production Prediction":
    st.header("Production Prediction Model")
    
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    
    # Input parameters matching your Excel columns (WITHOUT Material_Quality_Percentage)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model & Schedule")
        model_type = st.selectbox("Select Model (KW)", list(MODEL_MAP.keys()))
        shift_type = st.selectbox("Shift", list(SHIFT_MAP.keys()))
        shift_value = SHIFT_MAP[shift_type]
        plan_capacity = st.number_input("Planned Capacity", min_value=1, max_value=100, value=16, step=1)
    
    with col2:
        st.subheader("Resources")
        manpower = st.number_input("Manpower", min_value=1, max_value=10, value=6, step=1)
        total_downtime = st.number_input("Total Downtime (min)", min_value=0, max_value=500, value=149, step=1)
        material_issue = st.selectbox("Material Issue", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col3:
        st.subheader("Quality & History")
        has_downtime = st.selectbox("Has Downtime", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prev_day_qty = st.number_input("Previous Day Quantity", min_value=0, max_value=100, value=0, step=1)
    
    if st.button("Predict Production", type="primary"):
        try:
            # Prepare input data matching your Excel features (8 features only)
            input_dict = {
                'Model': model_type,
                'Plan_Capacity': plan_capacity,
                'Manpower': manpower,
                'SHIFT': shift_value,
                'Total_Downtime': total_downtime,
                'Material_Issue': material_issue,
                'Has_downtime': has_downtime,
                'Prev_Day_Quantity': prev_day_qty
            }
            
            input_data = pd.DataFrame([input_dict])
            
            # Display input summary
            st.subheader("Input Summary")
            display_data = input_data.copy()
            display_data['SHIFT'] = display_data['SHIFT'].apply(lambda x: f"{x} ({'Full' if x == 1.0 else 'Half'})")
            display_data['Material_Issue'] = display_data['Material_Issue'].apply(lambda x: "Yes" if x == 1 else "No")
            display_data['Has_downtime'] = display_data['Has_downtime'].apply(lambda x: "Yes" if x == 1 else "No")
            st.dataframe(display_data, use_container_width=True, hide_index=True)
            
            # Make prediction
            if model is not None and scaler is not None:
                # One-hot encode Model if needed
                if 'Model' in feature_cols:
                    # Create dummy variables for Model
                    model_dummies = pd.get_dummies(input_data['Model'], prefix='Model')
                    input_data = pd.concat([input_data.drop('Model', axis=1), model_dummies], axis=1)
                
                # Ensure all required columns are present
                for col in feature_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                # Select only the features the model was trained on
                input_data = input_data[feature_cols]
                
                # Scale the features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction = max(0, int(round(prediction)))  # Ensure non-negative integer
                
                # Calculate metrics
                expected_vs_planned = (prediction / plan_capacity) * 100 if plan_capacity > 0 else 0
                
                # Estimate confidence based on downtime and material issues
                confidence_score = 95
                if material_issue == 1:
                    confidence_score -= 15
                if has_downtime == 1:
                    confidence_score -= 10
                if total_downtime > 200:
                    confidence_score -= 10
                confidence_score = max(confidence_score, 60)
                
            else:
                # Fallback prediction if model not loaded
                prediction = int(plan_capacity * 0.85) if material_issue == 0 else int(plan_capacity * 0.6)
                expected_vs_planned = (prediction / plan_capacity) * 100
                confidence_score = 75
            
            # Display prediction results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Predicted Actual Quantity", f"{prediction} units", 
                         delta=f"{prediction - plan_capacity} vs planned")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Expected vs Planned", f"{expected_vs_planned:.1f}%",
                         delta="On Target" if expected_vs_planned >= 90 else "Below Target")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence_score}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization 1: Planned vs Predicted
            st.markdown("---")
            st.subheader("Production Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=['Planned Capacity', 'Predicted Output'],
                    y=[plan_capacity, prediction],
                    text=[f'{plan_capacity} units', f'{prediction} units'],
                    textposition='outside',
                    marker_color=['#1f77b4', '#ff7f0e']
                ))
                fig1.update_layout(
                    title="Planned vs Predicted Production",
                    yaxis_title="Units",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Gauge chart for efficiency
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=expected_vs_planned,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Production Efficiency"},
                    delta={'reference': 100},
                    gauge={
                        'axis': {'range': [None, 120]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "lightyellow"},
                            {'range': [90, 120], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Impact Analysis
            st.subheader("Impact Factor Analysis")
            
            impact_data = pd.DataFrame({
                'Factor': ['Downtime', 'Material Issue', 'Manpower', 'Shift Type'],
                'Impact': [
                    -total_downtime * 0.05 if has_downtime else 0,
                    -15 if material_issue else 0,
                    manpower * 2,
                    20 if shift_value == 1.0 else 10
                ]
            })
            
            fig3 = px.bar(impact_data, x='Factor', y='Impact', 
                         title="Production Impact Factors",
                         color='Impact',
                         color_continuous_scale=['red', 'yellow', 'green'])
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("Recommendations")
            
            if prediction < plan_capacity:
                st.warning(f"Production shortfall expected: {plan_capacity - prediction} units")
                
                if material_issue == 1:
                    st.info("Action: Resolve material issues to improve output")
                if total_downtime > 100:
                    st.info(f"Action: Reduce downtime (currently {total_downtime} min) to increase production")
                if shift_value == 0.5:
                    st.info("Consider: Switch to full shift for higher output")
            else:
                st.success("Production target achievable!")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sample data from Excel
    st.markdown("---")
    st.subheader("Sample Historical Data")
    
    sample_data = pd.DataFrame({
        'Model': ['4KW', '4KW', '4KW', '11KW6P', '7.5KW', '4KW', '11KW', '7.5KW'],
        'Plan_Capacity': [16, 40, 26, 2, 9, 16, 9, 9],
        'Manpower': [6, 6, 4, 5, 4, 6, 5, 4],
        'SHIFT': [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5],
        'Total_Downtime': [149, 0, 0, 270, 0, 149, 450, 0],
        'Material_Issue': [1, 0, 0, 1, 0, 1, 1, 0],
        'Has_downtime': [1, 0, 0, 1, 0, 1, 1, 0],
        'Prev_Day_Qty': [0, 0, 0, 0, 0, 0, 0, 0],
        'Actual_Qty': [16, 40, 26, 2, 9, 16, 2, 9]
    })
    
    st.dataframe(sample_data, use_container_width=True, hide_index=True)
    
    # Historical trends
    st.subheader("Historical Production Trends")
    
    fig4 = px.scatter(sample_data, x='Plan_Capacity', y='Actual_Qty', 
                     color='Model', size='Total_Downtime',
                     title="Planned vs Actual Production by Model",
                     labels={'Plan_Capacity': 'Planned Capacity', 'Actual_Qty': 'Actual Quantity'})
    fig4.add_trace(go.Scatter(x=[0, 50], y=[0, 50], mode='lines', 
                              name='Perfect Match', line=dict(dash='dash', color='gray')))
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Mont Line OEE":
    st.header("Mont Line OEE Calculator")
    
    # Display formulas
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown("""
    **OEE Formulas:**
    - **Availability Rate (AR)** = (Loading time - Down time) / Loading time
    - **Performance Rate (PR)** = (Specified Cycle Time X Units produced) / Operating time
    - **Quality Rate (QR)** = (Units Produced - Defective units) / Unit Produced
    - **OEE** = AR * PR * QR * 100%
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="oee-section">', unsafe_allow_html=True)
    
    # Input parameters for Mont Line (with default values from your data)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Production Data")
        agreed_output = st.number_input("Agreed Output per Shift", min_value=1, value=45)
        days_worked = st.number_input("Days Worked", min_value=1, value=13)
        units_produced = st.number_input("Units Produced", min_value=0, value=475)
        defective_units = st.number_input("Defective Units", min_value=0, value=3)
    
    with col2:
        st.subheader("Time Data (minutes)")
        loading_time = st.number_input("Loading Time", min_value=1.0, value=5250.0)
        down_time = st.number_input("Down Time", min_value=0.0, value=50.0)
        agreed_cycle_time = st.number_input("Agreed Cycle Time (min)", min_value=0.1, value=11.05)
    
    with col3:
        st.subheader("Calculated Values")
        capacity_single_shift = agreed_output * days_worked
        st.metric("Capacity (Single Shift)", capacity_single_shift)
        operating_time = loading_time - down_time
        st.metric("Operating Time (min)", f"{operating_time:.2f}")
        credit_minutes = units_produced * agreed_cycle_time
        st.metric("Credit Minutes", f"{credit_minutes:.0f}")
    
    # Calculate OEE Components
    if st.button("Calculate OEE", type="primary"):
        # Calculate all components with FULL PRECISION
        ar = calculate_availability(loading_time, down_time)
        pr = calculate_performance(agreed_cycle_time, units_produced, operating_time)
        qr = calculate_quality(units_produced, defective_units)
        oee = calculate_oee(ar, pr, qr)
        
        # Display Results
        st.markdown("---")
        st.subheader("OEE Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Availability Rate (AR)", f"{ar:.2f}", f"{ar*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Performance Rate (PR)", f"{pr:.2f}", f"{pr*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Quality Rate (QR)", f"{qr:.2f}", f"{qr*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Overall OEE", f"{oee:.2f}%", 
                     delta=f"{oee - 99.37:.2f}%" if abs(oee - 99.37) > 0.01 else "Target Met")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Calculation Breakdown
        st.subheader("Calculation Breakdown")
        
        breakdown_df = pd.DataFrame({
            'Metric': ['Availability Rate', 'Performance Rate', 'Quality Rate', 'Overall OEE'],
            'Formula': [
                f'({loading_time:.2f} - {down_time:.2f}) / {loading_time:.2f}',
                f'({agreed_cycle_time:.2f} × {units_produced}) / {operating_time:.2f}',
                f'({units_produced} - {defective_units}) / {units_produced}',
                f'{ar:.6f} × {pr:.6f} × {qr:.6f} × 100'
            ],
            'Result': [f'{ar:.6f}', f'{pr:.6f}', f'{qr:.6f}', f'{oee:.2f}%'],
            'Percentage': [f'{ar*100:.2f}%', f'{pr*100:.2f}%', f'{qr*100:.2f}%', f'{oee:.2f}%']
        })
        
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        
        # OEE Breakdown Chart
        st.subheader("OEE Component Breakdown")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='OEE Components',
            x=['Availability', 'Performance', 'Quality', 'Overall OEE'],
            y=[ar*100, pr*100, qr*100, oee],
            text=[f'{ar*100:.2f}%', f'{pr*100:.2f}%', f'{qr*100:.2f}%', f'{oee:.2f}%'],
            textposition='outside',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ))
        
        fig.update_layout(
            title="Mont Line OEE Metrics",
            yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, max(ar*100, pr*100, qr*100, oee) + 10]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with Expected Values
        st.subheader("Comparison with Expected Mont Line Performance")
        
        expected_data = pd.DataFrame({
            'Metric': ['AR', 'PR', 'QR', 'OEE'],
            'Current': [ar*100, pr*100, qr*100, oee],
            'Expected': [99.05, 100.94, 99.37, 99.37]
        })
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Current', 
            x=expected_data['Metric'], 
            y=expected_data['Current'], 
            marker_color='#1f77b4',
            text=[f'{v:.2f}%' for v in expected_data['Current']],
            textposition='outside'
        ))
        fig2.add_trace(go.Bar(
            name='Expected', 
            x=expected_data['Metric'], 
            y=expected_data['Expected'], 
            marker_color='#2ca02c',
            text=[f'{v:.2f}%' for v in expected_data['Expected']],
            textposition='outside'
        ))
        
        fig2.update_layout(
            barmode='group', 
            height=400, 
            yaxis_title="Percentage (%)",
            title="Current vs Expected Performance (Mont Line)"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Performance Insights
        st.subheader("Performance Insights")
        
        if oee >= 99.0:
            st.success(f"Excellent performance! OEE of {oee:.2f}% meets or exceeds target of 99.37%")
        elif oee >= 90.0:
            st.warning(f"Good performance at {oee:.2f}%, but below Mont line target of 99.37%")
        else:
            st.error(f"OEE of {oee:.2f}% is significantly below target. Immediate action required.")
        
        # Recommendations
        st.subheader("Improvement Recommendations")
        
        if ar < 0.99:
            st.info(f"Availability: Current {ar*100:.2f}% vs Target 99.05%. Reduce downtime by {(0.9905-ar)*loading_time:.2f} minutes")
        if pr < 1.0094:
            st.info(f"Performance: Current {pr*100:.2f}% vs Target 100.94%. Optimize cycle time or increase production")
        if qr < 0.9937:
            st.info(f"Quality: Current {qr*100:.2f}% vs Target 99.37%. Reduce defects to maximum {units_produced*0.0063:.0f} units")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Montari ILine - Production Analytics System")