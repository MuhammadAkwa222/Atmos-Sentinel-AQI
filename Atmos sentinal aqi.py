!pip install streamlit pandas scikit-learn plotly

%%writefile pak_aqi_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import datetime

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="AQI Pro: Pakistan Environmental Intelligence", layout="wide")

# Professional Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #e1e4e8; border-radius: 8px; background-color: white; padding: 10px; }
    div[data-testid="stMetricValue"] { color: #004a99; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE DATA ENGINE (100,000 ROWS) ---
@st.cache_resource
def build_intelligence_system():
    n = 100000
    np.random.seed(42)
    
    data = {
        'PM25': np.random.uniform(10, 500, n),
        'PM10': np.random.uniform(20, 650, n),
        'NO2':  np.random.uniform(5, 180, n),
        'SO2':  np.random.uniform(2, 140, n),
        'CO':   np.random.uniform(0.1, 15, n),
        'O3':   np.random.uniform(5, 220, n),
        'Temp': np.random.uniform(5, 45, n),
        'DewPoint': np.random.uniform(0, 32, n),
        'Mist_Intensity': np.random.uniform(0, 100, n) 
    }
    df = pd.DataFrame(data)
    
    # Pakistan Smog Logic: High PM2.5 + SO2 (Industrial) + Mist (Trapping factor)
    df['AQI'] = (df['PM25'] * 1.5) + (df['SO2'] * 1.2) + (df['CO'] * 2.0) + \
                (df['Mist_Intensity'] * 0.45) - (df['DewPoint'] * 0.15) + \
                np.random.normal(0, 3, n)
    
    X = df.drop('AQI', axis=1)
    y = df['AQI']
    
    model = RandomForestRegressor(n_estimators=15, max_depth=10, n_jobs=-1)
    model.fit(X, y)
    return model, df

model, full_data = build_intelligence_system()

# --- 2. SIDEBAR: PROFESSIONAL INPUTS ---
st.sidebar.title("Environmental Controls")
city = st.sidebar.selectbox("Metro Region", ["Lahore", "Karachi", "Faisalabad", "Islamabad", "Peshawar"])

st.sidebar.subheader("Chemical Sensors")
s_pm25 = st.sidebar.slider("PM2.5 (Fine Dust)", 0, 500, 160)
s_so2  = st.sidebar.slider("Sulfur Dioxide (SO2)", 0, 150, 45)
s_co   = st.sidebar.slider("Carbon Monoxide (CO)", 0.0, 20.0, 5.5)

st.sidebar.subheader("Atmospheric Data")
s_temp = st.sidebar.number_input("Air Temp (°C)", value=18)
s_dew  = st.sidebar.number_input("Dew Point (°C)", value=15)
s_mist = st.sidebar.select_slider("Mist/Fog Level", options=["Clear", "Mist", "Fog", "Dense Smog"])
mist_map = {"Clear": 0, "Mist": 30, "Fog": 70, "Dense Smog": 100}

# --- 3. MAIN DASHBOARD ---
st.title("Atmos Sentinel: AI-Integrated Environmental Intelligence")
st.markdown(f"**Station:** {city} Main Monitoring Node | **Dataset:** 100,000 Professional Synthetic Records")

# Prediction Logic
input_features = np.array([[s_pm25, s_pm25*1.4, 45, s_so2, s_co, 60, s_temp, s_dew, mist_map[s_mist]]])
prediction = int(model.predict(input_features)[0])

# Metric Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted AQI", prediction)
c2.metric("Temp/Dew Gap", f"{abs(s_temp - s_dew)}°C")
c3.metric("Visibility", f"{100 - mist_map[s_mist]}%")
c4.metric("Risk Level", "Hazardous" if prediction > 300 else "Critical" if prediction > 200 else "Moderate")

st.divider()

# --- 4. DATA VISUALIZATION TOPICS ---
tab1, tab2 = st.tabs(["Atmospheric Correlations", "24h Smog Cycle Simulation"])

with tab1:
    st.subheader("Chemical & Atmospheric Relationship")
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.scatter(full_data.sample(1000), x="PM25", y="AQI", color="Mist_Intensity",
                         size="SO2", title="PM2.5 vs AQI (Color: Mist, Size: Sulfur)",
                         color_continuous_scale="Reds", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.density_heatmap(full_data.sample(2000), x="Temp", y="DewPoint", z="AQI",
                                 title="The 'Smog Trap' (Temp vs Dew Point Heatmap)",
                                 template="plotly_white", nbinsx=20, nbinsy=20)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Simulated Daily Smog Profile")
    hours = list(range(24))
    hourly_aqi = [prediction + (50 * np.sin((h-14)*np.pi/12)) + np.random.randint(-10, 10) for h in hours]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=hours, y=hourly_aqi, mode='lines+markers', 
                             line=dict(color='#004a99', width=3),
                             fill='tozeroy', name="Projected AQI"))
    fig3.update_layout(title=f"Predicted 24-Hour AQI Trend for {city}",
                      xaxis_title="Hour of Day", yaxis_title="AQI Value",
                      template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    st.info("Note: In Pakistan, AQI usually rises at night due to 'Temperature Inversion' where cold air traps industrial smoke.")

# --- 5. PROFESSIONAL REPORT ---
st.subheader("Professional Assessment")
st.write(f"Analysis for **{city}** suggests that at **{s_temp}°C** with **{s_mist}** conditions, pollutants will remain trapped near the ground.")