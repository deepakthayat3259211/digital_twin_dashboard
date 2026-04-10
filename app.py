import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from predict import predict

st.set_page_config(page_title="CNC Milling Digital Twin Dashboard", layout="wide")

# Base directory (IMPORTANT for cloud)
BASE_DIR = os.path.dirname(__file__)

# Load data
@st.cache_data
def load_data():
    processed_path = os.path.join(BASE_DIR, 'processed_data.csv')
    raw_path = os.path.join(BASE_DIR, 'dataset_with_tool_life.csv')
    
    processed_data = pd.read_csv(processed_path)
    raw_data = pd.read_csv(raw_path)
    
    return processed_data, raw_data

processed_data, raw_data = load_data()

st.title("CNC Milling Machine Digital Twin Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Tool Life Prediction", "Data Overview", "Force Analysis", "Model Insights"])

# Model path (FIXED)
model_path = os.path.join(BASE_DIR, 'tool_life_model.pkl')

with tab1:
    st.header("Tool Life and Force Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        rpm = st.slider("RPM", 1000, 5000, 1000, step=1000)
        feed = st.slider("Feed (mm/sec)", 1.0, 6.0, 2.0)
        doc = st.slider("Depth of Cut (mm)", 1.0, 5.0, 1.0)
        
        if st.button("Predict", type="primary"):
            try:
                result = predict(model_path, rpm, feed, doc)
                
                st.success("Prediction completed!")
                
                tool_life = result['Tool_Life_HSS_min']
                st.metric("Estimated Tool Life", f"{tool_life:.2f} minutes")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Prediction Visualization")
        if 'result' in locals():
            tool_life = result['Tool_Life_HSS_min']
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=tool_life,
                title={'text': "Tool Life"}
            ))
            st.plotly_chart(fig)
