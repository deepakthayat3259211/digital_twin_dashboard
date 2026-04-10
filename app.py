import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Ensure current directory is used
sys.path.append(os.path.dirname(__file__))

from predict import predict

st.set_page_config(page_title="CNC Milling Digital Twin Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    processed_data = pd.read_csv("processed_data.csv")
    raw_data = pd.read_csv("dataset_with_tool_life.csv")
    return processed_data, raw_data

processed_data, raw_data = load_data()

st.title("CNC Milling Machine Digital Twin Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Tool Life Prediction", "Data Overview", "Force Analysis", "Model Insights"])

# Model path (same folder)
model_path = "tool_life_model.pkl"

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
                
                st.subheader("Predicted Results")
                
                # Tool Life
                st.markdown("### Tool Life Prediction")
                tool_life = result['Tool_Life_HSS_min']
                st.metric(
                    "Estimated Tool Life",
                    f"{tool_life:.2f} minutes",
                    delta=f"{'Good' if tool_life > 30 else 'Monitor' if tool_life > 15 else 'Replace Soon'}",
                    delta_color="normal"
                )
                
                # Forces
                st.markdown("### Maximum Cutting Forces")
                force_values = [
                    abs(result['Max_Force_X']),
                    abs(result['Max_Force_Y']),
                    abs(result['Max_Force_Z'])
                ]
                force_labels = ["X Direction (N)", "Y Direction (N)", "Z Direction (N)"]
                force_cols = st.columns(3)
                
                for col, label, value in zip(force_cols, force_labels, force_values):
                    with col:
                        st.markdown(f"**{label}**")
                        st.markdown(f"# {value:.2f}")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    with col2:
        st.subheader("Prediction Visualization")
        if 'result' in locals():
            tool_life = result['Tool_Life_HSS_min']

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=tool_life,
                title={'text': "Tool Life (minutes)"},
                gauge={
                    'axis': {'range': [0, 60]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "red"},
                        {'range': [15, 30], 'color': "orange"},
                        {'range': [30, 60], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'value': tool_life
                    }
                }
            ))
            st.plotly_chart(fig_gauge)

            forces = [
                abs(result['Max_Force_X']),
                abs(result['Max_Force_Y']),
                abs(result['Max_Force_Z'])
            ]

            fig_bar = go.Figure(data=go.Bar(
                x=['X', 'Y', 'Z'],
                y=forces,
                text=[f"{v:.1f} N" for v in forces],
                textposition='auto'
            ))

            fig_bar.update_layout(
                title="Predicted Maximum Force Magnitudes",
                yaxis_title="Force Magnitude (N)",
                xaxis_title="Direction"
            )

            st.plotly_chart(fig_bar)

with tab2:
    st.header("Data Overview")
    
    st.subheader("Processed Dataset Summary")
    st.dataframe(processed_data.describe())
    
    st.subheader("Feature Correlations")
    corr = processed_data[['RPM', 'Feed_mm_per_sec', 'DOC_mm',
                           'Tool_Life_HSS_min', 'Max_Force_X',
                           'Max_Force_Y', 'Max_Force_Z']].corr()
    
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig)
    
    st.subheader("Scatter Plots")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(processed_data, x='RPM', y='Tool_Life_HSS_min')
        st.plotly_chart(fig)
        
    with col2:
        fig = px.scatter(processed_data, x='DOC_mm', y='Max_Force_X')
        st.plotly_chart(fig)

with tab3:
    st.header("Force Analysis")
    
    sequence = st.selectbox("Select Milling Sequence", processed_data['sequence_id'].unique())
    
    seq_data = raw_data[raw_data['sequence_id'] == sequence]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (X) [N]'], name='Force X'))
    fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (Y) [N]'], name='Force Y'))
    fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (Z) [N]'], name='Force Z'))
    
    st.plotly_chart(fig)
    
    params = processed_data[processed_data['sequence_id'] == sequence][
        ['RPM', 'Feed_mm_per_sec', 'DOC_mm', 'Tool_Life_HSS_min']
    ].iloc[0]
    
    st.write(params)

with tab4:
    st.header("Model Insights")
    
    metrics = {
        'Target': ['Tool Life', 'Max Force X', 'Max Force Y', 'Max Force Z'],
        'R² Score': [0.997, 0.784, 0.747, 0.565]
    }
    
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    
    fig = px.bar(metrics_df, x='Target', y='R² Score')
    st.plotly_chart(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("**Digital Twin Features:**")
st.sidebar.markdown("- Predictive tool life estimation")
st.sidebar.markdown("- Force monitoring and analysis")
st.sidebar.markdown("- Data-driven insights")
st.sidebar.markdown("- Interactive visualizations")
