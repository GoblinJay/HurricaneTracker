import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import folium
    from streamlit_folium import st_folium
    import torch
    from models.storm_predictor import StormPredictor
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
except ImportError as e:
    st.error(f"Failed to import required packages. Please check requirements.txt: {str(e)}")
    logger.error(f"Import error: {str(e)}")
    st.stop()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/hurricane_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

# Page config
st.set_page_config(
    page_title="Hurricane Tracker",
    page_icon="🌀",
    layout="wide"
)

# Main title
st.title("Hurricane Analysis and Prediction System")
st.write("Track and analyze hurricane patterns using historical data and machine learning predictions")

# Sidebar filters
st.sidebar.header("Filters")

# Load the data
try:
    df = load_data()
    
    # Year range filter
    years = df['date'].dt.year.unique()
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(years.min()),
        max_value=int(years.max()),
        value=(int(years.min()), int(years.max()))
    )
    
    # Filter data by year
    mask = (df['date'].dt.year >= year_range[0]) & (df['date'].dt.year <= year_range[1])
    filtered_df = df[mask]
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Hurricane Tracks")
        fig = px.line_mapbox(filtered_df,
                            lat='latitude',
                            lon='longitude',
                            color='storm_id',
                            zoom=3,
                            mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Storm Statistics")
        st.metric("Total Storms", len(filtered_df['storm_id'].unique()))
        st.metric("Average Wind Speed", f"{filtered_df['max_wind'].mean():.1f} knots")
        st.metric("Average Pressure", f"{filtered_df['min_pressure'].mean():.1f} mb")
        
        # Storm category distribution
        st.subheader("Category Distribution")
        category_counts = filtered_df['category'].value_counts().sort_index()
        fig = px.bar(x=category_counts.index,
                     y=category_counts.values,
                     labels={'x': 'Category', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    logger.error(f"Error in app: {str(e)}") 