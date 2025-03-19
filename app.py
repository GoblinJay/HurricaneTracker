import streamlit as st

st.set_page_config(
    page_title="Hurricane Tracker",
    page_icon="🌀",
    layout="wide"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Show loading message
with st.spinner('Loading required packages...'):
    try:
        # Basic imports first
        import numpy as np
        import pandas as pd
        import plotly.express as px
        
        # Then try ML imports
        import torch
        from models.storm_predictor import StormPredictor, load_model
        
        # Finally visualization
        import folium
        from streamlit_folium import st_folium
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        import logging
        
        st.session_state.model_loaded = True
        
    except ImportError as e:
        st.error(f"""
        ⚠️ Error loading required packages:
        ```python
        {str(e)}
        ```
        Please check the installation and try again.
        """)
        st.stop()

# Set up logging with error handling
try:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
except Exception as e:
    st.warning(f"Logging setup failed: {str(e)}")

# Load data
@st.cache_data
def load_data():
    try:
        # First try loading from local file
        data_path = 'data/processed/hurricane_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Fallback to URL or generate data
            st.warning("Local data not found, loading from backup source...")
            # Add your fallback data loading logic here
            
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load model
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model()
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Error loading model: {str(e)}")
        return None

def plot_storm_track(storm_data, predictions=None):
    """Create an interactive map with storm track"""
    # Create map centered on storm
    center_lat = storm_data['latitude'].mean()
    center_lon = storm_data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    # Plot actual track
    points = list(zip(storm_data['latitude'], storm_data['longitude']))
    folium.PolyLine(points, color='blue', weight=2, opacity=0.8).add_to(m)
    
    # Add markers for start and end
    folium.Marker(
        points[0],
        popup='Start',
        icon=folium.Icon(color='green')
    ).add_to(m)
    
    folium.Marker(
        points[-1],
        popup='End',
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    # Plot predictions if available
    if predictions is not None:
        pred_points = list(zip(predictions[:, 0], predictions[:, 1]))
        folium.PolyLine(pred_points, color='red', weight=2, opacity=0.8, dash_array='5').add_to(m)
    
    return m

def plot_intensity_timeline(storm_data):
    """Create a plot showing wind speed and pressure over time"""
    fig = go.Figure()
    
    # Add wind speed trace
    fig.add_trace(go.Scatter(
        x=storm_data['date'],
        y=storm_data['max_wind'],
        name='Wind Speed',
        line=dict(color='blue')
    ))
    
    # Add pressure trace
    fig.add_trace(go.Scatter(
        x=storm_data['date'],
        y=storm_data['min_pressure'],
        name='Pressure',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title='Storm Intensity Over Time',
        xaxis_title='Date',
        yaxis_title='Wind Speed (knots)',
        yaxis2=dict(
            title='Pressure (mb)',
            overlaying='y',
            side='right'
        ),
        height=400
    )
    
    return fig

def main():
    st.title('Hurricane Analysis Dashboard')
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header('Filters')
    
    # Year range filter
    min_year = df['date'].dt.year.min()
    max_year = df['date'].dt.year.max()
    year_range = st.sidebar.slider(
        'Year Range',
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Category filter
    categories = sorted(df['category'].unique())
    selected_categories = st.sidebar.multiselect(
        'Storm Categories',
        categories,
        default=categories
    )
    
    # Apply filters
    filtered_df = df[
        (df['date'].dt.year >= year_range[0]) &
        (df['date'].dt.year <= year_range[1]) &
        (df['category'].isin(selected_categories))
    ]
    
    # Storm selection
    storms = filtered_df[['storm_id', 'storm_name']].drop_duplicates()
    storms['display_name'] = storms['storm_name'] + ' (' + storms['storm_id'] + ')'
    selected_storm = st.selectbox('Select Storm', storms['storm_id'], format_func=lambda x: storms[storms['storm_id'] == x]['display_name'].iloc[0])
    
    if selected_storm:
        storm_data = df[df['storm_id'] == selected_storm].sort_values('date')
        
        # Display storm details
        st.subheader(f"Storm Details: {storm_data['storm_name'].iloc[0]}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Duration', f"{(storm_data['date'].max() - storm_data['date'].min()).days} days")
        
        with col2:
            st.metric('Max Wind Speed', f"{storm_data['max_wind'].max():.1f} kt")
        
        with col3:
            st.metric('Min Pressure', f"{storm_data['min_pressure'].min():.1f} mb")
        
        # Plot storm track
        st.subheader('Storm Track')
        m = plot_storm_track(storm_data)
        st_folium(m)
        
        # Plot intensity timeline
        st.subheader('Storm Intensity')
        fig = plot_intensity_timeline(storm_data)
        st.plotly_chart(fig)
        
        # Model predictions
        st.subheader('Path Prediction')
        if st.button('Generate Prediction'):
            model = load_prediction_model()
            if model is not None:
                try:
                    # Prepare sequence
                    feature_cols = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
                    sequence = storm_data[feature_cols].values[-5:]  # Last 5 observations
                    
                    # Get prediction
                    predictions = model.predict_storm_path(sequence)
                    
                    # Plot with predictions
                    m = plot_storm_track(storm_data, predictions)
                    st_folium(m)
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
            else:
                st.warning("Model not available. Please train the model first.")
    
    # Statistical analysis
    st.header('Statistical Analysis')
    
    # Category distribution
    st.subheader('Storm Category Distribution')
    category_counts = filtered_df['category'].value_counts().sort_index()
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title='Distribution of Storm Categories'
    )
    st.plotly_chart(fig)
    
    # Wind speed distribution
    st.subheader('Wind Speed Distribution')
    fig = px.histogram(
        filtered_df,
        x='max_wind',
        title='Distribution of Maximum Wind Speeds'
    )
    st.plotly_chart(fig)

if __name__ == '__main__':
    main() 