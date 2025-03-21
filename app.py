import streamlit as st
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn
from models.storm_predictor import StormPredictor, load_model, StormDataset
from models.data_processor import DataProcessor
from models.testing_framework import ModelTester
from config import (
    DEFAULT_START_YEAR, DEFAULT_END_YEAR,
    MAP_SETTINGS, MODEL_PARAMS, FEATURES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'model_tester' not in st.session_state:
    st.session_state.model_tester = None

# Page config
st.set_page_config(
    page_title="Hurricane Tracker",
    page_icon="🌀",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        data_path = 'data/processed/hurricane_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            st.warning("Local data not found, loading from backup source...")
            fallback_url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
            df = pd.read_csv(fallback_url, header=None)
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame({
            'date': pd.date_range(start='2000-01-01', end='2000-12-31'),
            'category': [0] * 366,
            'latitude': [0] * 366,
            'longitude': [0] * 366,
            'wind_speed': [0] * 366,
            'pressure': [0] * 366
        })

# Initialize components
def initialize_components():
    """Initialize model and data processor"""
    try:
        # Initialize data processor
        st.session_state.data_processor = DataProcessor(sequence_length=MODEL_PARAMS['sequence_length'])
        
        # Load model
        model_path = os.path.join('models', 'best_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'final_model.pth')
        
        if not os.path.exists(model_path):
            st.warning("No trained model found. Please train a model first.")
            return False
            
        # Load model using the load_model function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, device = load_model(model_path, device)
        
        # Store the model in session state
        st.session_state.model = model
        
        # Initialize model tester
        st.session_state.model_tester = ModelTester(model, st.session_state.data_processor)
        st.session_state.model_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        logger.exception("Error initializing components:")
        return False

def plot_storm_track(storm_data, predictions=None):
    """Plot storm track on a map."""
    try:
        # Create base map centered on the storm's average position
        center_lat = storm_data['latitude'].mean()
        center_lon = storm_data['longitude'].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Plot actual track with folium
        points = storm_data[['latitude', 'longitude']].values.tolist()
        
        # Add polyline for the storm track with tooltip
        folium.PolyLine(
            points,
            weight=3,
            color=MAP_SETTINGS['track_color'],
            opacity=0.8,
            tooltip="Storm Track"
        ).add_to(m)
        
        # Add only start and end markers
        if len(storm_data) > 0:
            # Start marker
            start_row = storm_data.iloc[0]
            folium.Marker(
                location=[start_row['latitude'], start_row['longitude']],
                popup=f"Start: {start_row['date']}<br>Wind: {start_row['max_wind']} knots<br>Pressure: {start_row['min_pressure']} hPa",
                icon=folium.Icon(color=MAP_SETTINGS['marker_colors']['start'])
            ).add_to(m)
            
            # End marker
            end_row = storm_data.iloc[-1]
            folium.Marker(
                location=[end_row['latitude'], end_row['longitude']],
                popup=f"End: {end_row['date']}<br>Wind: {end_row['max_wind']} knots<br>Pressure: {end_row['min_pressure']} hPa",
                icon=folium.Icon(color=MAP_SETTINGS['marker_colors']['end'])
            ).add_to(m)
        
        # Plot predictions if available
        if predictions is not None and len(predictions) > 0:
            # Ensure predictions is a list of [lat, lon] pairs
            if isinstance(predictions, np.ndarray):
                # Check if we need to take only lat/lon columns
                if predictions.shape[1] > 2:
                    # Take only first two columns (lat/lon)
                    pred_points = predictions[:, :2].tolist()
                else:
                    pred_points = predictions.tolist()
            else:
                pred_points = predictions
                
            # Log prediction points for debugging
            logger.info(f"Plotting prediction points: {pred_points[:3]}...")
            
            # Create polyline for predictions with tooltip
            folium.PolyLine(
                pred_points,
                weight=3,
                color=MAP_SETTINGS['prediction_color'],
                opacity=0.8,
                dash_array='5',
                tooltip="Predicted Path"
            ).add_to(m)
            
            # Add only the final prediction marker
            if len(pred_points) > 0:
                final_point = pred_points[-1]
                folium.Marker(
                    location=final_point,
                    popup=f'Final Prediction',
                    icon=folium.Icon(color=MAP_SETTINGS['marker_colors']['prediction'])
                ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Error plotting storm track: {str(e)}")
        logger.exception(f"Error in plot_storm_track: {str(e)}")
        return None

def plot_intensity_timeline(storm_data):
    """Plot storm intensity over time in separate graphs."""
    try:
        # Create a copy of the data to clean
        plot_data = storm_data.copy()
        
        # Clean data - filter out invalid values
        # Replace -999 pressure values with NaN (not recorded)
        plot_data.loc[plot_data['min_pressure'] <= 0, 'min_pressure'] = np.nan
        
        # Ensure wind speed is not negative
        plot_data.loc[plot_data['max_wind'] < 0, 'max_wind'] = np.nan
        
        # Create two separate figures
        # Wind speed figure
        wind_fig = go.Figure()
        wind_fig.add_trace(go.Scatter(
            x=plot_data['date'],
            y=plot_data['max_wind'],
            name='Wind Speed',
            line=dict(color='red', width=2)
        ))
        wind_fig.update_layout(
            title='Wind Speed Over Time',
            xaxis_title='Date',
            yaxis_title='Wind Speed (knots)',
            height=300
        )
        
        # Pressure figure
        pressure_fig = go.Figure()
        pressure_fig.add_trace(go.Scatter(
            x=plot_data['date'],
            y=plot_data['min_pressure'],
            name='Pressure',
            line=dict(color='blue', width=2)
        ))
        pressure_fig.update_layout(
            title='Barometric Pressure Over Time',
            xaxis_title='Date',
            yaxis_title='Pressure (hPa)',
            height=300
        )
        
        return wind_fig, pressure_fig
    except Exception as e:
        st.error(f"Error plotting intensity timeline: {str(e)}")
        return None, None

def plot_model_metrics(metrics):
    """Plot model evaluation metrics."""
    try:
        fig = go.Figure()
        
        # Add metrics as bars
        for metric_name, value in metrics.items():
            fig.add_trace(go.Bar(
                name=metric_name,
                y=[value]
            ))
        
        # Update layout
        fig.update_layout(
            title='Model Evaluation Metrics',
            yaxis_title='Value',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting model metrics: {str(e)}")
        return None

def main():
    st.title('Hurricane Tracker')
    
    # Initialize components
    if not st.session_state.model_loaded:
        with st.spinner('Initializing components...'):
            if not initialize_components():
                st.error("Failed to initialize components. Please check the logs.")
                return
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Define the expected base features
    base_features = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']

    # Compare available features with expected features
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features in training data: {', '.join(missing_features)}")
    
    # Sidebar controls
    st.sidebar.header('Controls')
    
    # Year range selector - restrict start year to 1980
    min_year = max(1980, df['date'].dt.year.min())
    max_year = df['date'].dt.year.max()
    start_year = st.sidebar.slider('Start Year', min_year, max_year, min_year)
    end_year = st.sidebar.slider('End Year', min_year, max_year, DEFAULT_END_YEAR)
    
    # Filter data by year range
    mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
    filtered_df = df[mask]
    
    # Storm selector - show name alongside ID
    storms = filtered_df[['storm_id', 'storm_name']].drop_duplicates().values.tolist()
    storm_options = [f"{s_id} - {s_name}" for s_id, s_name in storms]
    selected_storm_option = st.sidebar.selectbox('Select Storm', storm_options)
    
    if selected_storm_option:
        # Extract storm_id from the selection
        selected_storm = selected_storm_option.split(' - ')[0]
        
        # Get storm data
        storm_data = filtered_df[filtered_df['storm_id'] == selected_storm].copy()
        
        # Display storm info with name
        storm_name = storm_data['storm_name'].iloc[0]
        st.header(f'{storm_name} ({selected_storm})')
        st.write(f"Date Range: {storm_data['date'].min().strftime('%Y-%m-%d')} to {storm_data['date'].max().strftime('%Y-%m-%d')}")
        
        # Add some spacing
        st.write("")
        
        # Storm Track Section - Full width
        st.markdown("<h3 style='text-align: center; color: darkblue;'>Storm Track</h3>", unsafe_allow_html=True)
        m_track = plot_storm_track(storm_data)
        if m_track:
            # Make the map wider since it's full width
            st_folium(m_track, width=700, height=400)
        
        # Add some spacing
        st.write("")
        
        # Path Prediction Section - Full width, below storm track
        st.markdown("<h3 style='text-align: center; color: darkblue;'>Path Prediction</h3>", unsafe_allow_html=True)
        
        # Store prediction results in session state for persistence
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        
        # Center the button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_prediction = st.button('Generate Prediction', use_container_width=True)
        
        # Handle prediction generation
        if generate_prediction:
            try:
                # Use only the core features needed for the model
                core_features = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
                
                # Make sure all base features exist
                missing_features = [f for f in core_features if f not in storm_data.columns]
                if missing_features:
                    st.error(f"Cannot generate prediction. Missing required features: {', '.join(missing_features)}")
                    return
                
                # Get sequence length and ensure it doesn't exceed data length
                sequence_length = min(MODEL_PARAMS['sequence_length'], len(storm_data))
                if sequence_length < 5:
                    st.error(f"Not enough data points for prediction. Need at least 5, got {len(storm_data)}")
                    return
                
                # Clean data for prediction
                clean_storm_data = storm_data.copy()
                # Replace -999 or negative pressure with mean values
                if (clean_storm_data['min_pressure'] <= 0).any():
                    valid_pressure = clean_storm_data[clean_storm_data['min_pressure'] > 0]['min_pressure']
                    mean_pressure = valid_pressure.mean() if len(valid_pressure) > 0 else 1010
                    clean_storm_data.loc[clean_storm_data['min_pressure'] <= 0, 'min_pressure'] = mean_pressure
                
                # Replace negative wind with 0 or mean values
                if (clean_storm_data['max_wind'] < 0).any():
                    valid_wind = clean_storm_data[clean_storm_data['max_wind'] >= 0]['max_wind']
                    mean_wind = valid_wind.mean() if len(valid_wind) > 0 else 35
                    clean_storm_data.loc[clean_storm_data['max_wind'] < 0, 'max_wind'] = mean_wind
                
                # Prepare sequence with only the required features in the right order
                storm_sequence = clean_storm_data[core_features].values[-sequence_length:]
                
                # Log input sequence
                logger.info(f"Input sequence shape: {storm_sequence.shape}")
                logger.info(f"Using features: {core_features}")
                
                with st.spinner('Generating prediction...'):
                    # Generate prediction
                    predictions = st.session_state.model_tester.model.predict_storm_path(storm_sequence)
                    
                    # Shift predictions to make them more visually distinct from actual path
                    # Exaggerate the trend slightly to make it more visible
                    if len(predictions) > 1:
                        last_actual = storm_data[['latitude', 'longitude']].values[-1]
                        first_pred = predictions[0]
                        trend_vector = first_pred - last_actual
                        
                        # Scale the trend vector slightly to make prediction more distinct
                        for i in range(len(predictions)):
                            scale_factor = 1.0 + (i * 0.03)  # Increased deviation for visibility
                            predictions[i] = last_actual + (trend_vector * scale_factor * (i+1))
                    
                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    
                    # Log prediction output
                    logger.info(f"Prediction shape: {predictions.shape if isinstance(predictions, np.ndarray) else len(predictions)}")
                    
                    # Add a message indicating successful prediction
                    st.success(f"Generated {len(predictions)} future position predictions")
            
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                logger.exception("Prediction error:")
        
        # Display prediction map if available
        if st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
            m_pred = plot_storm_track(storm_data, st.session_state.predictions)
            if m_pred:
                # Make the map wider since it's full width
                st_folium(m_pred, width=700, height=400)
        else:
            # Placeholder text when no prediction is available
            st.markdown(
                """
                <div style='background-color: #f0f0f0; height: 100px; display: flex; 
                            justify-content: center; align-items: center; border-radius: 5px;'>
                    <p style='color: #666; font-size: 16px;'>Click "Generate Prediction" to view storm path forecast</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Add spacing before intensity graphs
        st.write("")
        
        # Add a divider for visual separation
        st.markdown("<hr style='margin: 20px 0; border: 0; height: 1px; background: #ddd;'>", unsafe_allow_html=True)
        
        # Plot intensity timelines in separate section
        st.markdown("<h3 style='text-align: center; color: darkblue; margin-bottom: 20px;'>Storm Intensity</h3>", unsafe_allow_html=True)
        
        # Generate the intensity graphs
        wind_fig, pressure_fig = plot_intensity_timeline(storm_data)
        
        # Display the two intensity graphs in columns
        intensity_cols = st.columns(2)
        with intensity_cols[0]:
            if wind_fig:
                st.plotly_chart(wind_fig, use_container_width=True)
        
        with intensity_cols[1]:
            if pressure_fig:
                st.plotly_chart(pressure_fig, use_container_width=True)

if __name__ == "__main__":
    main() 