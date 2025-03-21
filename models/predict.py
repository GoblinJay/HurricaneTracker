import torch
import logging
import pandas as pd
import numpy as np
from models.storm_predictor import StormPredictor, load_model
from models.data_processor import DataProcessor
import joblib
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path='models/best_model.pth', scaler_path='models/best_scaler.joblib'):
    """Load the trained model and scaler."""
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path)
        
        # Get input size from the model state dict
        state_dict = checkpoint['model_state_dict']
        input_size = state_dict['lstm.weight_ih_l0'].size(1)
        
        # Initialize model with correct parameters matching the training architecture
        model = StormPredictor(
            input_size=input_size,
            hidden_size=256,  # This matches MODEL_PARAMS['hidden_size'] * 2 from training
            num_layers=4      # This matches MODEL_PARAMS['num_layers'] + 1 from training
        )
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        # Load the scaler
        scaler = joblib.load(scaler_path)
        
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

def predict_path(initial_conditions, model, data_processor, num_predictions=5):
    """
    Predict the future path of a hurricane given initial conditions.
    
    Args:
        initial_conditions (dict): Dictionary containing initial storm conditions
            {
                'latitude': float,
                'longitude': float,
                'max_wind': float,
                'min_pressure': float,
                'category': int,
                'date': str (YYYY-MM-DD),
                'storm_id': str
            }
        model: Trained StormPredictor model
        data_processor: DataProcessor instance with fitted scaler
        num_predictions (int): Number of future predictions to make
    
    Returns:
        list: List of dictionaries containing predicted positions and conditions
    """
    try:
        # Create initial sequence data
        sequence_length = 5
        initial_df = pd.DataFrame([initial_conditions] * sequence_length)
        initial_df['date'] = pd.to_datetime(initial_df['date'])
        
        # Add hour offsets to dates to create a sequence
        for i in range(sequence_length):
            initial_df.iloc[i, initial_df.columns.get_loc('date')] += pd.Timedelta(hours=i*6)
        
        # Define feature order to match training
        feature_order = [
            'latitude', 'longitude', 'max_wind', 'min_pressure', 'category',
            'hour', 'day', 'month', 'year',
            'movement_speed', 'pressure_trend', 'wind_trend',
            'wind_pressure_ratio', 'distance_from_land', 'ocean_temperature'
        ]
        
        # Extract time-based features
        initial_df['hour'] = initial_df['date'].dt.hour
        initial_df['day'] = initial_df['date'].dt.day
        initial_df['month'] = initial_df['date'].dt.month
        initial_df['year'] = initial_df['date'].dt.year
        
        # Calculate movement speed (simplified for initial conditions)
        initial_df['movement_speed'] = 0.0  # Initialize with zero
        
        # Calculate wind-pressure ratio
        initial_df['wind_pressure_ratio'] = initial_df['max_wind'] / initial_df['min_pressure']
        
        # Initialize trends with zero
        initial_df['pressure_trend'] = 0.0
        initial_df['wind_trend'] = 0.0
        
        # Calculate distance from land (simplified)
        initial_df['distance_from_land'] = np.sqrt(
            (initial_df['latitude'] - 25)**2 +  # Approximate US coastline
            (initial_df['longitude'] + 80)**2
        )
        
        # Estimate ocean temperature based on latitude and month (simplified)
        initial_df['ocean_temperature'] = 30 - (initial_df['latitude'] - 25)**2 / 100
        
        # Ensure features are in the correct order
        features_df = initial_df[feature_order].copy()  # Create a copy to avoid warnings
        
        # Scale numerical features using the provided scaler
        scaled_features = data_processor.scaler.transform(features_df)
        
        # Create sequence manually
        sequence = np.zeros((1, sequence_length, len(feature_order)))
        for i in range(sequence_length):
            sequence[0, i] = scaled_features[i]
            
        # Convert to tensor
        device = next(model.parameters()).device
        current_sequence = torch.FloatTensor(sequence).to(device)
        
        predictions = []
        last_lat = initial_conditions['latitude']
        last_lon = initial_conditions['longitude']
        last_wind = initial_conditions['max_wind']
        last_pressure = initial_conditions['min_pressure']
        
        # Make predictions
        with torch.no_grad():
            for step in range(num_predictions):
                try:
                    # Get prediction
                    output = model(current_sequence)
                    pred = output.cpu().numpy()[0]  # Get the first (and only) prediction
                    
                    # Create a dummy array with all features to inverse transform
                    dummy_features = np.zeros(len(feature_order))
                    dummy_features[0] = pred[0]  # latitude
                    dummy_features[1] = pred[1]  # longitude
                    dummy_features[2] = pred[2]  # max_wind
                    dummy_features[3] = pred[3]  # min_pressure
                    
                    # Inverse transform to get actual values
                    unscaled_pred = data_processor.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
                    
                    # Create prediction dictionary with unscaled values
                    prediction = {
                        'latitude': float(unscaled_pred[0]),
                        'longitude': float(unscaled_pred[1]),
                        'max_wind': float(unscaled_pred[2]),
                        'min_pressure': float(unscaled_pred[3])
                    }
                    predictions.append(prediction)
                    
                    # Calculate new features for next prediction
                    movement_speed = np.sqrt(
                        (prediction['latitude'] - last_lat)**2 +
                        (prediction['longitude'] - last_lon)**2
                    )
                    pressure_trend = prediction['min_pressure'] - last_pressure
                    wind_trend = prediction['max_wind'] - last_wind
                    wind_pressure_ratio = prediction['max_wind'] / prediction['min_pressure']
                    distance_from_land = np.sqrt(
                        (prediction['latitude'] - 25)**2 + 
                        (prediction['longitude'] + 80)**2
                    )
                    ocean_temp = 30 - (prediction['latitude'] - 25)**2 / 100
                    
                    # Update last values
                    last_lat = prediction['latitude']
                    last_lon = prediction['longitude']
                    last_wind = prediction['max_wind']
                    last_pressure = prediction['min_pressure']
                    
                    # Create new feature vector with all features
                    new_features = np.array([[
                        prediction['latitude'],
                        prediction['longitude'],
                        prediction['max_wind'],
                        prediction['min_pressure'],
                        initial_conditions['category'],
                        current_sequence[0, -1, 5].cpu().numpy(),  # hour
                        current_sequence[0, -1, 6].cpu().numpy(),  # day
                        current_sequence[0, -1, 7].cpu().numpy(),  # month
                        current_sequence[0, -1, 8].cpu().numpy(),  # year
                        movement_speed,
                        pressure_trend,
                        wind_trend,
                        wind_pressure_ratio,
                        distance_from_land,
                        ocean_temp
                    ]])
                    
                    # Scale all features together
                    scaled_features = data_processor.scaler.transform(new_features)
                    
                    # Update sequence for next prediction
                    new_sequence = current_sequence.clone()
                    new_sequence[0, :-1] = new_sequence[0, 1:].clone()
                    new_sequence[0, -1] = torch.FloatTensor(scaled_features[0]).to(device)
                    current_sequence = new_sequence
                
                except Exception as step_error:
                    logger.error(f"Error in prediction step {step}: {str(step_error)}")
                    logger.error(f"Step traceback: {traceback.format_exc()}")
                    # Continue with next prediction if possible
                    continue
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def format_prediction(predictions):
    """Format the predictions for display."""
    formatted = []
    for i, pred in enumerate(predictions, 1):
        formatted.append(
            f"Step {i}:\n"
            f"  Position: ({pred['latitude']:.2f}°N, {pred['longitude']:.2f}°W)\n"
            f"  Wind Speed: {pred['max_wind']:.1f} knots\n"
            f"  Pressure: {pred['min_pressure']:.1f} hPa"
        )
    return "\n".join(formatted)

if __name__ == "__main__":
    # Example usage
    model, scaler = load_trained_model()
    if model is None or scaler is None:
        logger.error("Could not load model or scaler")
        exit(1)
    
    # Create data processor with loaded scaler
    data_processor = DataProcessor()
    data_processor.scaler = scaler
    
    # Test cases with different initial conditions
    test_cases = [
        {
            'name': 'Category 1 Hurricane near Florida',
            'conditions': {
                'latitude': 25.0,
                'longitude': -80.0,
                'max_wind': 75,
                'min_pressure': 985,
                'category': 1,
                'date': '2023-09-01',
                'storm_id': 'AL092023'
            }
        },
        {
            'name': 'Category 3 Hurricane in Gulf',
            'conditions': {
                'latitude': 28.0,
                'longitude': -90.0,
                'max_wind': 120,
                'min_pressure': 950,
                'category': 3,
                'date': '2023-09-01',
                'storm_id': 'AL092023'
            }
        },
        {
            'name': 'Category 5 Hurricane in Atlantic',
            'conditions': {
                'latitude': 20.0,
                'longitude': -60.0,
                'max_wind': 160,
                'min_pressure': 920,
                'category': 5,
                'date': '2023-09-01',
                'storm_id': 'AL092023'
            }
        }
    ]
    
    # Test each case
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}:")
        print("-" * 50)
        predictions = predict_path(test_case['conditions'], model, data_processor)
        
        if predictions:
            print(format_prediction(predictions))
        else:
            print("Could not generate predictions") 