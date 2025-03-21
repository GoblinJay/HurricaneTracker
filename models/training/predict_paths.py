import os
import logging
import pandas as pd
import torch
import numpy as np
from models.storm_predictor import StormPredictor, StormDataset
from models.data_processor import DataProcessor
import joblib
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load the hurricane data from CSV."""
    try:
        data_path = 'data/processed/hurricane_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.error("Data file not found.")
            return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def load_saved_model(model_path='models/best_model.pth', scaler_path='models/best_scaler.joblib'):
    """Load the saved model and scaler."""
    try:
        # Load the scaler
        scaler = joblib.load(scaler_path)
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path)
        
        # Create model with same architecture
        model = StormPredictor(
            input_size=checkpoint['model_state_dict']['lstm.weight_ih_l0'].size(1),
            hidden_size=checkpoint['model_state_dict']['lstm.weight_hh_l0'].size(0),
            num_layers=len([k for k in checkpoint['model_state_dict'].keys() if 'lstm.weight_ih_l' in k])
        )
        
        # Load the model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("Successfully loaded model and scaler")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

def predict_storm_path(model, data_processor, storm_data, sequence_length=5):
    """Predict the path of a single storm."""
    predictions = []
    actuals = []
    
    # Prepare sequences
    features, targets = data_processor.prepare_sequences(storm_data)
    
    if len(features) == 0:
        logger.warning("No valid sequences found for this storm")
        return None, None
    
    # Create dataset and loader
    dataset = StormDataset(features, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Make predictions
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            
            # Store predictions and actuals
            predictions.append(outputs.cpu().numpy()[0])
            actuals.append(batch_targets.cpu().numpy()[0])
    
    return np.array(predictions), np.array(actuals)

def plot_storm_path(actual_path, predicted_path, storm_id, dates):
    """Plot the actual vs predicted path of a storm."""
    plt.figure(figsize=(12, 8))
    
    # Plot actual path
    plt.plot(actual_path[:, 1], actual_path[:, 0], 'b-', label='Actual Path')
    plt.scatter(actual_path[:, 1], actual_path[:, 0], c='blue', s=50)
    
    # Plot predicted path
    plt.plot(predicted_path[:, 1], predicted_path[:, 0], 'r--', label='Predicted Path')
    plt.scatter(predicted_path[:, 1], predicted_path[:, 0], c='red', s=50)
    
    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Storm {storm_id} Path (2021-2022)\nActual vs Predicted')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/storm_{storm_id}_path.png')
    plt.close()

def main():
    # Load the saved model and scaler
    model, scaler = load_saved_model()
    if model is None or scaler is None:
        return
    
    # Load the data
    df = load_data()
    if df is None:
        return
    
    # Filter for 2021-2022 storms
    test_df = df[
        (df['date'].dt.year >= 2021) &
        (df['date'].dt.year <= 2022)
    ]
    
    # Initialize data processor
    data_processor = DataProcessor()
    data_processor.scaler = scaler  # Use the loaded scaler
    
    # Process each storm
    for storm_id, storm_data in test_df.groupby('storm_id'):
        logger.info(f"Processing storm {storm_id}")
        
        # Keep only the base columns required for preprocessing
        base_features = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
        required_columns = ['date', 'storm_id'] + base_features
        storm_data = storm_data[required_columns].copy()
        
        # Engineer features
        processed_df = data_processor.engineer_features(storm_data)
        
        # Make predictions
        predictions, actuals = predict_storm_path(model, data_processor, processed_df)
        
        if predictions is not None and actuals is not None:
            # Plot the results
            plot_storm_path(actuals, predictions, storm_id, storm_data['date'])
            
            # Calculate and log errors
            lat_error = np.mean(np.abs(predictions[:, 0] - actuals[:, 0]))
            lon_error = np.mean(np.abs(predictions[:, 1] - actuals[:, 1]))
            wind_error = np.mean(np.abs(predictions[:, 2] - actuals[:, 2]))
            pressure_error = np.mean(np.abs(predictions[:, 3] - actuals[:, 3]))
            
            logger.info(f"Storm {storm_id} Results:")
            logger.info(f"Average Latitude Error: {lat_error:.4f} degrees")
            logger.info(f"Average Longitude Error: {lon_error:.4f} degrees")
            logger.info(f"Average Wind Speed Error: {wind_error:.4f} knots")
            logger.info(f"Average Pressure Error: {pressure_error:.4f} hPa")

if __name__ == "__main__":
    main() 