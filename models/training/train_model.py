import os
import logging
import pandas as pd
import torch
import torch.nn as nn
from models.storm_predictor import StormPredictor, load_model, StormDataset
from models.data_processor import DataProcessor
from config import MODEL_PARAMS
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data function
def load_data():
    try:
        data_path = 'data/processed/hurricane_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.warning("Local data not found.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def train_model(epochs=100, batch_size=32, learning_rate=0.0005, train_years=(1980, 2020)):
    # Load data
    df = load_data()
    if df.empty:
        logger.error("No data available for training.")
        return

    # Log the columns of the DataFrame
    logger.info(f"Loaded DataFrame columns: {df.columns.tolist()}")

    # Get only the data we need to avoid feature mismatch
    train_df = df[
        (df['date'].dt.year >= train_years[0]) &
        (df['date'].dt.year <= train_years[1])
    ]

    # Log the columns of the filtered DataFrame
    logger.info(f"Filtered DataFrame columns: {train_df.columns.tolist()}")
    print("DataFrame columns before feature engineering:", train_df.columns)

    # Initialize data processor
    data_processor = DataProcessor()
    
    # Step 1: Keep only the base columns required for preprocessing
    base_features = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
    required_columns = ['date', 'storm_id'] + base_features

    # Filter the DataFrame for base features only
    train_df = train_df[required_columns].copy()

    # Step 2: Engineer features (requires storm_id and date)
    processed_df = data_processor.engineer_features(train_df)

    # Step 3: Prepare sequences (requires storm_id for grouping)
    features, targets = data_processor.prepare_sequences(processed_df)

    # Check if we have any valid sequences
    if len(features) == 0:
        logger.error("No valid training sequences found. Make sure your data has enough sequential observations.")
        return
    
    # Log the shape of the features to determine input size
    logger.info(f"Feature shape: {features.shape}")
    input_size = features.shape[-1]  # Get the number of features from the shape
    
    # Step 4: Split the data into training and validation sets
    features_train, features_val, targets_train, targets_val = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = StormDataset(features_train, targets_train)
    val_dataset = StormDataset(features_val, targets_val)

    # Create data loaders with smaller batch size for better generalization
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create a new model with adjusted hyperparameters
    model = StormPredictor(
        input_size=input_size,  # Use the actual number of features
        hidden_size=MODEL_PARAMS['hidden_size'] * 2,  # Increased hidden size
        num_layers=MODEL_PARAMS['num_layers'] + 1  # Added one more layer
    )
    
    # Set up training with weight decay for regularization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # Added L2 regularization
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Early stopping parameters with longer patience
    patience = 15  # Increased from 5 to 15 epochs
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'models/best_model.pth'
    best_scaler_path = 'models/best_scaler.joblib'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Log progress with more detailed information
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                   f'Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
        
        # Early stopping check with longer patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model and scaler separately
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_model_path)
            joblib.dump(data_processor.scaler, best_scaler_path)
            logger.info(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. No improvement in {patience} epochs.")
                break
    
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # Load the best model and scaler for final evaluation
    if os.path.exists(best_model_path) and os.path.exists(best_scaler_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        data_processor.scaler = joblib.load(best_scaler_path)
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    return model, data_processor.scaler

def test_model(model, data_processor, test_years=(2021, 2022)):
    """Test the model on data from specified years."""
    # Load data
    df = load_data()
    if df.empty:
        logger.error("No data available for testing.")
        return

    # Filter for test years
    test_df = df[
        (df['date'].dt.year >= test_years[0]) &
        (df['date'].dt.year <= test_years[1])
    ]

    logger.info(f"Testing on data from {test_years[0]} to {test_years[1]}")
    logger.info(f"Number of test samples: {len(test_df)}")

    # Keep only the base columns required for preprocessing
    base_features = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
    required_columns = ['date', 'storm_id'] + base_features
    test_df = test_df[required_columns].copy()

    # Create a new DataProcessor instance for testing
    test_processor = DataProcessor()
    test_processor.scaler = data_processor  # Use the saved scaler

    # Engineer features using the same processor
    processed_df = test_processor.engineer_features(test_df)

    # Prepare sequences
    features, targets = test_processor.prepare_sequences(processed_df)

    if len(features) == 0:
        logger.error("No valid test sequences found.")
        return

    # Create test dataset and loader
    test_dataset = StormDataset(features, targets)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # Set up testing
    device = next(model.parameters()).device
    criterion = nn.MSELoss()

    # Test the model
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            test_loss += loss.item()

            # Store predictions and actuals
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_targets.cpu().numpy())

    # Calculate average test loss
    test_loss /= len(test_loader)

    # Convert predictions and actuals to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate additional metrics
    lat_error = np.mean(np.abs(predictions[:, 0] - actuals[:, 0]))
    lon_error = np.mean(np.abs(predictions[:, 1] - actuals[:, 1]))
    wind_error = np.mean(np.abs(predictions[:, 2] - actuals[:, 2]))
    pressure_error = np.mean(np.abs(predictions[:, 3] - actuals[:, 3]))

    logger.info(f"Test Results:")
    logger.info(f"Overall Test Loss: {test_loss:.4f}")
    logger.info(f"Average Latitude Error: {lat_error:.4f} degrees")
    logger.info(f"Average Longitude Error: {lon_error:.4f} degrees")
    logger.info(f"Average Wind Speed Error: {wind_error:.4f} knots")
    logger.info(f"Average Pressure Error: {pressure_error:.4f} hPa")

    return {
        'test_loss': test_loss,
        'lat_error': lat_error,
        'lon_error': lon_error,
        'wind_error': wind_error,
        'pressure_error': pressure_error,
        'predictions': predictions,
        'actuals': actuals
    }

if __name__ == "__main__":
    # Train the model
    model, scaler = train_model(epochs=100, batch_size=32, learning_rate=0.0005, train_years=(1980, 2020))
    
    # Test the model
    test_results = test_model(model, scaler, test_years=(2021, 2022))