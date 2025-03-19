import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from typing import Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

class StormPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(StormPredictor, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Predict lat/long
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
    
    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Predict next position
        predictions = self.fc(last_output)
        return predictions
    
    def fit_scaler(self, data):
        """Fit the scaler on training data"""
        self.scaler.fit(data)
    
    def transform_data(self, data):
        """Transform data using fitted scaler"""
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform predictions back to original scale"""
        return self.scaler.inverse_transform(data)
        
    def generate_prediction(self, storm_data: pd.DataFrame, sequence_length: int = 5) -> Tuple[float, float]:
        """Generate next position prediction for a storm.
        
        Args:
            storm_data: DataFrame containing storm data
            sequence_length: Number of previous observations to use
            
        Returns:
            Tuple of (latitude, longitude) for predicted position
        """
        try:
            # Get last sequence_length observations
            last_sequence = storm_data.iloc[-sequence_length:].copy()
            
            # Scale the input data
            scaled_data = self.transform_data(last_sequence)
            
            # Reshape for model input (batch_size, sequence_length, features)
            model_input = scaled_data.reshape(1, sequence_length, -1)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self(model_input)
                prediction = prediction.cpu().numpy()
            
            # Reshape prediction to match expected shape
            prediction = prediction.reshape(-1, 2)  # Ensure shape is (1, 2)
            
            # Inverse transform to get actual coordinates
            prediction = self.inverse_transform(prediction)
            
            # Extract latitude and longitude
            lat, lon = prediction[0]
            
            # Validate predictions
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                logger.warning(f"Invalid prediction generated: lat={lat}, lon={lon}")
                return None
            
            return lat, lon
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return None
            
    def predict_storm_path(self, initial_sequence: pd.DataFrame, num_steps: int = 24) -> List[Tuple[float, float]]:
        """Generate multi-step predictions for a storm's path.
        
        Args:
            initial_sequence: DataFrame containing initial storm observations
            num_steps: Number of future positions to predict
            
        Returns:
            List of (latitude, longitude) tuples for predicted positions
        """
        try:
            predictions = []
            current_sequence = initial_sequence.copy()
            
            for _ in range(num_steps):
                # Generate next position prediction
                next_pos = self.generate_prediction(current_sequence)
                if next_pos is None:
                    break
                    
                predictions.append(next_pos)
                
                # Update sequence with prediction
                new_row = pd.DataFrame({
                    'latitude': [next_pos[0]],
                    'longitude': [next_pos[1]],
                    'max_wind': [current_sequence['max_wind'].iloc[-1]],  # Keep last known wind speed
                    'min_pressure': [current_sequence['min_pressure'].iloc[-1]],  # Keep last known pressure
                    'category': [current_sequence['category'].iloc[-1]]  # Keep last known category
                })
                
                current_sequence = pd.concat([current_sequence, new_row], ignore_index=True)
                current_sequence = current_sequence.iloc[-5:]  # Keep only last 5 observations
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting storm path: {str(e)}")
            return []

class StormDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, seq_length=5):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.targets[idx])
        )

def prepare_sequences(df, seq_length=5):
    """Prepare sequences for training"""
    features = []
    targets = []
    
    # Group by storm
    for _, storm_data in df.groupby('storm_id'):
        # Sort by date
        storm_data = storm_data.sort_values('date')
        
        # Extract features
        feature_cols = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
        X = storm_data[feature_cols].values
        
        # Create sequences
        for i in range(len(X) - seq_length):
            seq = X[i:i+seq_length]
            target = X[i+seq_length, :2]  # Next lat/long
            
            features.append(seq)
            targets.append(target)
    
    return np.array(features), np.array(targets)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        # Log progress
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model and scaler
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model state dict
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': model.scaler
            }, 'models/best_model.pth')

def save_model(model, scaler, path='models/best_model.pth'):
    """Save the trained model and scaler"""
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, path)
    logger.info(f"Saved model to {path}")

def load_model(path='models/best_model.pth'):
    """Load the trained model and scaler"""
    try:
        # Create model instance
        model = StormPredictor()
        
        # Load saved state
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.scaler = checkpoint['scaler']
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("Successfully loaded model and scaler")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('models/training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load and preprocess data
    df = pd.read_csv('data/processed/hurricane_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare features
    feature_cols = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
    X = df[feature_cols].values
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    sequence_length = 5
    X_sequences, y = create_sequences(X_scaled, sequence_length)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_sequences, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = StormDataset(X_train, y_train)
    val_dataset = StormDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = StormPredictor()
    model.scaler = scaler  # Attach scaler to model
    
    # Train model
    model = train_model(model, train_loader, val_loader)
    
    # Save final model
    save_model(model, scaler) 