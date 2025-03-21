import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from typing import Tuple, List, Optional
import os
from config import MODEL_PARAMS

# Set up logging
logger = logging.getLogger(__name__)

class StormPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4):
        super(StormPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize base features and scaler
        self.base_features = ['latitude', 'longitude', 'max_wind', 'min_pressure', 'category']
        self.scaler = None
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 4)  # Predict lat, lon, wind, pressure
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions of shape (batch_size, 4) for lat, lon, wind, pressure
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Final prediction
        output = self.fc(context)
        return output
    
    def fit_scaler(self, data):
        """Fit the scaler on training data"""
        self.scaler.fit(data)
    
    def transform_data(self, data):
        """Transform data using fitted scaler"""
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform predictions back to original scale"""
        return self.scaler.inverse_transform(data)
        
    def generate_prediction(self, storm_data: pd.DataFrame, sequence_length: int = None) -> Tuple[float, float]:
        """Generate next position prediction for a storm."""
        try:
            if sequence_length is None:
                sequence_length = MODEL_PARAMS['sequence_length']
                
            # Get last sequence_length observations
            last_sequence = storm_data.iloc[-sequence_length:].copy()
            
            # Ensure we have all required features
            for feature in self.base_features:
                if feature not in last_sequence.columns:
                    raise ValueError(f"Missing required feature: {feature}")
            
            # Select only the base features
            last_sequence = last_sequence[self.base_features]
            
            # Scale the input data
            if self.scaler is None:
                raise ValueError("Scaler not initialized. Please load or fit a scaler first.")
            scaled_data = self.scaler.transform(last_sequence)
            
            # Reshape for model input (batch_size, sequence_length, features)
            model_input = scaled_data.reshape(1, sequence_length, -1)
            model_input = torch.FloatTensor(model_input).to(next(self.parameters()).device)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self(model_input)
                prediction = prediction.cpu().numpy()
            
            # Reshape prediction to match expected shape
            prediction = prediction.reshape(-1, 4)  # Ensure shape is (1, 4)
            
            # Inverse transform to get actual coordinates
            prediction = self.scaler.inverse_transform(prediction)
            
            # Extract latitude and longitude
            lat, lon = prediction[0][:2]
            
            # Validate predictions
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError(f"Invalid prediction generated: lat={lat}, lon={lon}")
            
            return lat, lon
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise

    def predict_storm_path(self, initial_sequence: np.ndarray, num_steps: int = 24) -> np.ndarray:
        """Generate multi-step predictions for a storm's path."""
        try:
            predictions = []
            
            # Make a copy to avoid modifying the original
            current_sequence = initial_sequence.copy()
            
            # Ensure current_sequence has the right shape for LSTM input
            # LSTM expects (batch_size, seq_len, features)
            if len(current_sequence.shape) == 2:
                # If it's (seq_len, features), add batch dimension
                current_sequence = current_sequence.reshape(1, *current_sequence.shape)
            
            # Get reference to device
            device = next(self.parameters()).device
            
            for _ in range(num_steps):
                try:
                    # Create tensor for model input
                    model_input = torch.FloatTensor(current_sequence).to(device)
                    
                    # Generate prediction
                    with torch.no_grad():
                        # Model output is (batch_size, 4) for lat, lon, wind, pressure
                        prediction = self(model_input).cpu().numpy()
                    
                    # Extract latitude and longitude (first 2 values)
                    lat, lon = prediction[0][:2]
                    
                    # Validate predictions
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        logger.warning(f"Invalid prediction generated: lat={lat}, lon={lon}")
                        break
                    
                    # Add prediction to results
                    predictions.append([lat, lon])
                    
                    # Update sequence for next prediction
                    # Remove the first observation
                    next_sequence = current_sequence[0, 1:, :].copy()
                    
                    # Add new prediction as last observation
                    # Keep other features the same as the last observation
                    new_observation = current_sequence[0, -1, :].copy()
                    new_observation[0] = lat  # latitude
                    new_observation[1] = lon  # longitude
                    
                    # Append the new observation to the sequence
                    next_sequence = np.vstack((next_sequence, new_observation))
                    
                    # Reshape back to (1, seq_len, features)
                    current_sequence = next_sequence.reshape(1, *next_sequence.shape)
                
                except Exception as step_error:
                    logger.error(f"Error in prediction step: {str(step_error)}")
                    break
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting storm path: {str(e)}")
            return np.array([])

    def train_model(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        """Train the model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                outputs = self(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    outputs = self(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            # Log progress
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model and scaler
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(self, self.scaler)

class StormDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features: Input features of shape (n_samples, sequence_length, n_features)
            targets: Target values of shape (n_samples, 4) for lat, lon, wind, pressure
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

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
            target = X[i+seq_length, :4]  # Next lat, lon, wind, pressure
            
            features.append(seq)
            targets.append(target)
    
    return np.array(features), np.array(targets)

def save_model(model: StormPredictor, scaler: Optional[StandardScaler] = None, path: str = 'models/best_model.pth'):
    """Save the trained model and scaler"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state and scaler
        save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler if scaler is not None else model.scaler
        }
        
        torch.save(save_dict, path)
        logger.info(f"Saved model to {path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[StormPredictor, torch.device]:
    """Load a trained model from a checkpoint file."""
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Get exact dimensions from the checkpoint
        input_size = state_dict['lstm.weight_ih_l0'].shape[1]  # 15
        
        # For the bidirectional LSTM, the hidden size is 1/4 of weight_ih_l0's first dimension
        # In the checkpoint, lstm.weight_ih_l0 is [1024, 15] and the actual hidden_size is 256
        hidden_size = state_dict['lstm.weight_hh_l0'].shape[1]  # 256
        
        # Count number of layers
        num_layers = sum(1 for k in state_dict.keys() if 'weight_ih_l' in k and not 'reverse' in k)
        
        logger.info(f"Creating model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
        
        # Create a model with the EXACT same architecture
        model = StormPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Load the checkpoint
        model.load_state_dict(state_dict)
        if 'scaler' in checkpoint:
            model.scaler = checkpoint['scaler']
        
        model = model.to(device)
        model.eval()
        
        return model, device
        
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