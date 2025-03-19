import torch
import torch.nn as nn
import torch.nn.functional as F

class HurricaneCNN(nn.Module):
    """CNN backbone for processing satellite imagery"""
    def __init__(self, in_channels=3):
        super(HurricaneCNN, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolution block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third convolution block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        return x

class HurricanePredictor(nn.Module):
    """Main model combining CNN and LSTM for hurricane prediction"""
    def __init__(self, 
                 in_channels=3,
                 lstm_hidden_size=256,
                 num_lstm_layers=2,
                 num_classes=5,  # For classification (hurricane categories)
                 predict_track=True):  # Whether to predict track coordinates
        super(HurricanePredictor, self).__init__()
        
        # CNN backbone
        self.cnn = HurricaneCNN(in_channels)
        
        # Calculate CNN output size
        self.cnn_output_size = 256 * 8 * 8  # Adjust based on input size
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.5 if num_lstm_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Track prediction head (if enabled)
        self.predict_track = predict_track
        if predict_track:
            self.track_predictor = nn.Sequential(
                nn.Linear(lstm_hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)  # Predict lat/long coordinates
            )
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape for CNN processing
        x = x.view(-1, channels, height, width)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)
        
        # Reshape for LSTM
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last LSTM output for predictions
        last_hidden = lstm_out[:, -1, :]
        
        # Classification prediction
        class_pred = self.classifier(last_hidden)
        
        # Track prediction (if enabled)
        if self.predict_track:
            track_pred = self.track_predictor(last_hidden)
            return class_pred, track_pred
        
        return class_pred

def create_model(config):
    """Factory function to create model based on configuration"""
    return HurricanePredictor(
        in_channels=config.get('in_channels', 3),
        lstm_hidden_size=config.get('lstm_hidden_size', 256),
        num_lstm_layers=config.get('num_lstm_layers', 2),
        num_classes=config.get('num_classes', 5),
        predict_track=config.get('predict_track', True)
    ) 