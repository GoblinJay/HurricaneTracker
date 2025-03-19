import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import logging
import os
from typing import Dict, Tuple, Optional
import json

from ..architectures.hurricane_model import create_model
from ...data.preprocessing.data_processor import DataProcessor

class HurricaneTrainer:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.config)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(self.config).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.track_criterion = nn.MSELoss()
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['output_dir'], 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_track_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config['predict_track']:
                class_pred, track_pred = self.model(images)
                class_loss = self.class_criterion(class_pred, labels[:, 0].long())
                track_loss = self.track_criterion(track_pred, labels[:, 1:])
                loss = class_loss + self.config['track_loss_weight'] * track_loss
            else:
                class_pred = self.model(images)
                loss = self.class_criterion(class_pred, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            if self.config['predict_track']:
                total_class_loss += class_loss.item()
                total_track_loss += track_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(class_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()
        
        # Calculate average metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
        
        if self.config['predict_track']:
            metrics.update({
                'class_loss': total_class_loss / len(train_loader),
                'track_loss': total_track_loss / len(train_loader)
            })
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_class_loss = 0
        total_track_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.config['predict_track']:
                    class_pred, track_pred = self.model(images)
                    class_loss = self.class_criterion(class_pred, labels[:, 0].long())
                    track_loss = self.track_criterion(track_pred, labels[:, 1:])
                    loss = class_loss + self.config['track_loss_weight'] * track_loss
                else:
                    class_pred = self.model(images)
                    loss = self.class_criterion(class_pred, labels)
                
                # Update metrics
                total_loss += loss.item()
                if self.config['predict_track']:
                    total_class_loss += class_loss.item()
                    total_track_loss += track_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(class_pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels[:, 0]).sum().item()
        
        # Calculate average metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100 * correct / total
        }
        
        if self.config['predict_track']:
            metrics.update({
                'class_loss': total_class_loss / len(val_loader),
                'track_loss': total_track_loss / len(val_loader)
            })
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        path = os.path.join(
            self.config['output_dir'],
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, path)
        logging.info(f'Saved checkpoint to {path}')
    
    def train(self, num_epochs: int):
        """Train the model"""
        # Prepare data loaders
        train_loader, val_loader, _ = self.data_processor.prepare_dataset()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logging.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logging.info(f'Train metrics: {train_metrics}')
            
            # Validate
            val_metrics = self.validate(val_loader)
            logging.info(f'Validation metrics: {val_metrics}')
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save checkpoint if validation loss improves
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                logging.info('Early stopping triggered')
                break

def main():
    # Example configuration
    config = {
        'data_dir': 'data',
        'output_dir': 'models/saved',
        'sequence_length': 24,
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'track_loss_weight': 1.0,
        'predict_track': True,
        'patience': 10,
        'in_channels': 3,
        'lstm_hidden_size': 256,
        'num_lstm_layers': 2,
        'num_classes': 5
    }
    
    # Save configuration
    os.makedirs('models/saved', exist_ok=True)
    with open('models/saved/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize trainer
    trainer = HurricaneTrainer('models/saved/config.json')
    
    # Train model
    trainer.train(num_epochs=100)

if __name__ == '__main__':
    main() 