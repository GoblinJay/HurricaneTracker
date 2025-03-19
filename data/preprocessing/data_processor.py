import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta

class HurricaneDataset(Dataset):
    """Custom dataset for hurricane prediction"""
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 sequence_length: int = 24,  # Number of time steps
                 predict_track: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.predict_track = predict_track
        
        # Load hurricane track data
        self.track_data = self._load_track_data()
        
        # Load satellite imagery paths
        self.image_paths = self._load_image_paths()
        
    def _load_track_data(self) -> pd.DataFrame:
        """Load and preprocess hurricane track data from HURDAT2 format"""
        # Example path - adjust based on your data structure
        track_file = os.path.join(self.data_dir, 'raw', 'hurdat2.txt')
        
        # Read HURDAT2 data
        # This is a simplified version - you'll need to implement the actual parsing
        # based on the HURDAT2 format specification
        tracks = []
        with open(track_file, 'r') as f:
            for line in f:
                # Parse HURDAT2 format
                # Example format: YYYYMMDD, HHMM, LAT, LON, VMAX, MSLP, etc.
                pass
                
        return pd.DataFrame(tracks)
    
    def _load_image_paths(self) -> List[str]:
        """Load paths to satellite imagery"""
        image_dir = os.path.join(self.data_dir, 'raw', 'satellite_images')
        return [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                if f.endswith(('.jpg', '.png', '.nc'))]
    
    def _load_satellite_image(self, path: str) -> np.ndarray:
        """Load and preprocess satellite image"""
        if path.endswith('.nc'):
            # Handle NetCDF format (common for weather data)
            with nc.Dataset(path) as ds:
                # Extract relevant variables (e.g., temperature, pressure)
                # Adjust based on your data structure
                data = ds.variables['temperature'][:]
        else:
            # Handle regular image formats
            img = Image.open(path)
            data = np.array(img)
            
        return data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return (data - data.min()) / (data.max() - data.min())
    
    def __len__(self) -> int:
        return len(self.image_paths) - self.sequence_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence of satellite images and corresponding labels"""
        # Load sequence of images
        images = []
        for i in range(self.sequence_length):
            img_path = self.image_paths[idx + i]
            img_data = self._load_satellite_image(img_path)
            img_data = self._normalize_data(img_data)
            images.append(img_data)
        
        # Stack images into sequence
        images = np.stack(images)
        images = torch.from_numpy(images).float()
        
        # Get corresponding track data
        track_data = self.track_data.iloc[idx:idx + self.sequence_length]
        
        # Prepare labels
        if self.predict_track:
            # For track prediction, use lat/lon coordinates
            labels = torch.tensor(track_data[['latitude', 'longitude']].values, 
                                dtype=torch.float32)
        else:
            # For classification, use hurricane category
            labels = torch.tensor(track_data['category'].values, 
                                dtype=torch.long)
        
        return images, labels

class DataProcessor:
    """Class for processing and preparing hurricane data"""
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config['data_dir']
        
    def prepare_dataset(self, 
                       split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test datasets"""
        # Create datasets
        train_dataset = HurricaneDataset(
            os.path.join(self.data_dir, 'train'),
            sequence_length=self.config['sequence_length'],
            predict_track=self.config['predict_track']
        )
        
        val_dataset = HurricaneDataset(
            os.path.join(self.data_dir, 'val'),
            sequence_length=self.config['sequence_length'],
            predict_track=self.config['predict_track']
        )
        
        test_dataset = HurricaneDataset(
            os.path.join(self.data_dir, 'test'),
            sequence_length=self.config['sequence_length'],
            predict_track=self.config['predict_track']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        return train_loader, val_loader, test_loader
    
    def preprocess_single_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single satellite image for inference"""
        dataset = HurricaneDataset(
            os.path.dirname(image_path),
            sequence_length=1,
            predict_track=self.config['predict_track']
        )
        image, _ = dataset[0]  # Get first (and only) image
        return image.unsqueeze(0)  # Add batch dimension 