# Hurricane Analysis and Prediction System

A comprehensive web application for analyzing historical hurricane data and predicting hurricane paths using deep learning. The system provides interactive visualizations, statistical analysis, and path predictions based on historical data from the National Hurricane Center.

## Core Features

### 1. Data Management
- Automatic download and processing of HURDAT2 data
- Data preprocessing pipeline for cleaning and normalization
- Support for historical hurricane data from 1851-2022
- Efficient data storage and retrieval system

### 2. Interactive Visualization
- Interactive map display of hurricane tracks
- Multiple visualization modes:
  - Historical track visualization
  - Predicted path visualization
  - Combined historical and predicted paths
- Customizable map views and zoom levels
- Color-coded storm intensity indicators

### 3. Storm Analysis
- Detailed storm information display:
  - Storm ID and name
  - Date range
  - Maximum wind speed
  - Minimum pressure
  - Storm category
- Intensity timeline plots showing:
  - Wind speed over time
  - Pressure changes
  - Category changes
- Statistical analysis of storm characteristics

### 4. Prediction System
- LSTM-based path prediction model
- Features:
  - 5 input features (latitude, longitude, wind speed, pressure, category)
  - 2 LSTM layers with dropout
  - Fully connected layers for final prediction
  - Predicts next position based on previous 5 observations
- Prediction visualization with confidence intervals
- Customizable prediction time horizons

### 5. Filtering and Search
- Year range filtering
- Storm category filtering
- Search by storm ID or name
- Dynamic filtering of visualization data

### 6. Statistical Analysis
- Distribution analysis of:
  - Storm intensities
  - Path patterns
  - Seasonal trends
- Historical trend visualization
- Category distribution analysis

## Technical Architecture

### Backend
- Python-based web application
- FastAPI for API endpoints
- LSTM neural network for predictions
- Pandas for data manipulation
- NumPy for numerical computations

### Frontend
- Interactive web interface
- Plotly for map and chart visualizations
- Responsive design
- Real-time data updates

### Data Pipeline
- Automated data download
- Data cleaning and preprocessing
- Feature engineering
- Model training pipeline

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hurricane-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Download and process the HURDAT2 data:
```bash
python data/preprocessing/download_data.py
```

2. Train the prediction model:
```bash
python models/storm_predictor.py
```

3. Run the web application:
```bash
python app.py
```

The application will be available at http://localhost:8501

## Project Structure

```
hurricane-analysis/
├── app.py              # Main application entry point
├── models/             # Model implementation
│   ├── storm_predictor.py  # LSTM model implementation
│   ├── architectures/      # Model architecture definitions
│   ├── training/          # Training scripts
│   └── saved/            # Saved model checkpoints
├── data/               # Data management
│   ├── raw/            # Raw HURDAT2 data
│   ├── processed/      # Processed data
│   └── preprocessing/  # Data preprocessing scripts
├── frontend/           # Frontend components
├── api/                # API endpoints
├── requirements.txt    # Python dependencies
└── setup.py           # Package setup
```

## Data Sources

- HURDAT2 (Hurricane Database 2) from the National Hurricane Center
- Atlantic basin tropical cyclones from 1851-2022
- Includes detailed storm information:
  - Position data
  - Intensity measurements
  - Storm characteristics
  - Meteorological parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 