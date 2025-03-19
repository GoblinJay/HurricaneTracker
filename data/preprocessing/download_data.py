import requests
import os
import pandas as pd
from datetime import datetime
import logging

def download_hurdat2_data():
    """Download HURDAT2 data from NOAA"""
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # URL for HURDAT2 data
    url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"
    
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        output_path = os.path.join('data/raw', 'hurdat2.txt')
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"Successfully downloaded HURDAT2 data to {output_path}")
        
        # Parse and preprocess the data
        parse_hurdat2_data(output_path)
        
    except Exception as e:
        logging.error(f"Error downloading HURDAT2 data: {str(e)}")
        raise

def parse_coordinate(coord_str):
    """Parse coordinate string with N/S or E/W suffix"""
    value = float(coord_str[:-1])  # Remove the last character (N/S/E/W)
    if coord_str.endswith('S') or coord_str.endswith('W'):
        value = -value
    return value

def parse_hurdat2_data(file_path):
    """Parse HURDAT2 data into a structured format"""
    storms = []
    current_storm = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('AL'):  # Header line for a new storm
                if current_storm:
                    storms.append(current_storm)
                
                # Parse header line
                parts = line.strip().split(',')
                current_storm = {
                    'storm_id': parts[0].strip(),
                    'name': parts[1].strip(),
                    'entries': []
                }
            elif current_storm and line.strip():  # Data line
                # Parse data line
                parts = line.strip().split(',')
                entry = {
                    'date': datetime.strptime(parts[0].strip(), '%Y%m%d'),
                    'time': parts[1].strip(),
                    'record_identifier': parts[2].strip(),
                    'status': parts[3].strip(),
                    'latitude': parse_coordinate(parts[4].strip()),
                    'longitude': parse_coordinate(parts[5].strip()),
                    'max_wind': float(parts[6].strip()),
                    'min_pressure': float(parts[7].strip()) if parts[7].strip() else None,
                    'category': determine_category(float(parts[6].strip()))
                }
                current_storm['entries'].append(entry)
    
    # Add the last storm
    if current_storm:
        storms.append(current_storm)
    
    # Convert to DataFrame
    all_entries = []
    for storm in storms:
        for entry in storm['entries']:
            entry['storm_id'] = storm['storm_id']
            entry['storm_name'] = storm['name']
            all_entries.append(entry)
    
    df = pd.DataFrame(all_entries)
    
    # Save processed data
    output_path = os.path.join('data/processed', 'hurricane_data.csv')
    df.to_csv(output_path, index=False)
    logging.info(f"Saved processed hurricane data to {output_path}")
    
    return df

def determine_category(wind_speed):
    """Determine hurricane category based on wind speed"""
    if wind_speed < 38:  # Tropical Depression
        return 1
    elif wind_speed < 74:  # Tropical Storm
        return 2
    elif wind_speed < 96:  # Category 1
        return 3
    elif wind_speed < 111:  # Category 2
        return 4
    else:  # Category 3+
        return 5

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Download and process data
    download_hurdat2_data() 