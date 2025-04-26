#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
from typing import Union, List, Tuple, Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocess')


class Scaler:
    """Base class for data scalers."""
    
    def __init__(self):
        self.params = {}
        
    def fit(self, data: np.ndarray) -> 'Scaler':
        """Compute parameters needed for scaling."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply scaling to the data."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Revert scaling applied to the data."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def save_params(self, filepath: str) -> None:
        """Save scaler parameters to file."""
        np.savez(filepath, **self.params)
        logger.info(f"Scaler parameters saved to {filepath}")
        
    def load_params(self, filepath: str) -> None:
        """Load scaler parameters from file."""
        loaded = np.load(filepath)
        self.params = {key: loaded[key] for key in loaded.files}
        logger.info(f"Scaler parameters loaded from {filepath}")


class MinMaxScaler(Scaler):
    """Min-Max scaler that scales data to a specified range."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.feature_range = feature_range
        
    def fit(self, data: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max values for scaling."""
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        
        self.params = {
            'data_min': data_min,
            'data_max': data_max,
            'feature_range': np.array(self.feature_range)
        }
        
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data to the specified range."""
        data_min = self.params['data_min']
        data_max = self.params['data_max']
        feature_min, feature_max = self.params['feature_range']
        
        # Create a copy to avoid modifying the original data
        scaled_data = np.copy(data).astype(float)
        
        # Scale each feature
        for i in range(data.shape[1]):
            # Handle constant features (where max == min)
            if data_max[i] == data_min[i]:
                scaled_data[:, i] = feature_min
            else:
                # Apply min-max scaling formula
                scaled_data[:, i] = feature_min + (feature_max - feature_min) * \
                                   (data[:, i] - data_min[i]) / (data_max[i] - data_min[i])
        
        return scaled_data
        
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform to get original data."""
        data_min = self.params['data_min']
        data_max = self.params['data_max']
        feature_min, feature_max = self.params['feature_range']
        
        # Create a copy to avoid modifying the original data
        original_data = np.copy(scaled_data).astype(float)
        
        # Inverse scale each feature
        for i in range(scaled_data.shape[1]):
            # Handle constant features (where max == min)
            if data_max[i] == data_min[i]:
                original_data[:, i] = data_min[i]
            else:
                # Apply inverse min-max scaling formula
                original_data[:, i] = data_min[i] + (data_max[i] - data_min[i]) * \
                                     (scaled_data[:, i] - feature_min) / (feature_max - feature_min)
        
        return original_data


class ZScoreScaler(Scaler):
    """Z-score scaler that standardizes data to have zero mean and unit variance."""
    
    def __init__(self):
        super().__init__()
        
    def fit(self, data: np.ndarray) -> 'ZScoreScaler':
        """Compute mean and standard deviation for scaling."""
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        
        # Handle zero standard deviation
        data_std[data_std == 0] = 1.0
        
        self.params = {
            'data_mean': data_mean,
            'data_std': data_std
        }
        
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Standardize data to have zero mean and unit variance."""
        data_mean = self.params['data_mean']
        data_std = self.params['data_std']
        
        scaled_data = (data - data_mean) / data_std
        
        return scaled_data
        
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform to get original data."""
        data_mean = self.params['data_mean']
        data_std = self.params['data_std']
        
        original_data = scaled_data * data_std + data_mean
        
        return original_data


def load_data(filepath: str, datetime_col: Optional[str] = None, datetime_format: Optional[str] = None) -> pd.DataFrame:
    """Load tabular data from various file formats.
    
    Args:
        filepath: Path to the data file
        datetime_col: Name of the datetime column to parse
        datetime_format: Format string for datetime parsing
        
    Returns:
        Loaded data as a pandas DataFrame
    """
    logger.info(f"Loading data from {filepath}")
    
    # Determine file extension
    ext = os.path.splitext(filepath)[1].lower()
    
    # Load data based on file extension
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.tsv':
        df = pd.read_csv(filepath, sep='\t')
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif ext == '.feather':
        df = pd.read_feather(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Parse datetime column if specified
    if datetime_col and datetime_col in df.columns:
        logger.info(f"Parsing datetime column: {datetime_col}")
        try:
            if datetime_format:
                df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
            else:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            logger.warning(f"Failed to parse datetime column: {e}")
    
    logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame, check_missing: bool = True, check_dimensions: bool = True) -> pd.DataFrame:
    """Validate the loaded data.
    
    Args:
        df: Input DataFrame
        check_missing: Whether to check for missing values
        check_dimensions: Whether to check data dimensions
        
    Returns:
        Validated DataFrame
    """
    logger.info("Validating data...")
    
    # Check for missing values
    if check_missing:
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
            
            # Fill missing values with appropriate methods
            # For numeric columns, use median
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    logger.info(f"Filling missing values in {col} with median")
                    df[col] = df[col].fillna(df[col].median())
            
            # For categorical columns, use mode
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    logger.info(f"Filling missing values in {col} with mode")
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            # For datetime columns, use forward fill
            datetime_cols = df.select_dtypes(include=['datetime']).columns
            for col in datetime_cols:
                if df[col].isnull().sum() > 0:
                    logger.info(f"Filling missing values in {col} with forward fill")
                    df[col] = df[col].fillna(method='ffill')
    
    # Check data dimensions
    if check_dimensions:
        if df.shape[0] < 10:
            logger.warning(f"Data has only {df.shape[0]} rows, which may be insufficient")
        if df.shape[1] < 2:
            logger.warning(f"Data has only {df.shape[1]} columns, which may be insufficient")
    
    logger.info("Data validation completed")
    return df


def create_sequences(data: np.ndarray, seq_len: int, horizon: int, target_idx: Union[int, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time series forecasting.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        seq_len: Length of input sequence
        horizon: Forecast horizon (steps ahead to predict)
        target_idx: Index or indices of target variable(s)
        
    Returns:
        X: Input sequences of shape (n_windows, seq_len, n_features)
        y: Target values of shape (n_windows,) or (n_windows, n_targets)
    """
    logger.info(f"Creating sequences with seq_len={seq_len}, horizon={horizon}")
    
    n_samples, n_features = data.shape
    n_windows = n_samples - seq_len - horizon + 1
    
    if n_windows <= 0:
        raise ValueError(f"Cannot create sequences: data has {n_samples} samples, "
                         f"but seq_len={seq_len} and horizon={horizon} require at least "
                         f"{seq_len + horizon} samples")
    
    # Initialize arrays
    X = np.zeros((n_windows, seq_len, n_features))
    
    # Convert target_idx to list if it's a single integer
    if isinstance(target_idx, int):
        target_idx = [target_idx]
    
    n_targets = len(target_idx)
    y = np.zeros((n_windows, n_targets)) if n_targets > 1 else np.zeros(n_windows)
    
    # Create sequences
    for i in range(n_windows):
        # Input sequence
        X[i] = data[i:i+seq_len]
        
        # Target value(s)
        target_values = data[i+seq_len+horizon-1, target_idx]
        if n_targets > 1:
            y[i] = target_values
        else:
            y[i] = target_values[0]
    
    logger.info(f"Created {n_windows} sequences")
    
    # Validate output shapes
    expected_x_shape = (n_windows, seq_len, n_features)
    expected_y_shape = (n_windows, n_targets) if n_targets > 1 else (n_windows,)
    
    if X.shape != expected_x_shape:
        raise ValueError(f"Expected X shape {expected_x_shape}, got {X.shape}")
    if y.shape != expected_y_shape:
        raise ValueError(f"Expected y shape {expected_y_shape}, got {y.shape}")
    
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, split_ratios: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.
    
    Args:
        X: Input sequences
        y: Target values
        split_ratios: List of ratios [train, val, test] that sum to 1.0
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    logger.info(f"Splitting data with ratios: {split_ratios}")
    
    # Validate split ratios
    if len(split_ratios) != 3:
        raise ValueError(f"Expected 3 split ratios, got {len(split_ratios)}")
    if not np.isclose(sum(split_ratios), 1.0):
        logger.warning(f"Split ratios {split_ratios} do not sum to 1.0, normalizing")
        split_ratios = [r / sum(split_ratios) for r in split_ratios]
    
    n_samples = len(X)
    train_end = int(n_samples * split_ratios[0])
    val_end = train_end + int(n_samples * split_ratios[1])
    
    # Split data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_processed_data(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray, scaler: Scaler, config: Dict, 
                        output_dir: str, format: str = 'npz') -> None:
    """Save processed data to disk.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Split datasets
        scaler: Fitted scaler object
        config: Configuration dictionary
        output_dir: Directory to save data
        format: Output format ('npy' or 'npz')
    """
    logger.info(f"Saving processed data to {output_dir} in {format} format")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    seq_len = config['preprocess']['seq_len']
    horizon = config['preprocess']['horizon']
    
    # Base filename
    base_filename = f"seq{seq_len}_h{horizon}_{timestamp}"
    
    # Save data based on format
    if format.lower() == 'npy':
        np.save(os.path.join(output_dir, f"X_train_{base_filename}.npy"), X_train)
        np.save(os.path.join(output_dir, f"y_train_{base_filename}.npy"), y_train)
        np.save(os.path.join(output_dir, f"X_val_{base_filename}.npy"), X_val)
        np.save(os.path.join(output_dir, f"y_val_{base_filename}.npy"), y_val)
        np.save(os.path.join(output_dir, f"X_test_{base_filename}.npy"), X_test)
        np.save(os.path.join(output_dir, f"y_test_{base_filename}.npy"), y_test)
    elif format.lower() == 'npz':
        np.savez(
            os.path.join(output_dir, f"data_{base_filename}.npz"),
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    # Save scaler parameters
    scaler_filename = os.path.join(output_dir, f"scaler_{base_filename}.npz")
    scaler.save_params(scaler_filename)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'seq_len': seq_len,
        'horizon': horizon,
        'target_idx': config['preprocess']['target_idx'],
        'normalization_method': config['preprocess']['normalization']['method'],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X_train.shape[2],
        'split_ratios': config['preprocess']['train_split']
    }
    
    with open(os.path.join(output_dir, f"metadata_{base_filename}.yaml"), 'w') as f:
        yaml.dump(metadata, f)
    
    logger.info(f"Saved processed data with base filename: {base_filename}")


def preprocess_pipeline(config_path: str) -> Dict:
    """Execute the complete preprocessing pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with paths to saved data files
    """
    logger.info(f"Starting preprocessing pipeline with config: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    raw_dir = config['dirs']['raw_dir']
    processed_dir = config['dirs']['processed_dir']
    seq_len = config['preprocess']['seq_len']
    horizon = config['preprocess']['horizon']
    target_idx = config['preprocess']['target_idx']
    split_ratios = config['preprocess']['train_split']
    norm_method = config['preprocess']['normalization']['method']
    output_format = config['preprocess']['output']['format']
    
    # Find data files in raw directory
    raw_files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    if not raw_files:
        raise FileNotFoundError(f"No data files found in {raw_dir}")
    
    # Use the first file for now (can be extended to handle multiple files)
    print(raw_files)
    data_file = os.path.join(raw_dir, raw_files[0 if raw_files[0].endswith('.csv') else 1])
    logger.info(f"Processing file: {data_file}")
    
    # Load and validate data
    df = load_data(
        data_file,
        datetime_col=config['preprocess'].get('datetime_column'),
        datetime_format=config['preprocess'].get('datetime_format')
    )
    df = validate_data(
        df,
        check_missing=config['preprocess']['validation'].get('check_missing', True),
        check_dimensions=config['preprocess']['validation'].get('check_dimensions', True)
    )
    
    # Convert datetime to numerical features if present
    datetime_col = config['preprocess'].get('datetime_column')
    if datetime_col and datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        logger.info(f"Converting datetime column {datetime_col} to numerical features")
        df['hour'] = df[datetime_col].dt.hour
        df['day'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_year'] = df[datetime_col].dt.dayofyear
        
        # Drop original datetime column
        df = df.drop(columns=[datetime_col])
    
    # Convert DataFrame to numpy array
    data = df.values
    
    # Initialize and fit scaler
    if norm_method.lower() == 'minmax':
        feature_range = tuple(config['preprocess']['normalization'].get('feature_range', [0, 1]))
        scaler = MinMaxScaler(feature_range=feature_range)
    elif norm_method.lower() == 'zscore':
        scaler = ZScoreScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {norm_method}")
    
    # Fit and transform data
    logger.info(f"Applying {norm_method} normalization")
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    
    # Check for NaN values after normalization
    if np.isnan(normalized_data).any():
        logger.error("NaN values detected after normalization")
        nan_indices = np.where(np.isnan(normalized_data))
        logger.error(f"NaN indices: {nan_indices}")
        # Replace NaNs with zeros as a fallback
        normalized_data = np.nan_to_num(normalized_data)
    
    # Create sequences
    X, y = create_sequences(normalized_data, seq_len, horizon, target_idx)
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, split_ratios)
    
    # Save processed data
    save_processed_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, config, processed_dir, output_format
    )
    
    # Return information about the preprocessing
    return {
        'processed_dir': processed_dir,
        'n_samples': data.shape[0],
        'n_features': data.shape[1],
        'n_sequences': X.shape[0],
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'seq_len': seq_len,
        'horizon': horizon,
        'normalization': norm_method
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess time series data for forecasting')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Run preprocessing pipeline
    result = preprocess_pipeline(args.config)
    
    # Print summary
    print("\nPreprocessing completed successfully!")
    print(f"Total samples: {result['n_samples']}")
    print(f"Features: {result['n_features']}")
    print(f"Sequences: {result['n_sequences']}")
    print(f"Train/Val/Test split: {result['train_size']}/{result['val_size']}/{result['test_size']}")
    print(f"Processed data saved to: {result['processed_dir']}")