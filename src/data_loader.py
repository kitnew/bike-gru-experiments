#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from typing import Dict, Tuple, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_loader')


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data that loads processed data from numpy arrays
    and converts them to PyTorch tensors.
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None, 
                 X: Optional[np.ndarray] = None, 
                 y: Optional[np.ndarray] = None,
                 config: Optional[Dict] = None,
                 verbose: bool = False):
        """
        Initialize the dataset either from files in a directory or from provided arrays.
        
        Args:
            data_path: Path to directory with processed data files
            X: Input sequences as numpy array (n_windows, seq_len, n_features)
            y: Target values as numpy array (n_windows,) or (n_windows, n_targets)
            config: Configuration dictionary with dataset parameters
        """
        self.config = config
        self.verbose = verbose
        
        # Load data either from arrays or from files
        if X is not None and y is not None:
            self.X = X
            self.y = y
            if verbose:
                logger.info(f"Dataset initialized from provided arrays: {self.X.shape} inputs, {self.y.shape} targets")
        elif data_path is not None:
            self._load_from_path(data_path)
        else:
            raise ValueError("Either provide data arrays (X, y) or a path to load data from")
        
        # Convert to torch tensors
        self.X_tensor = torch.FloatTensor(self.X)
        self.y_tensor = torch.FloatTensor(self.y)
        
        if verbose:
            logger.info(f"Dataset prepared with {len(self)} samples")
    
    def _load_from_path(self, data_path: str) -> None:
        """
        Load data from processed files in the specified directory.
        
        Args:
            data_path: Path to directory with processed data files
        """
        if self.verbose:
            logger.info(f"Loading data from {data_path}")
        
        # Find the most recent data file (assuming naming convention from preprocess.py)
        npz_files = [f for f in os.listdir(data_path) if f.startswith('data_') and f.endswith('.npz')]
        
        if not npz_files:
            raise FileNotFoundError(f"No data files found in {data_path}")
        
        # Sort by timestamp (assuming format: data_seq{seq_len}_h{horizon}_{timestamp}.npz)
        npz_files.sort(reverse=True)  # Most recent first
        latest_file = os.path.join(data_path, npz_files[0])
        
        if self.verbose:
            logger.info(f"Loading data from {latest_file}")
        
        # Load data
        data = np.load(latest_file)
        
        # Determine which split to use based on config
        split = self.config.get('split', 'train') if self.config else 'train'
        
        if split == 'train':
            self.X = data['X_train']
            self.y = data['y_train']
        elif split == 'val':
            self.X = data['X_val']
            self.y = data['y_val']
        elif split == 'test':
            self.X = data['X_test']
            self.y = data['y_test']
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of: train, val, test")
        
        if self.verbose:
            logger.info(f"Loaded {split} data: {self.X.shape} inputs, {self.y.shape} targets")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_sequence, target)
        """
        return self.X_tensor[idx], self.y_tensor[idx]


def get_dataloader(config: Dict, split: str = 'train', verbose: bool = False) -> DataLoader:
    """
    Create a DataLoader for the specified split.
    
    Args:
        config: Configuration dictionary with dataset and dataloader parameters
        split: Which data split to use ('train', 'val', or 'test')
        
    Returns:
        DataLoader for the specified split
    """
    if verbose:
        logger.info(f"Creating {split} dataloader")
    
    # Extract parameters
    data_path = config.dirs.processed_dir
    batch_size = config.dataloader.batch_size
    num_workers = config.dataloader.num_workers
    shuffle = config.dataloader.shuffle if split == 'train' else False
    
    # Create dataset config with the specified split
    dataset_config = {'split': split}
    
    # Create dataset
    dataset = TimeSeriesDataset(data_path=data_path, config=dataset_config)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    if verbose:
        logger.info(f"Created {split} dataloader with {len(dataset)} samples, batch size {batch_size}")
    
    return dataloader


def get_all_dataloaders(config: Dict, verbose: bool = False) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits (train, val, test).
    
    Args:
        config: Configuration dictionary with dataset and dataloader parameters
        
    Returns:
        Dictionary with DataLoaders for each split
    """
    return {
        'train': get_dataloader(config, 'train', verbose=verbose),
        'val': get_dataloader(config, 'val', verbose=verbose),
        'test': get_dataloader(config, 'test', verbose=verbose)
    }