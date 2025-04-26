#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import torch
import sys
import os
from pathlib import Path
import tempfile

# Add parent directory to path to import the modules
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import TimeSeriesDataset, get_dataloader


class TestTimeSeriesDataset(unittest.TestCase):
    """Test the TimeSeriesDataset class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data: 100 sequences, each with 10 timesteps and 5 features
        self.X = np.random.rand(100, 10, 5)
        
        # Create targets: 100 samples with a single target value
        self.y = np.random.rand(100)
        
        # Create multi-target data: 100 samples with 3 target values
        self.y_multi = np.random.rand(100, 3)
    
    def test_dataset_from_arrays(self):
        """Test creating a dataset from numpy arrays."""
        # Create dataset
        dataset = TimeSeriesDataset(X=self.X, y=self.y)
        
        # Check length
        self.assertEqual(len(dataset), 100)
        
        # Check item retrieval
        x_item, y_item = dataset[0]
        
        # Check types
        self.assertIsInstance(x_item, torch.Tensor)
        self.assertIsInstance(y_item, torch.Tensor)
        
        # Check shapes
        self.assertEqual(x_item.shape, (10, 5))  # (seq_len, n_features)
        self.assertEqual(y_item.shape, ())       # Scalar target
        
        # Check values
        np.testing.assert_allclose(x_item.numpy(), self.X[0], rtol=1e-5)
        self.assertAlmostEqual(y_item.item(), self.y[0], places=5)
    
    def test_dataset_multi_target(self):
        """Test dataset with multiple target values."""
        # Create dataset with multi-target data
        dataset = TimeSeriesDataset(X=self.X, y=self.y_multi)
        
        # Check item retrieval
        x_item, y_item = dataset[0]
        
        # Check shapes
        self.assertEqual(x_item.shape, (10, 5))  # (seq_len, n_features)
        self.assertEqual(y_item.shape, (3,))     # Multi-target
        
        # Check values
        np.testing.assert_allclose(y_item.numpy(), self.y_multi[0], rtol=1e-5)
    
    def test_dataloader(self):
        """Test creating a DataLoader from a dataset."""
        # Create dataset
        dataset = TimeSeriesDataset(X=self.X, y=self.y)
        
        # Create config
        config = {
            'dataloader': {
                'batch_size': 16,
                'shuffle': True,
                'num_workers': 0
            }
        }
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['dataloader']['batch_size'],
            shuffle=config['dataloader']['shuffle'],
            num_workers=config['dataloader']['num_workers']
        )
        
        # Check batch retrieval
        for batch_x, batch_y in dataloader:
            # Check batch shapes
            self.assertEqual(batch_x.shape[0], min(16, len(dataset)))  # Batch size
            self.assertEqual(batch_x.shape[1], 10)                     # Sequence length
            self.assertEqual(batch_x.shape[2], 5)                      # Features
            self.assertEqual(batch_y.shape[0], min(16, len(dataset)))  # Batch size
            break  # Only check the first batch
    
    def test_simple_array(self):
        """Test with a simple array like np.arange."""
        # Create a simple array with clear pattern
        X = np.arange(50).reshape(10, 5, 1)  # 10 sequences, 5 timesteps, 1 feature
        y = np.arange(10)                    # 10 target values
        
        # Create dataset
        dataset = TimeSeriesDataset(X=X, y=y)
        
        # Check item retrieval
        x_item, y_item = dataset[3]  # Get the 4th item
        
        # Expected values
        expected_x = np.arange(15, 20).reshape(5, 1)  # 4th sequence (0-indexed)
        expected_y = 3                                # 4th target
        
        # Check values
        np.testing.assert_array_equal(x_item.numpy(), expected_x)
        self.assertEqual(y_item.item(), expected_y)
        
        # Create config for dataloader
        config = {
            'dataloader': {
                'batch_size': 4,
                'shuffle': False,
                'num_workers': 0
            }
        }
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['dataloader']['batch_size'],
            shuffle=config['dataloader']['shuffle'],
            num_workers=config['dataloader']['num_workers']
        )
        
        # Get first batch
        batch_x, batch_y = next(iter(dataloader))
        
        # Expected batch values (first 4 sequences)
        expected_batch_x = torch.FloatTensor([
            [[0], [1], [2], [3], [4]],         # 1st sequence
            [[5], [6], [7], [8], [9]],         # 2nd sequence
            [[10], [11], [12], [13], [14]],    # 3rd sequence
            [[15], [16], [17], [18], [19]]     # 4th sequence
        ])
        expected_batch_y = torch.FloatTensor([0, 1, 2, 3])  # First 4 targets
        
        # Check batch shapes and values
        self.assertEqual(batch_x.shape, (4, 5, 1))  # 4 sequences, 5 timesteps, 1 feature
        self.assertEqual(batch_y.shape, (4,))       # 4 target values
        
        # Check values
        torch.testing.assert_close(batch_x, expected_batch_x)
        torch.testing.assert_close(batch_y, expected_batch_y)


if __name__ == '__main__':
    unittest.main()
