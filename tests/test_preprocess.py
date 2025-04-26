#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import the preprocess module
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import (
    create_sequences, 
    MinMaxScaler, 
    ZScoreScaler, 
    split_data
)


class TestCreateSequences(unittest.TestCase):
    """Test the create_sequences function with simple arrays."""
    
    def test_simple_array_single_feature(self):
        """Test create_sequences with a simple 1D array."""
        # Create a simple array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data = np.arange(10).reshape(-1, 1)
        seq_len = 3
        horizon = 2
        target_idx = 0
        
        X, y = create_sequences(data, seq_len, horizon, target_idx)
        
        # Expected shapes
        self.assertEqual(X.shape, (6, 3, 1))  # 10 - 3 - 2 + 1 = 6 windows
        self.assertEqual(y.shape, (6,))
        
        # Expected values for X
        expected_X = np.array([
            [[0], [1], [2]],  # First sequence
            [[1], [2], [3]],  # Second sequence
            [[2], [3], [4]],  # Third sequence
            [[3], [4], [5]],  # Fourth sequence
            [[4], [5], [6]],  # Fifth sequence
            [[5], [6], [7]],  # Sixth sequence
        ])
        np.testing.assert_array_equal(X, expected_X)
        
        # Expected values for y (target is 2 steps ahead)
        expected_y = np.array([4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(y, expected_y)
    
    def test_multi_feature_array(self):
        """Test create_sequences with a multi-feature array."""
        # Create a 2D array with 2 features
        data = np.array([
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
            [4, 14],
            [5, 15],
            [6, 16],
            [7, 17],
            [8, 18],
            [9, 19]
        ])
        seq_len = 3
        horizon = 1
        target_idx = 1  # Predict the second feature
        
        X, y = create_sequences(data, seq_len, horizon, target_idx)
        
        # Expected shapes
        self.assertEqual(X.shape, (7, 3, 2))  # 10 - 3 - 1 + 1 = 7 windows
        self.assertEqual(y.shape, (7,))
        
        # Expected values for first window
        expected_first_window = np.array([[0, 10], [1, 11], [2, 12]])
        np.testing.assert_array_equal(X[0], expected_first_window)
        
        # Expected target values (second feature, 1 step ahead)
        expected_y = np.array([13, 14, 15, 16, 17, 18, 19])
        np.testing.assert_array_equal(y, expected_y)
    
    def test_multi_target(self):
        """Test create_sequences with multiple target indices."""
        # Create a 2D array with 3 features
        data = np.array([
            [0, 10, 20],
            [1, 11, 21],
            [2, 12, 22],
            [3, 13, 23],
            [4, 14, 24],
            [5, 15, 25],
            [6, 16, 26],
            [7, 17, 27],
            [8, 18, 28],
            [9, 19, 29]
        ])
        seq_len = 2
        horizon = 2
        target_idx = [1, 2]  # Predict the second and third features
        
        X, y = create_sequences(data, seq_len, horizon, target_idx)
        
        # Expected shapes
        self.assertEqual(X.shape, (7, 2, 3))  # 10 - 2 - 2 + 1 = 7 windows
        self.assertEqual(y.shape, (7, 2))  # 7 windows, 2 target features
        
        # Expected values for first window
        expected_first_window = np.array([[0, 10, 20], [1, 11, 21]])
        np.testing.assert_array_equal(X[0], expected_first_window)
        
        # Expected target values (features 1 and 2, 2 steps ahead)
        expected_y = np.array([
            [13, 23],
            [14, 24],
            [15, 25],
            [16, 26],
            [17, 27],
            [18, 28],
            [19, 29]
        ])
        np.testing.assert_array_equal(y, expected_y)


class TestScalers(unittest.TestCase):
    """Test the MinMaxScaler and ZScoreScaler classes."""
    
    def test_minmax_scaler_transform_inverse(self):
        """Test that MinMaxScaler.inverse_transform reverses transform."""
        # Create sample data
        data = np.array([
            [1, 5, 10],
            [2, 6, 20],
            [3, 7, 30],
            [4, 8, 40]
        ])
        
        # Initialize and fit scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        
        # Transform data
        scaled_data = scaler.transform(data)
        
        # Check scaling is in range [0, 1]
        self.assertTrue(np.all(scaled_data >= 0))
        self.assertTrue(np.all(scaled_data <= 1))
        
        # Inverse transform
        restored_data = scaler.inverse_transform(scaled_data)
        
        # Check that we get the original data back
        np.testing.assert_allclose(restored_data, data)
    
    def test_minmax_scaler_custom_range(self):
        """Test MinMaxScaler with a custom feature range."""
        # Create sample data
        data = np.array([
            [1, 5, 10],
            [2, 6, 20],
            [3, 7, 30],
            [4, 8, 40]
        ])
        
        # Initialize and fit scaler with custom range
        feature_range = (-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(data)
        
        # Transform data
        scaled_data = scaler.transform(data)
        
        # Check scaling is in range [-1, 1]
        self.assertTrue(np.all(scaled_data >= -1))
        self.assertTrue(np.all(scaled_data <= 1))
        
        # Check min values are mapped to -1 and max values to 1
        self.assertAlmostEqual(scaled_data[0, 0], -1)  # Min of first column
        self.assertAlmostEqual(scaled_data[3, 0], 1)   # Max of first column
        self.assertAlmostEqual(scaled_data[0, 2], -1)  # Min of third column
        self.assertAlmostEqual(scaled_data[3, 2], 1)   # Max of third column
        
        # Inverse transform
        restored_data = scaler.inverse_transform(scaled_data)
        
        # Check that we get the original data back
        np.testing.assert_allclose(restored_data, data)
    
    def test_zscore_scaler_transform_inverse(self):
        """Test that ZScoreScaler.inverse_transform reverses transform."""
        # Create sample data
        data = np.array([
            [1, 5, 10],
            [2, 6, 20],
            [3, 7, 30],
            [4, 8, 40]
        ])
        
        # Initialize and fit scaler
        scaler = ZScoreScaler()
        scaler.fit(data)
        
        # Transform data
        scaled_data = scaler.transform(data)
        
        # Check that mean is approximately 0 and std is approximately 1
        self.assertAlmostEqual(np.mean(scaled_data[:, 0]), 0, places=10)
        self.assertAlmostEqual(np.std(scaled_data[:, 0]), 1, places=10)
        
        # Inverse transform
        restored_data = scaler.inverse_transform(scaled_data)
        
        # Check that we get the original data back
        np.testing.assert_allclose(restored_data, data)
    
    def test_zscore_scaler_zero_std(self):
        """Test ZScoreScaler with a column having zero standard deviation."""
        # Create sample data with a constant column
        data = np.array([
            [1, 5, 10],
            [2, 5, 20],
            [3, 5, 30],
            [4, 5, 40]
        ])
        
        # Initialize and fit scaler
        scaler = ZScoreScaler()
        scaler.fit(data)
        
        # Transform data
        scaled_data = scaler.transform(data)
        
        # Check that the constant column is all zeros
        np.testing.assert_array_equal(scaled_data[:, 1], np.zeros(4))
        
        # Inverse transform
        restored_data = scaler.inverse_transform(scaled_data)
        
        # Check that we get the original data back
        np.testing.assert_allclose(restored_data, data)


class TestSplitData(unittest.TestCase):
    """Test the split_data function."""
    
    def test_split_proportions(self):
        """Test that data is split according to the specified proportions."""
        # Create sample data
        X = np.arange(100).reshape(-1, 1, 1)  # 100 samples, 1 timestep, 1 feature
        y = np.arange(100)
        
        # Define split ratios
        split_ratios = [0.6, 0.2, 0.2]
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, split_ratios)
        
        # Check shapes
        self.assertEqual(len(X_train), 60)  # 60% of 100
        self.assertEqual(len(X_val), 20)    # 20% of 100
        self.assertEqual(len(X_test), 20)   # 20% of 100
        
        # Check that the data is split sequentially (not shuffled)
        np.testing.assert_array_equal(X_train[:, 0, 0], np.arange(60))
        np.testing.assert_array_equal(X_val[:, 0, 0], np.arange(60, 80))
        np.testing.assert_array_equal(X_test[:, 0, 0], np.arange(80, 100))
        
        np.testing.assert_array_equal(y_train, np.arange(60))
        np.testing.assert_array_equal(y_val, np.arange(60, 80))
        np.testing.assert_array_equal(y_test, np.arange(80, 100))
    
    def test_split_normalization(self):
        """Test that split ratios are normalized if they don't sum to 1."""
        # Create sample data
        X = np.arange(100).reshape(-1, 1, 1)
        y = np.arange(100)
        
        # Define split ratios that don't sum to 1
        split_ratios = [1, 1, 1]  # Should be normalized to [1/3, 1/3, 1/3]
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, split_ratios)
        
        # Check shapes
        self.assertEqual(len(X_train), 33)  # ~33% of 100
        self.assertEqual(len(X_val), 33)    # ~33% of 100
        self.assertEqual(len(X_test), 34)   # ~33% of 100 (remaining)
        
        # Check that all data is used
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 100)


if __name__ == '__main__':
    unittest.main()
