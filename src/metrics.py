import numpy as np
import torch

def mae(targets, predictions):
    """
    Mean Absolute Error
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Mean Absolute Error
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    return np.mean(np.abs(targets - predictions))

def mse(targets, predictions):
    """
    Mean Squared Error
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Mean Squared Error
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    return np.mean(np.square(targets - predictions))

def rmse(targets, predictions):
    """
    Root Mean Squared Error
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Root Mean Squared Error
    """
    return np.sqrt(mse(targets, predictions))

def r_squared(targets, predictions):
    """
    R-squared (Coefficient of Determination)
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: R-squared value
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    ss_total = np.sum(np.square(targets - np.mean(targets)))
    ss_residual = np.sum(np.square(targets - predictions))
    
    if ss_total == 0:
        return 0  # To handle the case when targets are all the same value
    
    return 1 - (ss_residual / ss_total)

def mape(targets, predictions):
    """
    Mean Absolute Percentage Error
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Mean Absolute Percentage Error
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Avoid division by zero
    mask = targets != 0
    if not np.any(mask):
        return 0.0
    
    return 100.0 * np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask]))

def smape(targets, predictions):
    """
    Symmetric Mean Absolute Percentage Error
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Symmetric Mean Absolute Percentage Error
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Avoid division by zero or very small numbers
    denominator = np.abs(targets) + np.abs(predictions)
    mask = denominator > 1e-8  # Small epsilon to avoid numerical issues
    
    if not np.any(mask):
        return 0.0
    
    return 200.0 * np.mean(np.abs(targets[mask] - predictions[mask]) / denominator[mask])

def explained_variance(targets, predictions):
    """
    Explained Variance Score
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Explained Variance Score
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    var_y = np.var(targets)
    if var_y == 0:
        return 0.0  # To handle the case when targets are all the same value
    
    return 1 - np.var(targets - predictions) / var_y

def peak_error(targets, predictions):
    """
    Peak Error (Maximum Absolute Error)
    
    Args:
        targets (torch.Tensor or np.ndarray): Ground truth values
        predictions (torch.Tensor or np.ndarray): Predicted values
        
    Returns:
        float: Maximum Absolute Error
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    return np.max(np.abs(targets - predictions))