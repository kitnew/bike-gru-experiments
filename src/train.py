import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from src.data_loader import get_dataloader, get_all_dataloaders
from src.metrics import mae, mse, rmse, r_squared, mape, smape, explained_variance, peak_error
from src.config import Config
from src.model import GRUNetwork

config = Config.from_yaml("/home/kitne/University/2lvl/SU/bike-gru-experiments/config/default.yaml")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.dirs.logs_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('train')


def train_epoch(model, dataloader, criterion, optimizer, device, verbose=False):
    """Train the model for one epoch.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on (cpu or cuda)
        
    Returns:
        dict: Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    
    # For metrics calculation
    all_targets = []
    all_predictions = []
    
    # Use tqdm for progress bar if verbose is True
    train_bar = tqdm(dataloader, desc="Training") if verbose else dataloader
    
    for inputs, targets in train_bar:
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(inputs)
        
        # Ensure targets have the same shape as outputs for loss calculation
        if outputs.shape != targets.shape:
            # If targets is a 1D array and outputs is 2D, reshape targets
            if len(targets.shape) == 1 and len(outputs.shape) == 2:
                targets = targets.view(-1, outputs.shape[1])
            # If outputs is a 2D array and targets is 1D, reshape outputs
            elif len(outputs.shape) == 2 and len(targets.shape) == 1:
                outputs = outputs.squeeze(1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * inputs.size(0)
        
        # Store predictions and targets for metric calculation
        # Ensure consistent shapes for stacking later
        targets_np = targets.cpu().detach().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        
        # Reshape if necessary to ensure consistent dimensions
        if len(targets_np.shape) == 1:
            targets_np = targets_np.reshape(-1, 1)
        if len(outputs_np.shape) == 1:
            outputs_np = outputs_np.reshape(-1, 1)
            
        all_targets.append(targets_np)
        all_predictions.append(outputs_np)
        
        # Update progress bar
        if verbose:
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Convert lists to numpy arrays
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
    metrics = {
        "loss": avg_loss,
        "mae": mae(all_targets, all_predictions),
        "mse": mse(all_targets, all_predictions),
        "rmse": rmse(all_targets, all_predictions),
        "r2": r_squared(all_targets, all_predictions),
        "mape": mape(all_targets, all_predictions),
        "smape": smape(all_targets, all_predictions),
        "explained_variance": explained_variance(all_targets, all_predictions),
        "peak_error": peak_error(all_targets, all_predictions)
    }
    
    return metrics


def validate(model, dataloader, criterion, device, verbose=False):
    """Validate the model on validation data.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on (cpu or cuda)
        
    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    
    # For metrics calculation
    all_targets = []
    all_predictions = []
    
    # No gradient computation during validation
    with torch.no_grad():
        # Use tqdm for progress bar if verbose is True
        val_bar = tqdm(dataloader, desc="Validation") if verbose else dataloader
        
        for inputs, targets in val_bar:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Ensure targets have the same shape as outputs for loss calculation
            if outputs.shape != targets.shape:
                # If targets is a 1D array and outputs is 2D, reshape targets
                if len(targets.shape) == 1 and len(outputs.shape) == 2:
                    targets = targets.view(-1, outputs.shape[1])
                # If outputs is a 2D array and targets is 1D, reshape outputs
                elif len(outputs.shape) == 2 and len(targets.shape) == 1:
                    outputs = outputs.squeeze(1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets for metric calculation
            # Ensure consistent shapes for stacking later
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # Reshape if necessary to ensure consistent dimensions
            if len(targets_np.shape) == 1:
                targets_np = targets_np.reshape(-1, 1)
            if len(outputs_np.shape) == 1:
                outputs_np = outputs_np.reshape(-1, 1)
                
            all_targets.append(targets_np)
            all_predictions.append(outputs_np)
            
            # Update progress bar
            if verbose:
                val_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Convert lists to numpy arrays
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
    metrics = {
        "loss": avg_loss,
        "mae": mae(all_targets, all_predictions),
        "mse": mse(all_targets, all_predictions),
        "rmse": rmse(all_targets, all_predictions),
        "r2": r_squared(all_targets, all_predictions),
        "mape": mape(all_targets, all_predictions),
        "smape": smape(all_targets, all_predictions),
        "explained_variance": explained_variance(all_targets, all_predictions),
        "peak_error": peak_error(all_targets, all_predictions)
    }
    
    return metrics


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, 
               num_epochs=50, device="cpu", checkpoint_dir=None, patience=10, verbose_modes=None):
    """Train the model with early stopping.
    
    Args:
        model: The neural network model
        dataloaders: Dictionary with train and val DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Maximum number of epochs to train
        device: Device to run training on (cpu or cuda)
        checkpoint_dir: Directory to save model checkpoints
        patience: Number of epochs to wait for improvement before early stopping
        verbose_modes: Dictionary with keys 'training', 'validation', 'testing' and boolean values

    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    current_train_dir = None
    run_time = time.time()
    # Create checkpoint directory if it doesn't exist
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        current_train_dir = os.path.join(checkpoint_dir, f"run_{run_time}")
        os.makedirs(current_train_dir, exist_ok=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    
    # Ensure verbose_modes is a dictionary
    if verbose_modes is None or not isinstance(verbose_modes, dict):
        verbose_modes = {'training': False, 'validation': False, 'testing': False}
    
    # Initialize history dictionary to store metrics
    history = {
        "train_loss": [], "train_mae": [], "train_mse": [], "train_rmse": [], "train_r2": [],
        "train_mape": [], "train_smape": [], "train_explained_variance": [], "train_peak_error": [],
        "val_loss": [], "val_mae": [], "val_mse": [], "val_rmse": [], "val_r2": [],
        "val_mape": [], "val_smape": [], "val_explained_variance": [], "val_peak_error": [],
        "lr": []
    }
    
    # Get start time
    start_time = time.time()
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
        if verbose_modes.get('training', False):
            logger.info(f"Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
        
        # Train for one epoch
        train_metrics = train_epoch(model, dataloaders['train'], criterion, optimizer, device, verbose=verbose_modes.get('training'))
        
        # Validate
        val_metrics = validate(model, dataloaders['val'], criterion, device, verbose=verbose_modes.get('validation'))
        
        # Update learning rate scheduler if provided
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Update history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        
        # Print metrics
        epoch_time = time.time() - epoch_start_time
        if verbose_modes.get('training', False):
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        if verbose_modes.get('training', False):
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"MAE: {train_metrics['mae']:.4f}, "
                      f"RMSE: {train_metrics['rmse']:.4f}, "
                      f"R²: {train_metrics['r2']:.4f}")
            logger.info(f"Train MAPE: {train_metrics['mape']:.2f}%, "
                  f"SMAPE: {train_metrics['smape']:.2f}%, "
                  f"Expl.Var: {train_metrics['explained_variance']:.4f}, "
                  f"Peak Err: {train_metrics['peak_error']:.4f}")
        
        if verbose_modes.get('validation', False):
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"MAE: {val_metrics['mae']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}, "
                  f"R²: {val_metrics['r2']:.4f}")
            logger.info(f"Val MAPE: {val_metrics['mape']:.2f}%, "
                  f"SMAPE: {val_metrics['smape']:.2f}%, "
                  f"Expl.Var: {val_metrics['explained_variance']:.4f}, "
                  f"Peak Err: {val_metrics['peak_error']:.4f}")
        
        # Check if this is the best model so far
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            no_improve_count = 0
            
            # Save the best model checkpoint
            if current_train_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(
                    current_train_dir, 
                    f"model_epoch{epoch+1}_valloss{val_metrics['loss']:.4f}_{timestamp}.pt"
                )
                
                # Save model state dict and metadata
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics
                }, checkpoint_path)
                
                if verbose_modes.get('training', False):
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    if verbose_modes.get('testing', False):
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Best model was at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    # Save training history
    if current_train_dir:
        history_path = os.path.join(current_train_dir, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_history = {}
            for k, v in history.items():
                serializable_history[k] = [float(x) for x in v]
            
            json.dump(serializable_history, f, indent=4)
        
        if verbose_modes.get('training', False):
            logger.info(f"Saved training history to {history_path}")
    
    return model, history, run_time

def evaluate(model, dataloader, criterion, device, verbose=False, run_time: str=None):
    # Test the model on test set
    logger.info("Evaluating model on test set...")
    
    test_metrics = validate(model, dataloader, criterion, device, verbose=verbose)
    
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    if verbose:
        logger.info(f"Test MAPE: {test_metrics['mape']:.2f}%")
        logger.info(f"Test SMAPE: {test_metrics['smape']:.2f}%")
        logger.info(f"Test Explained Variance: {test_metrics['explained_variance']:.4f}")
        logger.info(f"Test Peak Error: {test_metrics['peak_error']:.4f}")
    
    # Save test metrics
    if run_time is not None:
        test_metrics_path = os.path.join(os.path.join(config.dirs.checkpoint_dir, f"run_{run_time}"), "test_metrics.json")
        with open(test_metrics_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_metrics = {k: float(v) for k, v in test_metrics.items()}
            json.dump(serializable_metrics, f, indent=4)
    
    if verbose:
        logger.info(f"Saved test metrics to {test_metrics_path}")

def main():
    """Main function to run the training process."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)
    
    # Get dataloaders
    dataloaders = get_all_dataloaders(config)
    
    # Get sample batch to determine input size
    sample_batch, _ = next(iter(dataloaders['train']))
    input_size = sample_batch.shape[2]  # (batch_size, seq_len, input_size)
    
    # Define model
    model = GRUNetwork(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=len(config.preprocess.target_idx),
        bidirectional=False,
        return_sequences=False
    )
    model.to(device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    verbose_modes = {
        'training': False,
        'validation': False,
        'testing': False
    }
    
    # Train model
    trained_model, history, run_time = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        checkpoint_dir=config.dirs.checkpoint_dir,
        patience=15,
        verbose_modes=verbose_modes
    )

    # Evaluate model
    evaluate(
        model=trained_model,
        dataloader=dataloaders['test'],
        criterion=criterion,
        device=device,
        verbose=verbose_modes['testing'],
        run_time=run_time
    )

if __name__ == "__main__":
    main()
