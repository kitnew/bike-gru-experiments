import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization')

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_training_history(history_path: str) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file.
    
    Args:
        history_path: Path to the training history JSON file
        
    Returns:
        Dictionary containing training history metrics
    """
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        logger.info(f"Loaded training history from {history_path}")
        return history
    except Exception as e:
        logger.error(f"Error loading training history from {history_path}: {str(e)}")
        raise


def plot_training_metrics(history: Dict[str, List[float]], 
                         metrics: List[str] = ['loss'], 
                         title: str = "Training Metrics",
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """
    Plot training metrics (train and validation) over epochs.
    
    Args:
        history: Dictionary containing training history metrics
        metrics: List of metrics to plot (e.g., ['loss', 'mae', 'rmse'])
        title: Title for the plot
        save_path: Path to save the plot (if None, plot is not saved)
        show_plot: Whether to display the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics), sharex=True)
    
    # Convert to list of axes if there's only one metric
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history[f"train_{metrics[0]}"])+1)
    
    for i, metric in enumerate(metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history and val_key in history:
            ax = axes[i]
            
            # Plot training and validation metrics
            ax.plot(epochs, history[train_key], 'b-', label=f'Training {metric}')
            ax.plot(epochs, history[val_key], 'r-', label=f'Validation {metric}')
            
            # Add labels and legend
            ax.set_title(f'{metric.upper()} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.legend()
            
            # Format y-axis for percentage metrics
            if metric.lower() in ['mape', 'smape']:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            logger.warning(f"Metric {metric} not found in training history")
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(history: Dict[str, List[float]],
                      title: str = "Learning Rate Schedule",
                      save_path: Optional[str] = None,
                      show_plot: bool = True) -> None:
    """
    Plot learning rate changes over epochs.
    
    Args:
        history: Dictionary containing training history metrics
        title: Title for the plot
        save_path: Path to save the plot (if None, plot is not saved)
        show_plot: Whether to display the plot
    """
    if 'lr' not in history:
        logger.warning("Learning rate not found in training history")
        return
    
    epochs = range(1, len(history['lr'])+1)
    
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['lr'], 'g-', marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_configurations(histories: Dict[str, Dict[str, List[float]]],
                          metric: str = 'loss',
                          train_val: str = 'val',
                          title: str = "Configuration Comparison",
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Compare multiple configurations on the same plot.
    
    Args:
        histories: Dictionary mapping configuration names to their training histories
        metric: Metric to compare (e.g., 'loss', 'mae', 'rmse')
        train_val: Whether to plot 'train' or 'val' metrics
        title: Title for the plot
        save_path: Path to save the plot (if None, plot is not saved)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Set color palette for different configurations
    colors = sns.color_palette("husl", len(histories))
    
    for i, (config_name, history) in enumerate(histories.items()):
        metric_key = f"{train_val}_{metric}"
        
        if metric_key in history:
            epochs = range(1, len(history[metric_key])+1)
            plt.plot(epochs, history[metric_key], color=colors[i], label=config_name)
        else:
            logger.warning(f"Metric {metric_key} not found in history for {config_name}")
    
    plt.title(f"{title} - {train_val.capitalize()} {metric.upper()}")
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    
    # Format y-axis for percentage metrics
    if metric.lower() in ['mape', 'smape']:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_search_results(search_dir: str,
                          metric: str = 'val_rmse',
                          param_to_compare: str = 'hidden_size',
                          title: str = "Hyperparameter Search Results",
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Compare results from a hyperparameter search.
    
    Args:
        search_dir: Directory containing search results
        metric: Metric to compare (e.g., 'val_loss', 'val_rmse')
        param_to_compare: Parameter to compare on x-axis
        title: Title for the plot
        save_path: Path to save the plot (if None, plot is not saved)
        show_plot: Whether to display the plot
    """
    # Load all_results.json
    results_path = os.path.join(search_dir, "all_results.json")
    
    try:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        # Filter completed trials
        completed_results = [r for r in all_results if r['status'] == 'completed']
        
        if not completed_results:
            logger.warning("No completed trials found")
            return
        
        # Extract parameter values and metric values
        param_values = []
        metric_values = []
        
        for result in completed_results:
            if param_to_compare in result['params']:
                param_values.append(result['params'][param_to_compare])
                metric_values.append(result['best_value'])
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Check if parameter values are numeric
        if all(isinstance(x, (int, float)) for x in param_values):
            # Sort by parameter value for line plot
            sorted_indices = np.argsort(param_values)
            sorted_param_values = [param_values[i] for i in sorted_indices]
            sorted_metric_values = [metric_values[i] for i in sorted_indices]
            
            # Create scatter plot with connecting line
            plt.plot(sorted_param_values, sorted_metric_values, 'o-', markersize=8)
            
            # Add value annotations
            for x, y in zip(sorted_param_values, sorted_metric_values):
                plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                             xytext=(0, 10), ha='center')
        else:
            # For non-numeric parameters, create bar plot
            unique_params = list(set(param_values))
            param_to_metrics = {param: [] for param in unique_params}
            
            for param, metric_val in zip(param_values, metric_values):
                param_to_metrics[param].append(metric_val)
            
            # Calculate mean metric value for each parameter
            mean_metrics = [np.mean(param_to_metrics[param]) for param in unique_params]
            
            # Create bar plot
            plt.bar(range(len(unique_params)), mean_metrics)
            plt.xticks(range(len(unique_params)), unique_params)
            
            # Add value annotations
            for i, metric_val in enumerate(mean_metrics):
                plt.annotate(f"{metric_val:.4f}", (i, metric_val), textcoords="offset points", 
                             xytext=(0, 10), ha='center')
        
        plt.title(f"{title} - {metric}")
        plt.xlabel(param_to_compare)
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        logger.error(f"Error processing search results: {str(e)}")


def main():
    """
    Example usage of visualization functions.
    """
    from config import Config
    
    # Load configuration
    config = Config.from_yaml("/home/kitne/University/2lvl/SU/bike-gru-experiments/config/default.yaml")
    
    # Example 1: Plot metrics for a single training run
    checkpoint_dir = config.dirs.checkpoint_dir
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    
    if os.path.exists(history_path):
        history = load_training_history(history_path)
        
        # Plot loss
        plot_training_metrics(
            history=history,
            metrics=['loss'],
            title="Training and Validation Loss",
            save_path=os.path.join(config.dirs.plots_dir, "loss_curves.png"),
            show_plot=True
        )
        
        # Plot multiple metrics
        plot_training_metrics(
            history=history,
            metrics=['mae', 'rmse', 'r2'],
            title="Training and Validation Metrics",
            save_path=os.path.join(config.dirs.plots_dir, "metrics_curves.png"),
            show_plot=True
        )
        
        # Plot learning rate
        plot_learning_rate(
            history=history,
            title="Learning Rate Schedule",
            save_path=os.path.join(config.dirs.plots_dir, "learning_rate.png"),
            show_plot=True
        )
    else:
        logger.warning(f"Training history not found at {history_path}")
    
    # Example 2: Compare multiple configurations
    # This would typically come from different training runs
    # For demonstration, we'll create synthetic data
    
    # In a real scenario, you would load multiple history files
    # histories = {
    #     "hidden_size=32": load_training_history("path/to/history1.json"),
    #     "hidden_size=64": load_training_history("path/to/history2.json"),
    #     "hidden_size=128": load_training_history("path/to/history3.json"),
    # }
    
    # For demonstration purposes only:
    if os.path.exists(history_path):
        base_history = load_training_history(history_path)
        
        # Create synthetic histories with variations
        histories = {
            "hidden_size=32": {k: [v * 1.2 for v in vals] for k, vals in base_history.items()},
            "hidden_size=64": base_history,
            "hidden_size=128": {k: [v * 0.9 for v in vals] for k, vals in base_history.items()},
        }
        
        # Compare configurations
        compare_configurations(
            histories=histories,
            metric='loss',
            train_val='val',
            title="Effect of Hidden Size on Validation Loss",
            save_path=os.path.join(config.dirs.plots_dir, "config_comparison_loss.png"),
            show_plot=True
        )
        
        compare_configurations(
            histories=histories,
            metric='rmse',
            train_val='val',
            title="Effect of Hidden Size on Validation RMSE",
            save_path=os.path.join(config.dirs.plots_dir, "config_comparison_rmse.png"),
            show_plot=True
        )
    
    # Example 3: Visualize hyperparameter search results
    search_dir = os.path.join(config.dirs.logs_dir, "search", "random_search_example")
    
    if os.path.exists(os.path.join(search_dir, "all_results.json")):
        compare_search_results(
            search_dir=search_dir,
            metric='val_rmse',
            param_to_compare='hidden_size',
            title="Effect of Hidden Size on Model Performance",
            save_path=os.path.join(config.dirs.plots_dir, "search_hidden_size.png"),
            show_plot=True
        )
        
        compare_search_results(
            search_dir=search_dir,
            metric='val_rmse',
            param_to_compare='learning_rate',
            title="Effect of Learning Rate on Model Performance",
            save_path=os.path.join(config.dirs.plots_dir, "search_learning_rate.png"),
            show_plot=True
        )
    else:
        logger.warning(f"Search results not found at {search_dir}")


if __name__ == "__main__":
    main()