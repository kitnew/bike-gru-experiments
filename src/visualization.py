import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import random

random.seed(99634652)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
PLOT_DIR = Path("experiments/plots/")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_search_results(filepath: str, is_list: bool = True) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load search results from JSON file.
    
    Args:
        filepath: Path to the JSON file with search results
        is_list: Whether the file contains a list of trials or a dictionary
        
    Returns:
        List of trial results or dictionary with results
    """
    with open(filepath, 'r') as f:
        file = json.load(f)
        return [trial for trial in file if trial.get('status') != 'failed'] if is_list else [trial for trial in file['trials']]

def plot_epochs_vs_loss(
    search_results: List[Dict[str, Any]], 
    metric: str = 'val_loss',
    top_n: int = 3,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the number of epochs vs loss metrics with the top N performers highlighted.
    
    Args:
        search_results: List of search result dictionaries
        metric: Metric to plot (default: 'val_loss')
        top_n: Number of top performers to highlight
        save_path: Path to save the plot (if None, plot is shown)
    """
    plt.figure(figsize=(12, 8))
    
    # Extract and sort results by performance (handle missing 'best_value' key)
    sorted_results = sorted(search_results, key=lambda x: x.get('best_value', float('inf')))
    
    # Normalize values for better visualization
    all_values = []
    for result in search_results:
        if metric in result['history']:
            all_values.extend(result['history'][metric])
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Function to normalize values between 0 and 1
    normalize = lambda x: (x - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    # Plot background lines (all except top N)
    for i, result in enumerate(sorted_results[top_n:]):
        if metric in result['history']:
            # Get history data
            y_values = [normalize(val) for val in result['history'][metric]]
            x_values = list(range(1, len(y_values) + 1))
            
            # Plot with low alpha (transparency)
            plt.plot(x_values, y_values, color='black', alpha=0.3, linewidth=1)
    
    # Plot top N performers with distinct colors
    cmap = plt.get_cmap('autumn', top_n)
    for i, result in enumerate(sorted_results[:top_n]):
        if metric in result['history']:
            # Get history data
            y_values = [normalize(val) for val in result['history'][metric]]
            x_values = list(range(1, len(y_values) + 1))
            
            # Format params as string for label
            param_str = ', '.join([f"{k}={v}" for k, v in 
                                 sorted(result['params'].items(), 
                                        key=lambda x: x[0])[:3]])  # Show only first 3 params
            
            # Plot highlighted line
            plt.plot(x_values, y_values,
                     color=cmap(i+1),
                     linewidth=2.5,
                     label=f"Trial {result['trial_id']} (Best: {result['best_value']:.4f}, {param_str}...)")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel(f'Normalized {metric.replace("_", " ").title()}')
    plt.title(f'Training Progress: Epochs vs {metric.replace("_", " ").title()} (Top {top_n} Highlighted)')
    plt.legend(loc='best', fontsize=10)
    
    if save_path:
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.tight_layout()
        plt.show()

def plot_hyperparameter_influence(
    search_results: List[Dict[str, Any]],
    param_name: str,
    metric: str = 'best_value',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the influence of a specific hyperparameter on a target metric.
    
    Args:
        search_results: List of search result dictionaries
        param_name: Name of the hyperparameter to analyze
        metric: Target metric to analyze against (default: best_value)
        save_path: Path to save the plot (if None, plot is shown)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract parameter values and corresponding metrics
    param_values = []
    metric_values = []
    
    for result in search_results:
        if param_name in result['params']:
            # Handle conversion of string booleans to actual booleans if needed
            param_val = result['params'][param_name]
            if isinstance(param_val, str) and param_val.lower() in ['true', 'false']:
                param_val = param_val.lower() == 'true'
            
            param_values.append(param_val)
            
            # For metrics in history, use the best value
            if metric == 'best_value':
                metric_values.append(result['best_value'])
            # For training speed, use the number of epochs until best
            elif metric == 'training_speed':
                metric_values.append(result['best_epoch'])
    
    # Plot based on parameter type
    if all(isinstance(x, (int, float)) for x in param_values):
        # For numerical parameters, use scatter plot with trend line
        plt.scatter(param_values, metric_values, alpha=0.7, s=100)
        
        # Add trend line
        try:
            z = np.polyfit(param_values, metric_values, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(param_values), max(param_values), 100)
            plt.plot(x_range, p(x_range), "--", color='red')
        except:
            pass  # Skip trend line if fitting fails
            
        plt.title(f'Influence of {param_name} on {metric.replace("_", " ").title()}')
        plt.xlabel(param_name)
    else:
        # For categorical parameters, use bar plot
        df = pd.DataFrame({
            'param': param_values,
            'metric': metric_values
        })
        
        # Group by parameter value and calculate mean and std
        grouped = df.groupby('param')['metric'].agg(['mean', 'std']).reset_index()
        
        # Plot bars with error bars
        plt.bar(
            range(len(grouped)), 
            grouped['mean'], 
            yerr=grouped['std'], 
            alpha=0.7
        )
        plt.xticks(range(len(grouped)), grouped['param'])
        plt.title(f'Effect of {param_name} on {metric.replace("_", " ").title()}')
        plt.xlabel(param_name)
    
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.tight_layout()
        plt.show()

def plot_parameter_correlation_matrix(
    search_results: List[Dict[str, Any]], 
    target_metric: str = 'best_value',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a correlation matrix between numerical hyperparameters and the target metric.
    
    Args:
        search_results: List of search result dictionaries
        target_metric: Target metric for correlation analysis (default: best_value)
        save_path: Path to save the plot (if None, plot is shown)
    """
    # Extract parameters and metric values
    data = []
    for result in search_results:
        row = {}
        for param_name, param_value in result['params'].items():
            # Include only numerical parameters
            if isinstance(param_value, (int, float)) or (
                isinstance(param_value, str) and param_value.replace('.', '', 1).isdigit()):
                try:
                    row[param_name] = float(param_value)
                except:
                    pass
        
        # Add target metric
        if target_metric == 'best_value':
            row[target_metric] = result['best_value']
        elif target_metric == 'training_speed':
            row[target_metric] = result['best_epoch']
        
        data.append(row)
    
    # Create DataFrame and calculate correlation
    df = pd.DataFrame(data)
    corr = df.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f"
    )
    
    plt.title(f'Correlation Between Parameters and {target_metric.replace("_", " ").title()}')
    
    if save_path:
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.tight_layout()
        plt.show()

def plot_learning_curves(
    search_results: List[Dict[str, Any]],
    trial_ids: List[int] = None,
    metrics: List[str] = ['train_loss', 'val_loss', 'train_mae', 'val_mae'],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot learning curves for selected trials.
    
    Args:
        search_results: List of search result dictionaries
        trial_ids: List of trial IDs to plot (if None, plots top 3 by best_value)
        metrics: List of metrics to plot
        save_path: Path to save the plot (if None, plot is shown)
    """
    # If no trial_ids provided, use top 3 by best_value
    if trial_ids is None:
        sorted_results = sorted(search_results, key=lambda x: x['best_value'])
        trial_ids = [r['trial_id'] for r in sorted_results[:3]]
    
    # Get number of metrics to determine subplot layout
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Create color map for trials
    cmap = plt.get_cmap('viridis', len(trial_ids) + 1)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, trial_id in enumerate(trial_ids):
            trial = next((r for r in search_results if r['trial_id'] == trial_id), None)
            if trial and metric in trial['history']:
                values = trial['history'][metric]
                epochs = range(1, len(values) + 1)
                
                # Create label with key parameters
                param_str = ', '.join([f"{k}={v}" for k, v in 
                                     sorted(trial['params'].items(), 
                                            key=lambda x: x[0])[:2]])  # Show only first 2 params
                
                ax.plot(epochs, values, 
                         marker='o', markersize=4, 
                         color=cmap(j), 
                         linewidth=2, 
                         label=f"Trial {trial_id} ({param_str}...)")
                
                # Mark best epoch if it's a loss metric
                if 'loss' in metric.lower() and 'best_epoch' in trial:
                    best_epoch = trial['best_epoch']
                    if best_epoch < len(values):
                        ax.axvline(x=best_epoch + 1, color=cmap(j), linestyle='--', alpha=0.5)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Learning Curves for Selected Trials', fontsize=16)
    
    if save_path:
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.tight_layout()
        plt.show()

def plot_search_progress(
    results: List[Dict[str, Any]],
    title: str = 'Search Progress',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the progress of optimization algorithm over trials.
    
    Args:
        results: List of trial results
        save_path: Path to save the plot (if None, plot is shown)
    """
    plt.figure(figsize=(12, 6))
    
    # Extract trial information
    trial_nums = []
    values = []
    best_so_far = []
    
    current_best = float('inf')
    
    for trial in results:
        trial_nums.append(trial.get('number', trial.get('trial_id')))
        values.append(trial.get('value', trial.get('best_value')))
        
        # Update best value seen so far
        if trial.get('value', trial.get('best_value')) < current_best:
            current_best = trial.get('value', trial.get('best_value'))
        best_so_far.append(current_best)
    
    # Plot individual trial values
    plt.plot(trial_nums, values, 'o-', alpha=0.6, label='Trial Value')
    
    # Plot best value seen so far
    plt.plot(trial_nums, best_so_far, 'r-', linewidth=2, label='Best So Far')
    
    # Highlight final best value
    best_trial_idx = best_so_far.index(min(best_so_far))
    plt.scatter([trial_nums[best_trial_idx]], [best_so_far[best_trial_idx]], 
                 color='red', s=100, zorder=5, label='Best Trial')
    
    plt.xlabel('Trial Number')
    plt.ylabel('Best Value')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    if save_path:
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.tight_layout()
        plt.show()



# Example usage
def plot_metrics_comparison(history: Dict[str, List[float]], metrics: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot multiple metrics for comparison in a single plot with shared x-axis but separate y-axes.
    
    Args:
        history: Dictionary containing metric name as key and list of values as value
        metrics: List of metric names to include in the plot
        save_path: Path to save the plot (if None, plot is shown)
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Create color map for different metrics
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    axes = [ax1] + [ax1.twinx() for _ in range(len(metrics) - 1)]
    
    # Offset the right spines for additional y-axes
    for i, ax in enumerate(axes[2:], 2):
        ax.spines['right'].set_position(('outward', (i - 1) * 60))
    
    # Plot each metric on its own y-axis
    for i, (metric, color, ax) in enumerate(zip(metrics, colors, axes)):
        if metric in history:
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            line, = ax.plot(epochs, values, color=color, linewidth=2.5, marker='o', markersize=4,
                     label=metric.replace('_', ' ').title())
            
            # Set y-axis label with matching color
            ax.set_ylabel(metric.replace('_', ' ').title(), color=color)
            ax.tick_params(axis='y', colors=color)
            
            # Draw horizontal line at best value with annotation
            best_value = min(values) if 'loss' in metric.lower() else max(values)
            best_epoch = values.index(best_value) + 1
            ax.axhline(y=best_value, color=color, linestyle='--', alpha=0.5)
            ax.annotate(f'Best: {best_value:.4f} (Epoch {best_epoch})',
                       xy=(best_epoch, best_value),
                       xytext=(best_epoch + 1, best_value * (0.95 if 'loss' in metric.lower() else 1.05)),
                       color=color, fontsize=9)
    
    # Configure plot
    ax1.set_xlabel('Epochs')
    ax1.set_title('Model Performance Metrics Over Training Epochs')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create a single legend for all metrics
    lines = [ax.get_lines()[0] for ax in axes if len(ax.get_lines()) > 0]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(metrics))
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_train_val_comparison(history: Dict[str, List[float]], metrics_base: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot comparison between training and validation metrics.
    
    Args:
        history: Dictionary containing metric name as key and list of values as value
        metrics_base: List of base metric names (without train_ or val_ prefix)
        save_path: Path to save the plot (if None, plot is shown)
    """
    n_metrics = len(metrics_base)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, base_metric in enumerate(metrics_base):
        train_metric = f'train_{base_metric}'
        val_metric = f'val_{base_metric}'
        
        if train_metric in history and val_metric in history:
            train_values = history[train_metric]
            val_values = history[val_metric]
            epochs = range(1, len(train_values) + 1)
            
            ax = axes[i]
            
            # Plot training and validation metrics
            ax.plot(epochs, train_values, 'b-', linewidth=2, label=f'Training {base_metric}')
            ax.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation {base_metric}')
            
            # Highlight the best validation epoch
            best_val = min(val_values) if 'loss' in base_metric else max(val_values)
            best_epoch = val_values.index(best_val) + 1
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
            ax.axhline(y=best_val, color='g', linestyle='--', alpha=0.7)
            
            # Add annotation for best value
            ax.annotate(f'Best Validation: {best_val:.4f} (Epoch {best_epoch})',
                       xy=(best_epoch, best_val),
                       xytext=(best_epoch + 2, best_val * (0.95 if 'loss' in base_metric else 1.05)),
                       fontsize=10, color='green',
                       arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7))
            
            # Configure subplot
            ax.set_title(f'{base_metric.replace("_", " ").title()} During Training')
            ax.set_ylabel(base_metric.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
    
    # Configure overall plot
    axes[-1].set_xlabel('Epochs')
    plt.suptitle('Training vs Validation Metrics', fontsize=16)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_error_metrics_evolution(history: Dict[str, List[float]], metrics: List[str] = ['mape', 'smape', 'peak_error'], save_path: Optional[str] = None) -> None:
    """
    Plot the evolution of error metrics over training epochs.
    
    Args:
        history: Dictionary containing metric name as key and list of values as value
        metrics: List of error metrics to plot (default: ['mape', 'smape', 'peak_error'])
        save_path: Path to save the plot (if None, plot is shown)
    """
    plt.figure(figsize=(14, 8))
    
    for metric in metrics:
        train_metric = f'train_{metric}'
        val_metric = f'val_{metric}'

        if train_metric in history and val_metric in history:
            train_values = [metric * 100 for metric in history[train_metric]] if history[train_metric][-1] < 1 else history[train_metric]
            val_values = [metric * 100 for metric in history[val_metric]] if history[val_metric][-1] < 1 else history[val_metric]
            epochs = range(1, len(train_values) + 1)
            
            plt.plot(epochs, train_values, marker='o', markersize=3, linewidth=2, label=f'Train {metric.upper()}')
            plt.plot(epochs, val_values, marker='s', markersize=3, linewidth=2, label=f'Val {metric.upper()}')
    
    plt.title('Error Metrics Evolution During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Error Value (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_model_radar_chart(history: Dict[str, List[float]], epoch: int = -1, save_path: Optional[str] = None) -> None:
    """
    Create a radar chart showing multiple performance metrics at a specific epoch.
    
    Args:
        history: Dictionary containing metric name as key and list of values as value
        epoch: Epoch index to plot (default: -1, meaning the last epoch)
        save_path: Path to save the plot (if None, plot is shown)
    """
    # Define metrics to include in radar chart
    radar_metrics = {
        'val_r2': {'name': 'R\u00b2', 'scale': 1.0, 'is_higher_better': True},
        'val_explained_variance': {'name': 'Explained Variance', 'scale': 1.0, 'is_higher_better': True},
        'val_mape': {'name': 'MAPE', 'scale': 10.0, 'is_higher_better': False},  # Lower is better, invert
        'val_smape': {'name': 'SMAPE', 'scale': 10.0, 'is_higher_better': False},  # Lower is better, invert
        'val_mae': {'name': 'MAE', 'scale': 10.0, 'is_higher_better': False},  # Lower is better, invert
        'val_rmse': {'name': 'RMSE', 'scale': 10.0, 'is_higher_better': False},  # Lower is better, invert
        'val_peak_error': {'name': 'Peak Error', 'scale': 5.0, 'is_higher_better': False}  # Lower is better, invert
    }
    
    # Extract values for the selected epoch
    metrics_values = {}
    for metric in radar_metrics:
        if metric in history:
            # Get value (use last epoch if epoch=-1)
            value = history[metric][epoch] if epoch != -1 else history[metric][-1]
            
            # Apply scaling and inversion for visualization
            scale = radar_metrics[metric]['scale']
            if radar_metrics[metric]['is_higher_better']:
                # For higher-is-better metrics, scale directly
                metrics_values[metric] = value * scale
            else:
                # For lower-is-better metrics, invert so that higher values on chart = better performance
                # We use 1 - normalized value, with a floor of 0
                max_val = max(history[metric]) * 1.1  # Add 10% margin
                normalized = max(0, 1 - (value / max_val)) * scale
                metrics_values[metric] = normalized

    
    # Check if we have metrics to plot
    if not metrics_values:
        print("No metrics available for radar chart")
        return
    
    # Set up the radar chart
    metrics = list(metrics_values.keys())
    values = [metrics_values[m] for m in metrics]
    labels = [radar_metrics[m]['name'] for m in metrics]
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric (divide the plot into equal parts)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygon by repeating the first value
    values.append(values[0])
    angles.append(angles[0])
    labels.append(labels[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
    
    # Add a title
    epoch_label = "Last Epoch" if epoch == -1 else f"Epoch {epoch+1}"
    plt.title(f'Model Performance Metrics at {epoch_label}', size=15)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_learning_rate_schedule(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot the learning rate schedule used during training.
    
    Args:
        history: Dictionary containing metric name as key and list of values as value
        save_path: Path to save the plot (if None, plot is shown)
    """
    if 'lr' not in history:
        print("Learning rate history not available")
        return
    
    lr_values = history['lr']
    epochs = range(1, len(lr_values) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, lr_values, 'b-', linewidth=2.5, marker='o', markersize=4)
    
    # Find when learning rate changes
    lr_changes = [i for i in range(1, len(lr_values)) if lr_values[i] != lr_values[i-1]]
    for change in lr_changes:
        plt.axvline(x=change + 1, color='r', linestyle='--', alpha=0.7)
        plt.annotate(f'LR = {lr_values[change]:.6f}',
                    xy=(change + 1, lr_values[change]),
                    xytext=(change + 1, lr_values[change] * 1.1),
                    fontsize=10, color='red')
    
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def visualize_best_model(results: List[Dict[str, Any]], output_dir: str = "experiments/plots/best_model/") -> None:
    """
    Create a comprehensive set of visualizations for the best model.
    
    Args:
        results: List containing the best model results dictionary
        output_dir: Directory to save plots (default: "experiments/plots/best_model/")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if results are valid
    if not results or not isinstance(results, list) or len(results) == 0:
        print("No valid results found for best model visualization")
        return
    
    # We assume there's only one trial in the best model results
    model_result = results[0]
    history = model_result.get('history', {})
    
    if not history:
        print("No training history found in the best model results")
        return
        
    print(f"Loaded best model results with {len(history.get('train_loss', []))} epochs")
    
    # 1. Overall metrics comparison
    plot_metrics_comparison(
        history, 
        metrics=['train_loss', 'val_loss', 'train_r2', 'val_r2'],
        save_path=os.path.join(output_dir, "metrics_comparison.png")
    )
    
    # 2. Training vs Validation comparison for each metric
    for base_metric in ['loss', 'mae', 'rmse', 'r2']:
        plot_train_val_comparison(
            history,
            metrics_base=[base_metric],
            save_path=os.path.join(output_dir, f"{base_metric}_train_val_comparison.png")
        )
    
    # 3. Error metrics evolution
    plot_error_metrics_evolution(
        history,
        save_path=os.path.join(output_dir, "error_metrics_evolution.png")
    )
    
    # 4. Learning rate schedule
    if 'lr' in history:
        plot_learning_rate_schedule(
            history,
            save_path=os.path.join(output_dir, "learning_rate_schedule.png")
        )
    
    # 5. Radar chart for final model performance
    plot_model_radar_chart(
        history,
        epoch=-1,  # Last epoch
        save_path=os.path.join(output_dir, "model_radar_chart.png")
    )
    
    # 6. Best epoch radar chart
    best_epoch = model_result.get('best_epoch', 0)
    if best_epoch > 0 and best_epoch < len(history.get('val_loss', [])):
        plot_model_radar_chart(
            history,
            epoch=best_epoch,
            save_path=os.path.join(output_dir, "best_epoch_radar_chart.png")
        )
    
    print(f"All best model visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Load search results
    random_search_path = "/home/kitne/University/2lvl/SU/bike-gru-experiments/experiments/checkpoints/search/random_search/all_results.json"
    bayesian_search_path = "/home/kitne/University/2lvl/SU/bike-gru-experiments/experiments/checkpoints/search/bayesian_search/study_results.json"
    grid_search_path = "/home/kitne/University/2lvl/SU/bike-gru-experiments/experiments/checkpoints/search/grid_search/all_results.json"

    try:
        random_results = load_random_search_results(random_search_path)
        print(f"Loaded {len(random_results)} random search trials")
        
        # Example visualizations
        plot_epochs_vs_loss(
            random_results, 
            metric='val_loss',
            save_path="experiments/plots/epochs_vs_loss.png"
        )
        
        plot_hyperparameter_influence(
            random_results,
            param_name='hidden_size',
            metric='best_value',
            save_path="experiments/plots/hidden_size_influence.png"
        )
        
        plot_parameter_correlation_matrix(
            random_results,
            save_path="experiments/plots/parameter_correlation.png"
        )
        
        plot_learning_curves(
            random_results,
            save_path="experiments/plots/learning_curves.png"
        )
        
    except Exception as e:
        print(f"Error with random search visualization: {e}")
    
    try:
        bayesian_results = load_bayesian_search_results(bayesian_search_path)
        print(f"Loaded Bayesian search results with {len(bayesian_results['trials'])} trials")
        
        # Convert Bayesian results to format compatible with our visualization functions
        bayesian_converted = []
        for trial in bayesian_results['trials']:
            trial_dict = {
                'trial_id': trial['number'],
                'params': trial['params'],
                'best_value': trial['value'],
                'best_epoch': 0,  # Not available in Bayesian results
                'history': {}     # Not available in Bayesian results
            }
            bayesian_converted.append(trial_dict)
        
        plot_bayesian_search_progress(
            bayesian_results,
            save_path="experiments/plots/bayesian_progress.png"
        )
        
        plot_parallel_coordinates(
            bayesian_converted,
            save_path="experiments/plots/parallel_coordinates.png"
        )
        
    except Exception as e:
        print(f"Error with Bayesian search visualization: {e}")

    try:
        grid_results = load_random_search_results(grid_search_path)
        print(f"Loaded {len(grid_results)} grid search trials")
    
        # Example visualizations
        plot_epochs_vs_loss(
            grid_results, 
            metric='val_loss',
            save_path="experiments/plots/epochs_vs_loss_grid.png"
        )
    
        plot_hyperparameter_influence(
            grid_results,
            param_name='hidden_size',
            metric='best_value',
            save_path="experiments/plots/hidden_size_influence_grid.png"
        )
    
        plot_parameter_correlation_matrix(
            grid_results,
            save_path="experiments/plots/parameter_correlation_grid.png"
        )
    
        plot_learning_curves(
            grid_results,
            save_path="experiments/plots/learning_curves_grid.png"
        )
        
    except Exception as e:
        print(f"Error with grid search visualization: {e}")
    
    # Visualize best model results
    best_model_path = "/home/kitne/University/2lvl/SU/bike-gru-experiments/experiments/checkpoints/run_1746652725.7987037/all_results.json"
    try:
        best_model_results = load_search_results(best_model_path)
        visualize_best_model(best_model_results)
    except Exception as e:
        print(f"Error with best model visualization: {e}")