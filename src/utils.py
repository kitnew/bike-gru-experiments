import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
import numpy as np


def load_json_results(filepath: str) -> Dict[str, Any]:
    """Load JSON results from a file."""
    try:
        with open(filepath, 'r') as f:
            file = json.load(f)
            return file
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None


def find_trials_in_directory(directory_path: str) -> List[str]:
    """Find all trial result JSON files in a directory."""
    path = Path(directory_path)
    result_files = list(path.glob("*_result.json"))
    return [str(file) for file in result_files]


def extract_trial_results(file_path: str) -> Dict[str, Any]:
    """Extract relevant trial information from a JSON result file."""
    data = load_json_results(file_path)
    if data is None:
        return None
    
    return {
        'trial_id': data.get('trial_id'),
        'params': data.get('params', {}),
        'best_value': data.get('best_value'),
        'metric': data.get('metric', 'unknown'),
        'best_epoch': data.get('best_epoch'),
        'file_path': file_path
    }

def extract_multiple_trial_results(file_path: str) -> List[Dict[str, Any]]:
    data = load_json_results(file_path)
    if data is None:
        return None

    if file_path.endswith("study_results.json"):
        data = data['trials']
        return [
            {
                'trial_id': trial.get('number'),
                'params': trial.get('params', {}),
                'best_value': float(trial.get('value')),
                'best_epoch': trial.get('state', {}),
                'file_path': file_path
            }
            for trial in data
        ]
    
    return [
        {
            'trial_id': trial.get('trial_id'),
            'params': trial.get('params', {}),  
            'best_value': float(trial.get('best_value', 1)),
            'metric': trial.get('metric', 'unknown'),
            'best_epoch': trial.get('best_epoch'),
            'file_path': file_path
        }
        for trial in data
    ]


def get_best_trials(json_file_paths: List[str], metric_direction: str = 'minimize', num_best: int = 3) -> List[Dict[str, Any]]:
    """
    Extract and rank trial results from multiple JSON files.
    
    Args:
        json_file_paths: List of paths to JSON result files
        metric_direction: 'minimize' or 'maximize' the metric
        num_best: Number of top trials to return
        
    Returns:
        List of the top trial results
    """
    all_trials = []
    
    # Process each file
    for file_path in json_file_paths:
        # If the file is a directory, find all trial results in it
        if os.path.isdir(file_path):
            dir_trials = find_trials_in_directory(file_path)
            for trial_path in dir_trials:
                trial_data = extract_trial_results(trial_path)
                if trial_data:
                    all_trials.append(trial_data)
        elif file_path.endswith("results.json"):
            trial_data = extract_multiple_trial_results(file_path)
            if trial_data:
                all_trials.extend(trial_data)
        else:
            # Process individual file
            trial_data = extract_trial_results(file_path)
            if trial_data:
                all_trials.append(trial_data)
    
    # Sort trials by metric value
    if metric_direction.lower() == 'minimize':
        sorted_trials = sorted(all_trials, key=lambda x: x['best_value'])
    else:
        sorted_trials = sorted(all_trials, key=lambda x: x['best_value'], reverse=True)
    
    # Return the top N trials
    return sorted_trials[:num_best]



def format_parameter_output(trial_data: Dict[str, Any]) -> str:
    """Format parameter data for readable output."""
    params_str = json.dumps(trial_data['params'], indent=2)
    return (
        f"Trial ID: {trial_data['trial_id']}\n"
        f"Best Value: {trial_data['best_value']:.6f}\n"
        f"Best Epoch: {trial_data['best_epoch']}\n"
        f"File: {os.path.basename(trial_data['file_path'])}\n"
        f"Parameters:\n{params_str}\n"
    )


def extract_best_parameter_values(best_trials: List[Dict[str, Any]], max_values_per_param: int = 3) -> Dict[str, List[Any]]:
    """
    Extract the top parameter values from the best trials for each parameter.
    This helps identify promising parameter ranges for focused grid search.
    
    Args:
        best_trials: List of best trial results from get_best_trials()
        max_values_per_param: Maximum number of unique values to extract per parameter
        
    Returns:
        Dictionary mapping parameter names to their top values
    """
    if not best_trials:
        return {}
    
    # Extract all parameter names
    param_names = set()
    for trial in best_trials:
        param_names.update(trial['params'].keys())
    
    # For each parameter, collect all values from the best trials
    param_values = {param: [] for param in param_names}
    for trial in best_trials:
        for param, value in trial['params'].items():
            if isinstance(value, bool):
                value = f'{value}'
            param_values[param].append((value, trial['best_value']))
    
    # For each parameter, select the top values based on associated trial performance
    best_param_values = {}
    for param, values in param_values.items():
        # Sort unique values by their associated trial performance
        unique_values = {}
        for value, best_value in values:
            # For each unique parameter value, keep the best trial performance
            if value not in unique_values or best_value < unique_values[value]:
                unique_values[value] = best_value
        
        # Sort parameter values by their best associated trial performance
        sorted_values = sorted(unique_values.items(), key=lambda x: x[1])
        
        # Select the top values (up to max_values_per_param)
        best_param_values[param] = [v for v, _ in sorted_values[:max_values_per_param]]
    
    return best_param_values


def main():
    parser = argparse.ArgumentParser(description='Find the best trials from experiment JSON files.')
    parser.add_argument('json_files', nargs='+', help='Paths to JSON files or directories containing trial results')
    parser.add_argument('--direction', choices=['minimize', 'maximize'], default='minimize',
                        help='Direction of optimization (minimize or maximize)')
    parser.add_argument('--top', type=int, default=3, help='Number of top trials to show')
    parser.add_argument('--median', action='store_true', help='Compute and display median of best results')
    
    args = parser.parse_args()
    
    print(f"Finding top {args.top} trials ({args.direction} metric)...")
    best_trials = get_best_trials(args.json_files, args.direction, args.top)
    
    if not best_trials:
        print("No valid trials found.")
        return
    
    print(f"\nTop {len(best_trials)} Parameter Sets:")
    print("-" * 50)
    for i, trial in enumerate(best_trials, 1):
        print(f"Rank #{i}:")
        print(format_parameter_output(trial))
        print("-" * 50)
    
    # Extract and display best parameter values from the top trials
    best_param_values = extract_best_parameter_values(best_trials, max_values_per_param=3)
    if best_param_values:
        print("\nBest Parameter Values for Grid Search:")
        print("-" * 50)
        for param, values in best_param_values.items():
            print(f"{param}: {values}")
        print("-" * 50)


if __name__ == "__main__":
    main()