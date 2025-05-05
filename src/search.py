import os
import json
import time
import logging
import itertools
import numpy as np
import torch
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from functools import partial

# For Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Bayesian optimization will not be available.")
    print("Install with: pip install optuna")

from src.config import Config
from src.train import train_model
from src.data_loader import get_all_dataloaders
from src.model import GRUNetwork

config = Config.from_yaml("/home/kitne/University/2lvl/SU/bike-gru-experiments/config/default.yaml")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.dirs.logs_dir, f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('search')


class HyperparameterSearch:
    """Class for performing hyperparameter search using different strategies."""
    
    def __init__(self, 
                 base_config_path: str,
                 param_grid: Dict[str, List[Any]],
                 search_method: str = 'grid',
                 n_trials: int = 10,
                 metric: str = 'val_loss',
                 direction: str = 'minimize',
                 n_jobs: int = 1,
                 experiment_name: str = None,
                 verbose: bool = False):
        """
        Initialize the hyperparameter search.
        
        Args:
            base_config_path: Path to the base configuration YAML file
            param_grid: Dictionary with parameter names as keys and lists of parameter values to try
            search_method: Search method to use ('grid', 'random', or 'bayesian')
            n_trials: Number of trials for random and bayesian search
            metric: Metric to optimize ('val_loss', 'val_rmse', 'val_r2', etc.)
            direction: Direction of optimization ('minimize' or 'maximize')
            n_jobs: Number of parallel jobs to run (1 means sequential)
            experiment_name: Name for this experiment (used for saving results)
            verbose: Whether to print verbose output
        """
        self.base_config_path = base_config_path
        self.param_grid = param_grid
        self.search_method = search_method.lower()
        self.n_trials = n_trials
        self.metric = metric
        self.direction = direction
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Validate search method
        valid_methods = ['grid', 'random', 'bayesian']
        if self.search_method not in valid_methods:
            raise ValueError(f"Search method must be one of {valid_methods}")
        
        # Check if Bayesian search is requested but optuna is not available
        if self.search_method == 'bayesian' and not OPTUNA_AVAILABLE:
            raise ImportError("Bayesian search requires optuna. Please install with: pip install optuna")
        
        # Set up experiment name and directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"search_{self.search_method}_{timestamp}"
        
        # Load base configuration
        self.base_config = Config.from_yaml(base_config_path)
        
        # Create experiment directory
        self.experiment_dir = os.path.join(self.base_config.dirs.checkpoint_dir, "search", self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set up results tracking
        self.results = []
        
        logger.info(f"Initialized {self.search_method} search with {len(self._get_all_combinations()) if self.search_method == 'grid' else n_trials} configurations")
        if self.verbose:
            logger.info(f"Experiment directory: {self.experiment_dir}")
            logger.info(f"Optimizing for {self.metric} ({self.direction})")
    
    def _get_all_combinations(self) -> List[Dict[str, Any]]:
        """Get all combinations of parameters for grid search."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _get_random_combinations(self, n: int) -> List[Dict[str, Any]]:
        """Get n random combinations of parameters for random search."""
        combinations = []
        for _ in range(n):
            params = {}
            for key, values in self.param_grid.items():
                # Handle different types of parameter spaces
                if isinstance(values, list):
                    # Discrete choice
                    params[key] = np.random.choice(values).item() if isinstance(np.random.choice(values), np.number) else np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 3 and values[0] == 'log_uniform':
                    # Log-uniform distribution between min and max
                    low, high = values[1], values[2]
                    params[key] = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
                elif isinstance(values, tuple) and len(values) == 3 and values[0] == 'int_uniform':
                    # Uniform integer distribution between min and max
                    low, high = values[1], values[2]
                    params[key] = int(np.random.randint(low, high + 1))
                elif isinstance(values, tuple) and len(values) == 3:
                    # Uniform distribution between min and max
                    low, high = values[1], values[2]
                    params[key] = float(np.random.uniform(low, high))
                else:
                    raise ValueError(f"Unsupported parameter space for {key}: {values}")
            combinations.append(params)
        return combinations
    
    def _update_config_with_params(self, params: Dict[str, Any]) -> Config:
        """Create a new config with updated parameters."""
        # Start with a fresh copy of the base config
        config = Config.from_yaml(self.base_config_path)
        
        # Update config with parameters
        for param_name, param_value in params.items():
            # Handle nested parameters with dot notation (e.g., 'model.hidden_size')
            parts = param_name.split('.')
            
            # Navigate to the right part of the config
            current = config
            for part in parts[:-1]:
                if not hasattr(current, part):
                    setattr(current, part, type('', (), {})())  # Create empty object
                current = getattr(current, part)
            
            # Set the parameter value
            setattr(current, parts[-1], param_value)
        
        return config
    
    def _evaluate_config(self, params: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        """Evaluate a single configuration."""
        start_time = time.time()
        
        # Create a unique name for this trial
        trial_name = f"trial_{trial_id:04d}"
        if self.verbose:
            logger.info(f"Starting {trial_name} with params: {params}")
        else:
            logger.info(f"Starting {trial_name}")
        
        # Update config with parameters
        config = self._update_config_with_params(params)
        
        # Create checkpoint directory for this trial
        checkpoint_dir = os.path.join(self.experiment_dir, trial_name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Get dataloaders
            dataloaders = get_all_dataloaders(config, self.verbose)
            
            # Get sample batch to determine input size
            sample_batch, _ = next(iter(dataloaders['train']))
            input_size = sample_batch.shape[2]  # (batch_size, seq_len, input_size)
            
            # Define model
            hidden_size = params.get('hidden_size', 64)
            # Ensure hidden_size is the correct type (int or list of ints)
            if not isinstance(hidden_size, (int, list)):
                hidden_size = int(hidden_size)
                
            num_layers = params.get('num_layers', 2)
            if not isinstance(num_layers, int):
                num_layers = int(num_layers)
                
            dropout = params.get('dropout', 0.2)
            if not isinstance(dropout, float):
                dropout = float(dropout)
                
            model = GRUNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_size=len(config.preprocess.target_idx),
                bidirectional=bool(params.get('bidirectional', False)),
                return_sequences=False
            )
            model.to(device)
            
            # Define loss function
            criterion = torch.nn.MSELoss()
            
            # Define optimizer
            learning_rate = params.get('learning_rate', 0.001)
            if not isinstance(learning_rate, float):
                learning_rate = float(learning_rate)
                
            weight_decay = params.get('weight_decay', 0.0)
            if not isinstance(weight_decay, float):
                weight_decay = float(weight_decay)
                
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Define learning rate scheduler
            lr_factor = params.get('lr_factor', 0.5)
            if not isinstance(lr_factor, float):
                lr_factor = float(lr_factor)
                
            lr_patience = params.get('lr_patience', 5)
            if not isinstance(lr_patience, int):
                lr_patience = int(lr_patience)
                
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=lr_factor, 
                patience=lr_patience
            )
            
            # Train model
            num_epochs = params.get('num_epochs', 100)
            if not isinstance(num_epochs, int):
                num_epochs = int(num_epochs)
                
            early_stopping_patience = params.get('early_stopping_patience', 15)
            if not isinstance(early_stopping_patience, int):
                early_stopping_patience = int(early_stopping_patience)

            verbose_modes = {'training': False, 'validation': False, 'testing': False}
                
            _, history, _ = train_model(
                model=model,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=num_epochs,
                device=device,
                checkpoint_dir=checkpoint_dir,
                patience=early_stopping_patience,
                verbose_modes=verbose_modes
            )
            
            # Get the best metric value
            if self.metric in history:
                metric_values = history[self.metric]
                best_idx = np.argmin(metric_values) if self.direction == 'minimize' else np.argmax(metric_values)
                best_value = metric_values[best_idx]
            else:
                # If the specific metric isn't in history, use the last validation loss
                best_value = history['val_loss'][-1]
                best_idx = len(history['val_loss']) - 1
            
            # Prepare results
            # Convert all numpy types and other non-serializable types to Python native types
            serializable_params = {}
            for k, v in params.items():
                if isinstance(v, np.number):
                    serializable_params[k] = v.item()
                elif isinstance(v, (np.ndarray, list)):
                    serializable_params[k] = [x.item() if isinstance(x, np.number) else x for x in v]
                elif isinstance(v, bool):
                    # Explicitly handle boolean values
                    serializable_params[k] = bool(v)
                elif isinstance(v, (int, float, str)):
                    # These types are already JSON serializable
                    serializable_params[k] = v
                else:
                    # Convert any other types to string representation
                    serializable_params[k] = str(v)
            
            # Create a JSON-serializable result dictionary
            try:
                result = {
                    'trial_id': trial_id,
                    'params': serializable_params,
                    'best_epoch': int(best_idx + 1),
                    'best_value': float(best_value),
                    'history': {k: [float(x) for x in v] for k, v in history.items()},
                    'status': 'completed',
                    'duration': float(time.time() - start_time)
                }
                
                # Test JSON serialization to catch any issues early
                json.dumps(result)
            except TypeError as e:
                logger.error(f"JSON serialization error: {str(e)}")
                # Create a simpler result with string representations where needed
                result = {
                    'trial_id': trial_id,
                    'params': {k: str(v) for k, v in params.items()},
                    'best_epoch': int(best_idx + 1),
                    'best_value': float(best_value),
                    'status': 'completed',
                    'duration': float(time.time() - start_time),
                    'serialization_error': str(e)
                }
            
            logger.info(f"Completed {trial_name} with best {self.metric}: {best_value:.6f} at epoch {best_idx+1}")
            
        except Exception as e:
            logger.error(f"Error in {trial_name}: {str(e)}")
            result = {
                'trial_id': trial_id,
                'params': params,
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - start_time
            }
        
        # Save individual trial result
        try:
            with open(os.path.join(self.experiment_dir, f"{trial_name}_result.json"), 'w') as f:
                json.dump(result, f, indent=4)
        except TypeError as e:
            logger.error(f"Error saving result to JSON: {str(e)}")
            # Fall back to saving a simplified version
            with open(os.path.join(self.experiment_dir, f"{trial_name}_result.txt"), 'w') as f:
                f.write(f"Trial ID: {trial_id}\n")
                f.write(f"Parameters: {str(params)}\n")
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                if 'best_value' in result:
                    f.write(f"Best value: {result['best_value']}\n")
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
        
        return result
    
    def _objective(self, trial) -> float:
        """Objective function for Bayesian optimization with Optuna."""
        # Convert param_grid to Optuna parameter suggestions
        params = {}
        for key, values in self.param_grid.items():
            if isinstance(values, list):
                # Categorical parameter
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(values, tuple) and len(values) == 3:
                if values[0] == 'log_uniform':
                    # Log-uniform distribution
                    params[key] = trial.suggest_float(key, values[1], values[2], log=True)
                elif values[0] == 'int_uniform':
                    # Integer uniform distribution
                    params[key] = trial.suggest_int(key, values[1], values[2])
                else:
                    # Uniform distribution
                    params[key] = trial.suggest_float(key, values[1], values[2])
            else:
                raise ValueError(f"Unsupported parameter space for {key}: {values}")
        
        # Evaluate the configuration
        result = self._evaluate_config(params, trial.number)
        
        # Store the result
        self.results.append(result)
        
        # Return the metric value
        if result['status'] == 'completed':
            return result['best_value']
        else:
            # Return a very bad value for failed trials
            return float('inf') if self.direction == 'minimize' else float('-inf')
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the hyperparameter search."""
        start_time = time.time()
        logger.info(f"Starting {self.search_method} search with {self.n_jobs} parallel jobs")
        
        if self.search_method == 'grid':
            # Get all parameter combinations for grid search
            param_combinations = self._get_all_combinations()
            logger.info(f"Grid search with {len(param_combinations)} combinations")
            
        elif self.search_method == 'random':
            # Generate random parameter combinations
            param_combinations = self._get_random_combinations(self.n_trials)
            logger.info(f"Random search with {len(param_combinations)} trials")
            
        elif self.search_method == 'bayesian':
            # Use Optuna for Bayesian optimization
            direction = 'minimize' if self.direction == 'minimize' else 'maximize'
            study = optuna.create_study(direction=direction)
            study.optimize(self._objective, n_trials=self.n_trials)
            
            # Get the best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Bayesian search completed with best {self.metric}: {best_value:.6f}")
            logger.info(f"Best parameters: {best_params}")
            
            # Save study results
            study_results = {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': self.n_trials,
                'trials': [{
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                    'state': t.state.name
                } for t in study.trials]
            }
            
            with open(os.path.join(self.experiment_dir, "study_results.json"), 'w') as f:
                json.dump(study_results, f, indent=4)
            
            # Return the results collected by the objective function
            return self.results
        
        # For grid and random search, run trials in parallel or sequentially
        if self.n_jobs > 1:
            # Parallel execution
            with mp.Pool(processes=min(self.n_jobs, len(param_combinations))) as pool:
                try:
                    results = pool.starmap(
                        self._evaluate_config,
                        [(params, i) for i, params in enumerate(param_combinations)]
                    )
                except Exception as e:
                    logger.error(f"Error in parallel execution: {str(e)}")
                    # Fall back to sequential execution
                    results = []
                    for i, params in enumerate(param_combinations):
                        try:
                            result = self._evaluate_config(params, i)
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error in trial {i}: {str(e)}")
        else:
            # Sequential execution
            results = []
            for i, params in enumerate(param_combinations):
                try:
                    result = self._evaluate_config(params, i)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in trial {i}: {str(e)}")
        
        # Store all results
        self.results = results
        
        # Find the best configuration
        completed_results = [r for r in results if r['status'] == 'completed']
        if completed_results:
            if self.direction == 'minimize':
                best_result = min(completed_results, key=lambda x: x['best_value'])
            else:
                best_result = max(completed_results, key=lambda x: x['best_value'])
            
            logger.info(f"Best configuration found with {self.metric}: {best_result['best_value']:.6f}")
            logger.info(f"Best parameters: {best_result['params']}")
        else:
            logger.warning("No successful trials completed")
        
        # Save all results
        try:
            with open(os.path.join(self.experiment_dir, "all_results.json"), 'w') as f:
                json.dump(results, f, indent=4)
        except TypeError as e:
            logger.error(f"Error saving all results to JSON: {str(e)}")
            # Save a simplified version
            with open(os.path.join(self.experiment_dir, "all_results_summary.txt"), 'w') as f:
                f.write(f"Total trials: {len(results)}\n")
                f.write(f"Completed trials: {len(completed_results)}\n")
                if completed_results:
                    if self.direction == 'minimize':
                        best_result = min(completed_results, key=lambda x: x['best_value'])
                    else:
                        best_result = max(completed_results, key=lambda x: x['best_value'])
                    f.write(f"Best {self.metric}: {best_result['best_value']}\n")
                    f.write(f"Best parameters: {str(best_result['params'])}\n")
        
        total_time = time.time() - start_time
        logger.info(f"Search completed in {total_time/60:.2f} minutes")
        
        return results


def main():
    """Example usage of the hyperparameter search."""
    # Define parameter grid
    param_grid = {
        'hidden_size': [32, 64, 128, 256, 512],
        'num_layers': [1, 2, 3, 4, 5],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
        'weight_decay': [0.0, 0.0001, 0.001, 0.00001],
        'bidirectional': [False, True],
        'num_epochs': [100],
        'early_stopping_patience': [5, 10, 15],
        'lr_factor': [0.5, 0.25, 0.1],
        'lr_patience': [5, 10]
    }
    
    # Create and run the search
    search = HyperparameterSearch(
        base_config_path="/home/kitne/University/2lvl/SU/bike-gru-experiments/config/default.yaml",
        param_grid=param_grid,
        search_method='random',  # 'grid', 'random', or 'bayesian'
        n_trials=25,            # Only used for random and bayesian search
        metric='val_rmse',       # Metric to optimize
        direction='minimize',    # 'minimize' or 'maximize'
        n_jobs=1,               # Number of parallel jobs
        experiment_name="random_search",
        verbose=False
    )
    
    # Run the search
    results = search.run()
    
    # Print the best result
    completed_results = [r for r in results if r['status'] == 'completed']
    if completed_results:
        if search.direction == 'minimize':
            best_result = min(completed_results, key=lambda x: x['best_value'])
        else:
            best_result = max(completed_results, key=lambda x: x['best_value'])
        
        print(f"\nBest configuration:")
        print(f"  {search.metric}: {best_result['best_value']:.6f}")
        print(f"  Parameters: {best_result['params']}")
    else:
        print("No successful trials completed")


if __name__ == "__main__":
    main()