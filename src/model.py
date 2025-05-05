import torch
import torch.nn as nn
from typing import List, Union, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model')


class GRUNetwork(nn.Module):
    """
    Flexible GRU-based neural network for time series forecasting.
    
    Supports:
    - Single or multiple stacked GRU layers
    - Configurable hidden sizes for each layer
    - Dropout between layers
    - Output for the entire sequence or just the last timestep
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: Union[int, List[int]],
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 output_size: int = 1,
                 bidirectional: bool = False,
                 return_sequences: bool = False):
        """
        Initialize the GRU network.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state (int) or list of sizes for each layer
            num_layers: Number of stacked GRU layers
            dropout: Dropout probability between layers (0 = no dropout)
            output_size: Number of output features/targets
            bidirectional: Whether to use bidirectional GRU
            return_sequences: If True, return output for all timesteps, otherwise just the last
        """
        super(GRUNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        
        # Handle different hidden size configurations
        if isinstance(hidden_size, int):
            self.hidden_sizes = [hidden_size] * num_layers
        elif isinstance(hidden_size, list):
            if len(hidden_size) != num_layers:
                raise ValueError(f"Length of hidden_size list ({len(hidden_size)}) "
                               f"must match num_layers ({num_layers})")
            self.hidden_sizes = hidden_size
        else:
            raise TypeError("hidden_size must be an integer or a list of integers")
        
        # Create GRU layers
        self.gru_layers = nn.ModuleList()
        
        # First GRU layer (input_size -> hidden_sizes[0])
        self.gru_layers.append(
            nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_sizes[0],
                batch_first=True,
                bidirectional=bidirectional
            )
        )
        
        # Additional GRU layers if num_layers > 1
        for i in range(1, num_layers):
            # Account for bidirectional output from previous layer
            prev_layer_output_size = self.hidden_sizes[i-1] * (2 if bidirectional else 1)
            
            self.gru_layers.append(
                nn.GRU(
                    input_size=prev_layer_output_size,
                    hidden_size=self.hidden_sizes[i],
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output layer
        final_hidden_size = self.hidden_sizes[-1] * (2 if bidirectional else 1)
        
        # For more complex mappings, use two linear layers
        if final_hidden_size > 4 * output_size:
            intermediate_size = (final_hidden_size + output_size) // 2
            self.output_layers = nn.Sequential(
                nn.Linear(final_hidden_size, intermediate_size),
                nn.ReLU(),
                nn.Linear(intermediate_size, output_size)
            )
        else:
            # Simple mapping with a single linear layer
            self.output_layers = nn.Linear(final_hidden_size, output_size)
        
        logger.info(f"Initialized GRUNetwork with {num_layers} layers, "
                   f"hidden sizes {self.hidden_sizes}, "
                   f"input size {input_size}, output size {output_size}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, 
                initial_hidden: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            initial_hidden: Optional initial hidden state
            
        Returns:
            If return_sequences is True:
                Output tensor of shape (batch_size, seq_len, output_size)
            Else:
                Output tensor of shape (batch_size, output_size)
                
            If return_hidden is True, also returns the last hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through GRU layers
        hidden_states = []
        current_input = x
        
        for i, gru_layer in enumerate(self.gru_layers):
            # Get initial hidden state for this layer if provided
            layer_init_hidden = None
            if initial_hidden is not None:
                layer_init_hidden = initial_hidden[i].unsqueeze(0)
            
            # Forward pass through GRU layer
            output, hidden = gru_layer(current_input, layer_init_hidden)
            hidden_states.append(hidden)
            
            # Apply dropout between layers (except after the last GRU layer)
            if self.dropout_layer is not None and i < len(self.gru_layers) - 1:
                output = self.dropout_layer(output)
            
            # Update input for next layer
            current_input = output
        
        # Process the output through the linear layers
        if self.return_sequences:
            # Apply output layers to each timestep
            # Reshape to (batch_size * seq_len, hidden_size)
            reshaped_output = output.contiguous().view(-1, output.size(-1))
            # Apply output layers
            reshaped_output = self.output_layers(reshaped_output)
            # Reshape back to (batch_size, seq_len, output_size)
            final_output = reshaped_output.view(batch_size, seq_len, self.output_size)
        else:
            # Use only the last timestep
            last_output = output[:, -1, :]
            final_output = self.output_layers(last_output)
        
        # Stack hidden states from all layers
        # For layers with different hidden sizes, we can't concatenate directly
        # Instead, we'll return a list of hidden states or just the last layer's hidden state
        if len(set(self.hidden_sizes)) == 1 or self.bidirectional:
            # All layers have the same hidden size or we're using bidirectional GRU
            # We can safely concatenate along dim=0
            final_hidden = torch.cat([h for h in hidden_states], dim=0)
        else:
            # Different hidden sizes, return the last layer's hidden state
            final_hidden = hidden_states[-1]
        
        return final_output, final_hidden
    
    def get_last_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the last hidden state from the model."""
        _, hidden = self.forward(x)
        return hidden