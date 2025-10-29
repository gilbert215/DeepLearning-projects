import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from nn.linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Uncommented as instructed
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.
        -----
        Input
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]
        linear_weights:
                        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.
        -----
        Input
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified
        -------
        Returns
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        
        # Iterate through the sequence (time steps)
        for t in range(seq_len):
            # Iterate over the layers
            for l in range(self.num_layers):
                # Get input for this layer
                # Layer 0: input from x at time t
                # Other layers: input from previous layer's hidden state at time t
                if l == 0:
                    input_t = x[:, t, :]  
                else:
                    input_t = hidden[l-1]  
                
                # Get previous hidden state for this layer at time t-1
                h_prev_t = hidden[l] 
                
                # Run RNN cell forward
                hidden[l] = self.rnn[l].forward(input_t, h_prev_t)
            
            # Append a copy of the current hidden states to hiddens list
            self.hiddens.append(hidden.copy())
        
        # Get the outputs from the last time step using the linear layer
        # Use the last layer's hidden state at the last time step
        logits = self.output_layer.forward(hidden[-1])  # (batch_size, output_size)
        
        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Input
        ------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        # Iterate in reverse order of time 
        for t in range(seq_len - 1, -1, -1):
            # Iterate in reverse order of layers (from num_layers-1 to 0)
            for l in range(self.num_layers - 1, -1, -1):
                # Get h_prev_l either from hiddens or x depending on the layer
                # hiddens[t+1] contains hidden states after processing time step t
                # hiddens[t] contains hidden states before processing time step t
                if l == 0:
                    h_prev_l = self.x[:, t, :] 
                else:
                    # Hidden state from layer l-1 at time t (after processing)
                    h_prev_l = self.hiddens[t+1][l-1]
                
                # hidden state at previous time step for current layer
                h_prev_t = self.hiddens[t][l] 
                
                # Get current hidden state
                h_t = self.hiddens[t+1][l] 
                
                # Call backward on the RNN cell
                dx, dh_prev_t = self.rnn[l].backward(dh[l], h_t, h_prev_l, h_prev_t)
                
                # Update dh with gradient from previous time step
                dh[l] = dh_prev_t
                
                # If not at the first layer, add dx to the gradient from l-1th layer
                if l > 0:
                    dh[l-1] += dx
        
        # Normalize dh by batch_size since initial hidden states are treated as parameters
        return dh / batch_size