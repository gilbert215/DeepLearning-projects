import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)
        
        # Convert to numpy array
        extended_symbols = np.array(extended_symbols)
        
        # Initialize skip_connect
        skip_connect = np.zeros(N, dtype=int)
        
        # Skip connection is allowed at position i if:
        # extended_symbols[i] != BLANK and extended_symbols[i] != extended_symbols[i-2]
        for i in range(2, N):
            if extended_symbols[i] != self.BLANK and extended_symbols[i] != extended_symbols[i-2]:
                skip_connect[i] = 1

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # Initialize at t=0
        alpha[0, 0] = logits[0, extended_symbols[0]]
        if S > 1:
            alpha[0, 1] = logits[0, extended_symbols[1]]

        # Forward pass for t > 0
        for t in range(1, T):
            for s in range(S):
                # Current symbol probability at time t
                curr_prob = logits[t, extended_symbols[s]]
                
                # Can come from same state at t-1
                alpha[t, s] = alpha[t-1, s]
                
                # Can come from previous state at t-1
                if s > 0:
                    alpha[t, s] += alpha[t-1, s-1]
                
                # Can skip from s-2 if skip connection exists
                if s > 1 and skip_connect[s]:
                    alpha[t, s] += alpha[t-1, s-2]
                
                # Multiply by current symbol probability
                alpha[t, s] *= curr_prob

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # Initialize at last time step
        beta[T-1, S-1] = 1
        if S > 1:
            beta[T-1, S-2] = 1

        # Backward pass
        for t in range(T-2, -1, -1):
            for s in range(S):
                # Can transition to same state at t+1
                beta[t, s] = beta[t+1, s] * logits[t+1, extended_symbols[s]]
                
                # Can transition to next state at t+1
                if s < S - 1:
                    beta[t, s] += beta[t+1, s+1] * logits[t+1, extended_symbols[s+1]]
                
                # Can skip to s+2 if skip connection exists
                if s < S - 2 and skip_connect[s+2]:
                    beta[t, s] += beta[t+1, s+2] * logits[t+1, extended_symbols[s+2]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # Compute unnormalized gamma: γ(t,r) = α(t,r) * β(t,r)
        gamma = alpha * beta
        
        # Compute normalization factor for each time step
        sumgamma = np.sum(gamma, axis=1, keepdims=True)
        
        # Normalize: γ(t,r) = α(t,r)β(t,r) / ∑_r' α(t,r')β(t,r')
        gamma = gamma / (sumgamma + 1e-10)

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """
        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # Truncate the target to target length
            target_seq = target[batch_itr, :target_lengths[batch_itr]]
            
            # Truncate the logits to input length
            logit = logits[:input_lengths[batch_itr], batch_itr, :]
            
            # Extend target sequence with blank
            extended, skip_connect = self.ctc.extend_target_with_blank(target_seq)
            self.extended_symbols.append(extended)
            
            # Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logit, extended, skip_connect)
            
            # Compute backward probabilities
            beta = self.ctc.get_backward_probs(logit, extended, skip_connect)
            
            # Compute posteriors
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            self.gammas.append(gamma)
            
            # Compute loss 
            for t in range(len(logit)):
                for s in range(len(extended)):
                    total_loss[batch_itr] += gamma[t][s] * np.log(logit[t][extended[s]] + 1e-10)
            
            total_loss[batch_itr] = -total_loss[batch_itr]

        total_loss = np.sum(total_loss) / B

        return total_loss

    def backward(self):
        """
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0.0)

        for batch_itr in range(B):
            # Truncate the target to target length
            target = self.target[batch_itr][:self.target_lengths[batch_itr]]
            
            # Truncate the logits to input length
            logit = self.logits[:self.input_lengths[batch_itr], batch_itr]
            
            # Extend target sequence with blank
            extended, _ = self.ctc.extend_target_with_blank(target)
            
            # Get the gamma for this batch
            gamma = self.gammas[batch_itr]
            
            # Compute derivative
            T_batch = self.input_lengths[batch_itr]
            
            for t in range(T_batch):
                for s in range(len(extended)):
                    symbol = extended[s]
                    # Gradient contribution from position s in extended sequence
                    dY[t, batch_itr, symbol] -= gamma[t, s] / (logit[t, symbol] + 1e-10)

        return dY