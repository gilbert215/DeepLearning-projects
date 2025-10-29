import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # Get sequence length
        seq_length = y_probs.shape[1]
        
        # 1. Iterate over sequence length
        for t in range(seq_length):
            # Get probabilities at time t (for batch 0)
            probs_t = y_probs[:, t, 0]
            
            # 2. Find symbol with max probability
            max_idx = np.argmax(probs_t)
            max_prob = probs_t[max_idx]
            
            # 3. Update path probability
            path_prob *= max_prob
            
            # 4. Append symbol to decoded path
            decoded_path.append(max_idx)
        
        # 5. Compress sequence: remove blanks and repeated symbols
        compressed_path = []
        for i, symbol in enumerate(decoded_path):
            # Skip blanks
            if symbol == blank:
                continue
            # Skip repeated symbols (consecutive duplicates)
            if i > 0 and symbol == decoded_path[i-1]:
                continue
            # Add symbol to compressed path
            compressed_path.append(symbol)
        
        # Convert symbol indices to symbol strings
        # symbol_set is 0-indexed, but our symbols start from 1 (0 is blank)
        decoded_string = ''.join([self.symbol_set[s-1] for s in compressed_path])
        
        return decoded_string, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        blank = 0
        
        # Initialize: paths are stored as (decoded_string, last_symbol) -> score
        # last_symbol tracks whether we ended with blank ('') or a symbol
        paths = {('', ''): 1.0}  # (decoded_path, last_symbol) -> probability
        
        # Iterate through time steps
        for t in range(T):
            new_paths = {}
            
            # Get probabilities at time t (batch 0)
            probs_t = y_probs[:, t, 0]
            
            # Sort current paths by score and keep top beam_width
            sorted_paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
            sorted_paths = sorted_paths[:self.beam_width]
            
            # Extend each path with each symbol
            for (decoded, last_sym), score in sorted_paths:
                for symbol_idx in range(len(probs_t)):
                    prob = probs_t[symbol_idx]
                    new_score = score * prob
                    
                    if symbol_idx == blank:
                        # Adding blank: decoded stays same, last_sym becomes blank
                        new_path = (decoded, '')
                    else:
                        # Get the symbol character
                        symbol = self.symbol_set[symbol_idx - 1]
                        
                        if last_sym == symbol:
                            # Same symbol as last: don't extend (CTC collapse)
                            new_path = (decoded, symbol)
                        else:
                            # Different symbol: extend the decoded path
                            new_path = (decoded + symbol, symbol)
                    
                    # Accumulate scores for same path
                    if new_path in new_paths:
                        new_paths[new_path] += new_score
                    else:
                        new_paths[new_path] = new_score
            
            paths = new_paths
        
        # Merge paths with same decoded string
        merged_path_scores = {}
        for (decoded, last_sym), score in paths.items():
            if decoded in merged_path_scores:
                merged_path_scores[decoded] += score
            else:
                merged_path_scores[decoded] = score
        
        # Find best path
        bestPath = max(merged_path_scores.items(), key=lambda x: x[1])[0]
        
        return bestPath, merged_path_scores