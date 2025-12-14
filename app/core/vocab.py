"""
Simplified Vocabulary Module for Molecular Tokenization
"""

from typing import List, Union

import selfies as sf
import torch


class Vocabulary:
    """Lightweight vocabulary handler for SELFIES tokenization."""
    
    def __init__(self, vocab_list: List[str]):
        """
        Initialize vocabulary from token list.
        
        Args:
            vocab_list: List of vocabulary tokens including special tokens
        """
        self.vocab = vocab_list
        self.char2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2char = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Cache special token indices
        self.pad_idx = self.char2idx.get('<pad>', 0)
        self.sos_idx = self.char2idx.get('<sos>', 1)
        self.eos_idx = self.char2idx.get('<eos>', 2)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def encode(self, selfies_str: str, max_len: int = 100) -> List[int]:
        """
        Encode SELFIES string to indices.
        
        Args:
            selfies_str: SELFIES molecular representation
            max_len: Maximum sequence length
            
        Returns:
            List of token indices with padding
        """
        try:
            tokens = list(sf.split_selfies(selfies_str))
        except Exception:
            tokens = []
        
        # Build sequence: [SOS] + tokens + [EOS]
        indices = [self.sos_idx]
        indices.extend(self.char2idx.get(token, self.pad_idx) for token in tokens)
        indices.append(self.eos_idx)
        
        # Pad to max_len
        if len(indices) < max_len:
            indices.extend([self.pad_idx] * (max_len - len(indices)))
        
        return indices[:max_len]
    
    def decode(self, indices: Union[List[int], torch.Tensor]) -> str:
        """
        Decode indices back to SELFIES string.
        
        Args:
            indices: Token indices (list or tensor)
            
        Returns:
            SELFIES string representation
        """
        tokens = []
        
        for i in indices:
            # Convert tensor to int if needed
            idx = i.item() if torch.is_tensor(i) else i
            
            # Skip invalid indices
            if idx not in self.idx2char:
                continue
            
            token = self.idx2char[idx]
            
            # Stop at EOS
            if token == '<eos>':
                break
            
            # Skip special tokens
            if token not in ('<pad>', '<sos>'):
                tokens.append(token)
        
        return "".join(tokens)