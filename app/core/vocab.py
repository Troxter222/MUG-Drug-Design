"""
Simplified Vocabulary Module for Molecular Tokenization
"""

from typing import List, Union

import selfies as sf
import torch


class Vocabulary:
    """Lightweight vocabulary handler for SELFIES tokenization."""
    
    def __init__(self, vocab_list: List[str]):
        self.vocab = vocab_list
        self.char2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2char = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Cache special token indices
        self.pad_idx = self.char2idx.get('<pad>', 0)
        self.sos_idx = self.char2idx.get('<sos>', 1)
        self.eos_idx = self.char2idx.get('<eos>', 2)
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def encode(self, selfies_str: str, max_len: int = 100) -> List[int]:
        """
        Encode SELFIES string to indices.
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
    
    def batch_encode(self, selfies_list: List[str], max_len: int = 100) -> torch.Tensor:
        """
        Encodes a list of SELFIES strings into a tensor.
        Required for unit testing and batch processing.
        """
        encoded_data = [self.encode(s, max_len) for s in selfies_list]
        return torch.tensor(encoded_data, dtype=torch.long)
    
    def decode(self, indices: Union[List[int], torch.Tensor]) -> str:
        """
        Decode indices back to SELFIES string.
        """
        tokens = []
        
        # Handle single list or tensor row
        iterable = indices.tolist() if torch.is_tensor(indices) else indices

        for idx in iterable:
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