"""
Molecular Universe Generator (MUG) - Vocabulary Module
Author: Ali (Troxter222)
License: MIT

Vocabulary management for transformer-based molecular generation.
Handles SELFIES tokenization, encoding/decoding, and attention masking.
"""

from typing import List, Optional, Union

import selfies as sf
import torch


class Vocabulary:
    """
    Vocabulary handler for transformer-based molecular generation.
    
    Manages token-to-index mappings, SELFIES encoding/decoding,
    and attention mask generation for transformer models.
    
    Attributes:
        vocab (List[str]): Ordered list of vocabulary tokens
        char2idx (dict): Token to index mapping
        idx2char (dict): Index to token mapping
        pad_idx (int): Padding token index
        sos_idx (int): Start-of-sequence token index
        eos_idx (int): End-of-sequence token index
    """
    
    # Special tokens that must be present in vocabulary
    REQUIRED_TOKENS = ['<pad>', '<sos>', '<eos>']
    
    def __init__(self, vocab_list: List[str]):
        """
        Initialize vocabulary from token list.
        
        Args:
            vocab_list: Ordered list of vocabulary tokens.
                       Must include special tokens: <pad>, <sos>, <eos>
        
        Raises:
            ValueError: If required special tokens are missing
        """
        self.vocab = vocab_list
        self.char2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2char = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Validate and assign special token indices
        self._validate_special_tokens()
        
        self.pad_idx = self.char2idx['<pad>']
        self.sos_idx = self.char2idx['<sos>']
        self.eos_idx = self.char2idx['<eos>']
    
    def _validate_special_tokens(self):
        """Validate presence of required special tokens."""
        missing_tokens = [
            token for token in self.REQUIRED_TOKENS 
            if token not in self.char2idx
        ]
        
        if missing_tokens:
            raise ValueError(
                f"Vocabulary missing required tokens: {missing_tokens}. "
                f"Ensure vocabulary contains: {self.REQUIRED_TOKENS}"
            )
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return (
            f"Vocabulary(size={len(self)}, "
            f"pad={self.pad_idx}, sos={self.sos_idx}, eos={self.eos_idx})"
        )
    
    def encode(self, selfies_str: str, max_len: int = 200) -> List[int]:
        """
        Encode SELFIES string to token indices.
        
        Tokenizes SELFIES string, adds SOS/EOS tokens, and pads to max_len.
        
        Args:
            selfies_str: SELFIES representation of molecule
            max_len: Maximum sequence length (default: 200)
        
        Returns:
            List of token indices with SOS/EOS and padding
            
        Example:
            >>> vocab.encode("[C][C][O]", max_len=10)
            [1, 4, 4, 5, 2, 0, 0, 0, 0, 0]  # [sos, C, C, O, eos, pad, ...]
        """
        try:
            tokens = list(sf.split_selfies(selfies_str))
        except Exception:
            # Return minimal valid sequence if parsing fails
            tokens = []
        
        # Build sequence: [SOS] + tokens + [EOS]
        indices = [self.sos_idx]
        indices.extend(self.char2idx.get(token, self.pad_idx) for token in tokens)
        indices.append(self.eos_idx)
        
        # Pad or truncate to max_len
        if len(indices) < max_len:
            indices.extend([self.pad_idx] * (max_len - len(indices)))
        
        return indices[:max_len]
    
    def decode(self, indices: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token indices to SELFIES string.
        
        Converts sequence of indices back to SELFIES representation,
        stopping at EOS token and excluding special tokens.
        
        Args:
            indices: List or tensor of token indices
        
        Returns:
            SELFIES string representation
            
        Example:
            >>> vocab.decode([1, 4, 4, 5, 2, 0])
            "[C][C][O]"
        """
        tokens = []
        
        for idx in indices:
            # Convert tensor to int if needed
            if torch.is_tensor(idx):
                idx = idx.item()
            
            # Stop at end-of-sequence
            if idx == self.eos_idx:
                break
            
            # Skip special tokens
            if idx in (self.pad_idx, self.sos_idx):
                continue
            
            # Add valid token
            if idx in self.idx2char:
                tokens.append(self.idx2char[idx])
        
        return "".join(tokens)
    
    def batch_encode(
        self, 
        selfies_list: List[str], 
        max_len: int = 200
    ) -> torch.Tensor:
        """
        Encode batch of SELFIES strings to tensor.
        
        Args:
            selfies_list: List of SELFIES strings
            max_len: Maximum sequence length
        
        Returns:
            Tensor of shape [batch_size, max_len]
        """
        encoded = [self.encode(selfies, max_len) for selfies in selfies_list]
        return torch.tensor(encoded, dtype=torch.long)
    
    def batch_decode(self, indices_batch: torch.Tensor) -> List[str]:
        """
        Decode batch of token indices to SELFIES strings.
        
        Args:
            indices_batch: Tensor of shape [batch_size, seq_len]
        
        Returns:
            List of decoded SELFIES strings
        """
        return [self.decode(sequence) for sequence in indices_batch]
    
    # ========================
    # Attention Mask Generation
    # ========================
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Generate source (encoder) attention mask.
        
        Creates mask to prevent attention on padding tokens.
        
        Args:
            src: Source tensor of shape [batch_size, src_len]
        
        Returns:
            Boolean mask of shape [batch_size, 1, 1, src_len]
            True indicates positions that should be attended to.
            
        Example:
            >>> src = torch.tensor([[1, 4, 5, 2, 0, 0]])  # [sos, C, O, eos, pad, pad]
            >>> mask = vocab.make_src_mask(src)
            >>> mask.shape
            torch.Size([1, 1, 1, 6])
        """
        # Create mask: True where not padding, False where padding
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Generate target (decoder) attention mask.
        
        Creates combined mask that:
        1. Prevents attention on padding tokens
        2. Prevents attention on future tokens (causal mask)
        
        Args:
            trg: Target tensor of shape [batch_size, trg_len]
        
        Returns:
            Boolean mask of shape [batch_size, 1, trg_len, trg_len]
            True indicates positions that should be attended to.
            
        Example:
            >>> trg = torch.tensor([[1, 4, 5, 2]])  # [sos, C, O, eos]
            >>> mask = vocab.make_trg_mask(trg)
            >>> mask[0, 0]  # Causal structure visible
            tensor([[ True, False, False, False],
                    [ True,  True, False, False],
                    [ True,  True,  True, False],
                    [ True,  True,  True,  True]])
        """
        batch_size, trg_len = trg.shape
        device = trg.device
        
        # 1. Padding mask: hide padding tokens
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. Causal mask: hide future tokens (lower triangular matrix)
        trg_causal_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=device, dtype=torch.bool)
        )
        
        # Combine both masks with logical AND
        trg_mask = trg_pad_mask & trg_causal_mask
        
        return trg_mask
    
    def make_no_peak_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask without padding consideration.
        
        Useful for generating sequences where padding is not present.
        
        Args:
            size: Sequence length
            device: Target device for tensor
        
        Returns:
            Boolean mask of shape [1, size, size]
        """
        mask = torch.tril(torch.ones((size, size), device=device, dtype=torch.bool))
        return mask.unsqueeze(0)
    
    def get_token(self, idx: int) -> Optional[str]:
        """
        Get token by index.
        
        Args:
            idx: Token index
        
        Returns:
            Token string or None if index invalid
        """
        return self.idx2char.get(idx)
    
    def get_index(self, token: str) -> Optional[int]:
        """
        Get index by token.
        
        Args:
            token: Token string
        
        Returns:
            Token index or None if token not in vocabulary
        """
        return self.char2idx.get(token)
    
    def is_special_token(self, idx: int) -> bool:
        """
        Check if index corresponds to special token.
        
        Args:
            idx: Token index
        
        Returns:
            True if index is a special token
        """
        return idx in (self.pad_idx, self.sos_idx, self.eos_idx)
    
    def get_vocabulary_stats(self) -> dict:
        """
        Get vocabulary statistics.
        
        Returns:
            Dictionary containing vocabulary metrics
        """
        return {
            'total_tokens': len(self),
            'special_tokens': len(self.REQUIRED_TOKENS),
            'regular_tokens': len(self) - len(self.REQUIRED_TOKENS),
            'pad_idx': self.pad_idx,
            'sos_idx': self.sos_idx,
            'eos_idx': self.eos_idx
        }


# Alias for backward compatibility
TransformerVocabulary = Vocabulary