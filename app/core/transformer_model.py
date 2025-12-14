"""
Molecular Universe Generator (MUG) - Transformer Architecture
Author: Ali (Troxter222)
License: MIT

Transformer-based Variational Autoencoder for molecular generation.
Implements attention mechanisms with VAE latent space for controlled synthesis.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies as described in "Attention is All You Need".
    
    Args:
        d_model: Dimension of model embeddings
        max_len: Maximum sequence length to precompute
    """
    
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create wavelength scaling factors
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape for sequence-first format: [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
        
        Returns:
            Tensor with added positional information
        """
        return x + self.pe[:x.size(0), :]


class MoleculeTransformer(nn.Module):
    """
    Transformer-based Variational Autoencoder for molecular generation.
    
    Combines transformer architecture with VAE latent space for controllable
    molecular design. Uses pre-layer normalization for training stability.
    
    Architecture:
        - Encoder: Multi-head self-attention layers process input molecules
        - Latent space: VAE reparameterization trick for smooth generation
        - Decoder: Autoregressive generation with cross-attention to latent
    
    Args:
        vocab_size: Size of molecular vocabulary (SELFIES tokens)
        d_model: Dimension of model embeddings (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 4)
        num_decoder_layers: Number of decoder layers (default: 4)
        dim_feedforward: Dimension of feedforward networks (default: 1024)
        latent_size: Dimension of VAE latent space (default: 128)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        latent_size: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 2000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer encoder (processes input sequences)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True  # Pre-LN for stability (GPT-style)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Transformer decoder (generates output sequences)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # VAE latent space projections
        self.fc_mu = nn.Linear(d_model, latent_size)
        self.fc_logvar = nn.Linear(d_model, latent_size)
        
        # Latent to decoder initialization
        self.fc_latent_to_hidden = nn.Linear(latent_size, d_model)
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent distribution parameters.
        
        Args:
            src: Source sequence [seq_len, batch_size]
            src_mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Padding mask [batch_size, seq_len]
        
        Returns:
            Tuple of (mu, logvar) for VAE latent distribution
        """
        # Embed tokens and scale by sqrt(d_model) as in original transformer
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        
        # Encode sequence through transformer layers
        memory = self.transformer_encoder(
            src_emb, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Global pooling over sequence dimension
        if src_key_padding_mask is not None:
            # Mask out padding tokens before pooling
            # src_key_padding_mask: [batch, seq] -> [seq, batch, 1]
            mask = src_key_padding_mask.transpose(0, 1).unsqueeze(-1).float()
            memory_masked = memory * (1 - mask)
            
            # Compute mean only over valid (non-padding) positions
            sum_memory = torch.sum(memory_masked, dim=0)
            count = torch.sum(1 - mask, dim=0).clamp(min=1.0)
            pooled = sum_memory / count
        else:
            # Simple mean pooling if no padding mask provided
            pooled = torch.mean(memory, dim=0)
        
        # Project to latent distribution parameters
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        return mu, logvar
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        VAE reparameterization trick for backpropagation through sampling.
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_size]
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector z
        """
        if self.training:
            # Sample from N(mu, sigma^2)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean during evaluation
            return mu
    
    def decode(
        self,
        z: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent vector to output sequence.
        
        Args:
            z: Latent vector [batch_size, latent_size]
            tgt: Target sequence [seq_len, batch_size]
            tgt_mask: Causal attention mask
            tgt_key_padding_mask: Padding mask
        
        Returns:
            Output logits [seq_len, batch_size, vocab_size]
        """
        # Project latent to hidden dimension
        memory = self.fc_latent_to_hidden(z).unsqueeze(0)  # [1, batch, d_model]
        
        # Embed target tokens
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Decode with cross-attention to latent memory
        output = self.transformer_decoder(
            tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, sample latent, decode.
        
        Args:
            src: Source sequence [seq_len, batch_size]
            tgt: Target sequence [seq_len, batch_size]
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
            tgt_mask: Causal mask for autoregressive decoding
        
        Returns:
            Tuple of (logits, mu, logvar) for reconstruction and KL loss
        """
        # Encode to latent distribution
        mu, logvar = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode to output sequence
        logits = self.decode(z, tgt, tgt_mask, tgt_key_padding_mask)
        
        return logits, mu, logvar
    
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        vocab,
        max_len: int = 150,
        temperature: float = 1.0,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate molecular sequences from latent space.
        
        Args:
            n_samples: Number of molecules to generate
            device: Target device for tensors
            vocab: Vocabulary object with token mappings
            max_len: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            z: Optional latent vectors (if None, sample from prior)
        
        Returns:
            Generated sequences [n_samples, seq_len]
        """
        self.eval()
        
        with torch.no_grad():
            # Sample from prior if latent not provided
            if z is None:
                z = torch.randn(n_samples, self.latent_size, device=device)
            
            # Project latent to memory
            memory = self.fc_latent_to_hidden(z).unsqueeze(0)  # [1, n_samples, d_model]
            
            # Initialize with start-of-sequence token
            sos_idx = vocab.sos_idx
            current_tokens = torch.full(
                (1, n_samples), 
                sos_idx, 
                dtype=torch.long, 
                device=device
            )
            
            # Track which sequences have finished
            finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
            eos_idx = vocab.eos_idx
            
            # Autoregressive generation
            for step in range(max_len):
                # Embed current tokens
                tgt_emb = self.embedding(current_tokens) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)
                
                # Create causal mask (prevent attending to future)
                seq_len = current_tokens.size(0)
                causal_mask = self._generate_square_subsequent_mask(seq_len, device)
                
                # Decode next token
                output = self.transformer_decoder(
                    tgt_emb, 
                    memory, 
                    tgt_mask=causal_mask
                )
                
                # Get logits for last position
                logits = self.fc_out(output[-1, :, :])  # [n_samples, vocab_size]
                
                # Apply temperature and sample
                probs = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # [n_samples, 1]
                
                # Mark finished sequences (encountered EOS)
                eos_mask = (next_tokens.squeeze(-1) == eos_idx)
                finished = finished | eos_mask
                
                # Append next token
                current_tokens = torch.cat(
                    [current_tokens, next_tokens.transpose(0, 1)], 
                    dim=0
                )
                
                # Stop if all sequences finished
                if finished.all():
                    break
            
            # Remove SOS token and return [batch, seq_len]
            return current_tokens[1:].transpose(0, 1)
    
    @staticmethod
    def _generate_square_subsequent_mask(
        size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate causal mask for autoregressive decoding.
        
        Args:
            size: Sequence length
            device: Target device
        
        Returns:
            Upper triangular mask with -inf above diagonal
        """
        mask = torch.triu(
            torch.ones(size, size, device=device) * float('-inf'), 
            diagonal=1
        )
        return mask
    
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        vocab,
        device: torch.device,
        steps: int = 10,
        max_len: int = 150
    ) -> torch.Tensor:
        """
        Interpolate between two latent vectors to generate intermediate molecules.
        
        Args:
            z1: First latent vector [1, latent_size]
            z2: Second latent vector [1, latent_size]
            vocab: Vocabulary object
            device: Target device
            steps: Number of interpolation steps
            max_len: Maximum sequence length
        
        Returns:
            Interpolated sequences [steps, seq_len]
        """
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
        z_interp = z1 * (1 - alphas) + z2 * alphas
        
        # Generate from interpolated latents
        return self.sample(
            n_samples=steps,
            device=device,
            vocab=vocab,
            max_len=max_len,
            z=z_interp
        )
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            Dictionary with model specifications
        """
        return {
            'architecture': 'Transformer VAE',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'latent_size': self.latent_size,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def compute_vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    ignore_index: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss (reconstruction + KL divergence).
    
    Args:
        logits: Model predictions [seq_len, batch, vocab_size]
        targets: Target sequences [seq_len, batch]
        mu: Latent mean [batch, latent_size]
        logvar: Latent log variance [batch, latent_size]
        kl_weight: Weight for KL term (beta-VAE)
        ignore_index: Padding token index to ignore in loss
    
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss)
    """
    # Reconstruction loss (cross-entropy)
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss