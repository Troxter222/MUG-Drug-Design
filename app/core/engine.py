import torch
import torch.nn as nn

class MolecularVAE(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, latent_size: int, num_layers: int):
        super(MolecularVAE, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder
        self.encoder_gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_size * num_layers, latent_size)
        self.fc_logvar = nn.Linear(hidden_size * num_layers, latent_size)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_size, hidden_size * num_layers)
        self.decoder_gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def encode(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder_gru(embedded)
        hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def sample(self, n_samples, device, vocab, max_len=200, temp=1.0):
        z = torch.randn(n_samples, self.latent_size).to(device)
        
        hidden = self.decoder_input(z)
        hidden = hidden.view(n_samples, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        
        input_idx = torch.tensor([[vocab.char2idx['<sos>']]] * n_samples).to(device)
        decoded_indices = []
        
        for _ in range(max_len):
            embedded = self.embedding(input_idx)
            output, hidden = self.decoder_gru(embedded, hidden)
            logits = self.fc_out(output)
            
            probs = torch.nn.functional.softmax(logits.squeeze(1) / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            decoded_indices.append(next_token)
            input_idx = next_token
            
        return torch.cat(decoded_indices, dim=1)
