"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import os

# --- 1. ARCHITECTURE ---
class GNNTox(torch.nn.Module):
    def __init__(self):
        super(GNNTox, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))

# --- 2. PREDICTOR CLASS ---
class ToxPredictor:
    def __init__(self, model_path='checkpoints/gnn_tox_v1.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = GNNTox().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # nosec
            self.model.eval()
            print("GNN (Tox21) loaded successfully.")
        else:
            print(f"File {model_path} not found. GNN will not work.")
            self.model = None

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([atom.GetAtomicNum()])

        x = torch.tensor(atom_features, dtype=torch.float)
        
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
        if not edge_indices:
            return None
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        batch = torch.zeros(len(atom_features), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, batch=batch)

    def predict(self, smiles):
        if self.model is None:
            return 0.0
        
        try:
            data = self.smiles_to_graph(smiles)
            if not data:
                return 0.0
            
            data = data.to(self.device)
            
            with torch.no_grad():
                prob = self.model(data)
                
            return prob.item()
        except Exception:
            return 0.0