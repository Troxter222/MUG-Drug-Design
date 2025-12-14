import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import os

# --- 1. АРХИТЕКТУРА ---
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

# --- 2. КЛАСС ПРЕДСКАЗАТЕЛЯ ---
class ToxPredictor:
    def __init__(self, model_path='checkpoints/gnn_tox_v1.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация модели
        self.model = GNNTox().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Режим предсказания
            print("✅ GNN (Tox21) успешно загружена.")
        else:
            print(f"⚠️ Файл {model_path} не найден. GNN работать не будет.")
            self.model = None

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        
        # Атомы
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([atom.GetAtomicNum()])
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Связи
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
        if not edge_indices: return None
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Создаем батч из 1 элемента (так требует PyG)
        batch = torch.zeros(len(atom_features), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, batch=batch)

    def predict(self, smiles):
        if self.model is None: return 0.0
        
        try:
            data = self.smiles_to_graph(smiles)
            if not data: return 0.0
            
            data = data.to(self.device)
            
            with torch.no_grad():
                prob = self.model(data)
                
            return prob.item() # Возвращаем число от 0.0 до 1.0
        except:
            return 0.0