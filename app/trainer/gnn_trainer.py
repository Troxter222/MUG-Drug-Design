import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import os

# --- НАСТРОЙКИ ---
CSV_FILE = 'data/gnn_datasets/toxicity_data.csv' # Файл который мы скачали
MODEL_SAVE_PATH = 'checkpoints/gnn_tox_v1.pth'
EPOCHS = 20
BATCH_SIZE = 64

# --- 1. КОНВЕРТЕР SMILES -> GRAPH ---
def molecule_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # Узлы (Атомы)
    # Кодируем атом одним числом (Атомный номер)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Ребра (Связи)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i]) # Граф ненаправленный
    
    if not edge_indices: return None
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    y = torch.tensor([label], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

# --- 2. ЗАГРУЗКА ДАННЫХ ---
def load_dataset():
    print("⏳ Обработка графов...")
    df = pd.read_csv(CSV_FILE)
    # Берем колонки: SMILES и любой столбец токсичности (например, NR-AR)
    # В Tox21 много колонок, возьмем первую попавшуюся задачу для теста
    target_col = df.columns[1] 
    
    graphs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row['smiles'] # Проверь название колонки в CSV! (может быть 'SMILES' или 'smiles')
        label = row[target_col]
        
        # Пропускаем пустые метки
        if pd.isna(label): continue
        
        graph = molecule_to_graph(smi, label)
        if graph: graphs.append(graph)
        
    print(f"✅ Готово графов: {len(graphs)}")
    return graphs

# --- 3. МОДЕЛЬ GNN (GCN) ---
class GNNTox(torch.nn.Module):
    def __init__(self):
        super(GNNTox, self).__init__()
        # Вход: 1 фича (номер атома) -> Скрытый слой
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 1) # Выход: 0 или 1 (Токсичность)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # Global Pooling (собираем атомы в молекулу)
        x = global_mean_pool(x, batch)
        
        return torch.sigmoid(self.fc(x))

# --- 4. ОБУЧЕНИЕ ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = GNNTox().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss() # Binary Cross Entropy
    
    print(f"🚀 Старт обучения GNN на {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Ep {epoch+1} | Loss: {total_loss / len(loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("💾 GNN Модель сохранена!")

if __name__ == "__main__":
    # Создаем папку если нет
    os.makedirs('checkpoints', exist_ok=True)
    train()