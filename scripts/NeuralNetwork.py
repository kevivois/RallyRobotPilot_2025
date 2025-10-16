from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import lzma
import pickle
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import zipfile  
class DrivingNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128), 
            nn.ReLU(),      
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, features, threshold=0.15):
        if isinstance(features, np.ndarray):
            x = torch.from_numpy(features).float()
        elif isinstance(features, torch.Tensor):
            x = features.float()
        else:
            x = torch.tensor(features).float()
            
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)

            print("probabilities:", probs)

            
            
            return [
                ('forward', bool(probs[0, 0] > threshold)),
                ('back', bool(probs[0, 1] > threshold)),
                ('left', bool(probs[0, 2] > threshold)),
                ('right', bool(probs[0, 3] > threshold))
            ]
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Model loaded from {path}")
    
    def summary(self):
        print(self)

    def _create_dataset(self, record_files):
        class DrivingDataset(Dataset):
            def __init__(self, record_files):
                self.data = []
                self.samples = []

                for file in record_files:
                    if file.endswith(".zip"):
                        import zipfile
                        with zipfile.ZipFile(file, 'r') as zip_file:
                            # Lire le premier fichier dans le zip
                            file_list = zip_file.namelist()
                            if file_list:
                                with zip_file.open(file_list[0]) as f:
                                    records = pickle.load(f)
                                    self.data.extend(records)
                    
                    # ✅ Cas 3: Fichiers .npz (numpy compressé)
                    elif file.endswith(".npz"):
                        with lzma.open(file, "rb") as f:
                            data = pickle.load(f)
                            self.data.extend(data)

                print(f"  Total records loaded: {len(self.data)}")
                
                for i in range(1, len(self.data)):
                    current = self.data[i]
                    
                    features = np.array([

                        float(current.car_speed),
                        *current.raycast_distances
                    ], dtype=np.float32)
                    
                    if np.isnan(features).any() or np.isinf(features).any():
                        continue
                    
                    labels = np.array([
                        float(current.current_controls[0]),
                        float(current.current_controls[1]),
                        float(current.current_controls[2]),
                        float(current.current_controls[3])
                    ], dtype=np.float32)
                    
                    if not all(l in [0.0, 1.0] for l in labels):
                        continue
                    
                    self.samples.append({
                        'features': features,
                        'labels': labels,
                    })
                
                print(f"  Valid samples created: {len(self.samples)}")

            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                return (
                    torch.from_numpy(sample['features']).float(),
                    torch.from_numpy(sample['labels']).float()
                )
        
        return DrivingDataset(record_files)
    
    def loss_function(self, outputs, labels):
        bce_loss = nn.BCEWithLogitsLoss()
        return bce_loss(outputs, labels)
    
    def train_model(self, record_files, epochs=20, batch_size=20, learning_rate=0.0005, 
                    train_split=0.8):
        dataset = self._create_dataset(record_files)
        
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproductibilité
    )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.train()
            total_train_loss = 0.0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(features)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    outputs = self.forward(features)
                    loss = self.loss_function(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png')
        plt.show()
        
        return train_losses, val_losses