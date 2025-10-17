from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import lzma
import pickle
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import zipfile
from sklearn.preprocessing import StandardScaler

class DrivingNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        self.scaler = None  # Stockage du scaler

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, features, threshold=0.3):
        if isinstance(features, np.ndarray):
            x = features.copy()
        elif isinstance(features, torch.Tensor):
            x = features.numpy().copy()
        else:
            x = np.array(features, dtype=np.float32)
        
        # IMPORTANT: Normaliser les features avec le scaler
        if self.scaler is not None:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x = self.scaler.transform(x)
        
        x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            
            print("probabilities:", np.round(probs.numpy(), 3))
            print(probs)
            forward,back,left,right = True,True,True,True
            if features[0] < 20:
                threshold = threshold
            elif features[0] < 40:
                threshold = 0.6
            elif features[0] < 60:
                threshold = 0.8
                
            
            if features[0] > 30.0:
                forward = False
            
            if probs[0, 0] > threshold and probs[0, 1] > threshold:
                forward = probs[0, 0]> probs[0, 1]
                back = not forward
                
            if probs[0, 2] > threshold and probs[0, 3] > threshold:
                left = probs[0, 2] > probs[0, 3]
                right = not left

            return [
                ('forward', bool(probs[0, 0] > threshold and forward)),
                ('back', bool(probs[0, 1] > threshold) and back),
                ('left', bool(probs[0, 2] > threshold) and left),
                ('right', bool(probs[0, 3] > threshold) and right)
            ]
    
    def load_model(self, path):
        # Charger le modèle ET le scaler
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.eval()
        print(f"Model and scaler loaded from {path}")
    
    def summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if self.scaler is not None:
            print(f"\nScaler fitted: Yes")
            print(f"Feature means: {np.round(self.scaler.mean_, 3)}")
            print(f"Feature stds: {np.round(self.scaler.scale_, 3)}")

    def _create_dataset(self, record_files):
        class DrivingDataset(Dataset):
            def __init__(self, record_files, scaler=None, fit_scaler=True):
                self.data = []
                self.samples = []
                self.scaler = scaler

                for file in record_files:
                    if file.endswith(".zip"):
                        import zipfile
                        with zipfile.ZipFile(file, 'r') as zip_file:
                            file_list = zip_file.namelist()
                            if file_list:
                                with zip_file.open(file_list[0]) as f:
                                    records = pickle.load(f)
                                    self.data.extend(records)
                    
                    elif file.endswith(".npz"):
                        with lzma.open(file, "rb") as f:
                            data = pickle.load(f)
                            self.data.extend(data)

                print(f"  Total records loaded: {len(self.data)}")
                
                # Statistiques pour pondération
                action_counts = np.zeros(4)
                all_features = []
                
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
                    
                    action_counts += labels
                    all_features.append(features)
                    
                    self.samples.append({
                        'features': features,
                        'labels': labels,
                    })
                
                print(f"  Valid samples created: {len(self.samples)}")
                print(f"  Action distribution:")
                print(f"    Forward: {action_counts[0]:.0f} ({action_counts[0]/len(self.samples)*100:.1f}%)")
                print(f"    Back: {action_counts[1]:.0f} ({action_counts[1]/len(self.samples)*100:.1f}%)")
                print(f"    Left: {action_counts[2]:.0f} ({action_counts[2]/len(self.samples)*100:.1f}%)")
                print(f"    Right: {action_counts[3]:.0f} ({action_counts[3]/len(self.samples)*100:.1f}%)")
                
                # Normalisation avec StandardScaler
                if fit_scaler:
                    print("\n  Fitting StandardScaler on features...")
                    all_features_array = np.array(all_features)
                    self.scaler = StandardScaler()
                    self.scaler.fit(all_features_array)
                    print(f"  Feature means: {np.round(self.scaler.mean_, 3)}")
                    print(f"  Feature stds: {np.round(self.scaler.scale_, 3)}")
                
                # Appliquer la normalisation à tous les samples
                if self.scaler is not None:
                    print("  Applying normalization to all samples...")
                    for sample in self.samples:
                        sample['features'] = self.scaler.transform(
                            sample['features'].reshape(1, -1)
                        ).flatten().astype(np.float32)
                
                # Calcul des poids pour chaque classe
                total = len(self.samples)
                self.pos_weights = torch.tensor([
                    (total - action_counts[i]) / (action_counts[i] + 1e-6) 
                    for i in range(4)
                ], dtype=torch.float32)
                print(f"  Positive weights: {np.round(self.pos_weights.numpy(), 3)}")

            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                return (
                    torch.from_numpy(sample['features']).float(),
                    torch.from_numpy(sample['labels']).float()
                )
        
        return DrivingDataset(record_files, scaler=self.scaler, fit_scaler=(self.scaler is None))
    
    def loss_function(self, outputs, labels, pos_weight=None):
        if pos_weight is not None:
            criterion = nn.BCELoss(weight=pos_weight)
        else:
            criterion = nn.BCELoss()
        return criterion(outputs, labels)
    
    def train_model(self, record_files, epochs=200, batch_size=32, learning_rate=1e-4, 
                    train_split=0.8, use_pos_weight=True):
        dataset = self._create_dataset(record_files)
        
        # Sauvegarder le scaler dans le modèle
        self.scaler = dataset.scaler
        
        # Récupération des poids de classe (mais clippés pour éviter des valeurs extrêmes)
        if use_pos_weight:
            pos_weight = torch.clamp(dataset.pos_weights, min=0.5, max=3.0)
            print(f"\n  Using clipped pos_weights: {np.round(pos_weight.numpy(), 3)}")
        else:
            pos_weight = None
            print(f"\n  Not using pos_weights")
        
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42) 
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)

        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.train()
            total_train_loss = 0.0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(features)
                loss = self.loss_function(outputs, labels, pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation avec métriques
            self.eval()
            total_val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    outputs = self.forward(features)
                    loss = self.loss_function(outputs, labels, pos_weight)
                    total_val_loss += loss.item()
                    
                    # Calculer l'accuracy
                    probs = outputs
                    predictions = (probs > 0.5).float()
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.numel()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct_predictions / total_predictions
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Sauvegarde du meilleur modèle
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Plot avec 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o', markersize=3)
        ax1.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss over Epochs')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', marker='o', markersize=3, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy over Epochs')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curve.png')
        plt.show()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final validation accuracy: {val_accuracy:.4f}")
        
        return train_losses, val_losses