from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import lzma
import pickle
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import gc

class DrivingNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 32),       # ✅ Nouvelle couche
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, features, threshold=0.75):
        """Predict car controls from sensor features"""
        # Convert input to tensor
        if isinstance(features, np.ndarray):
            x = torch.from_numpy(features).float()
        elif isinstance(features, list):
            x = torch.tensor(features, dtype=torch.float32)
        elif isinstance(features, torch.Tensor):
            x = features.float()
        else:
            raise TypeError(f"Expected np.ndarray, list, or torch.Tensor, got {type(features)}")
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Move to device and detach
        x = x.to(self.device).detach()
        
        # Inference
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu()
            
            # Extract values
            forward_val = float(probs[0, 0].item())
            backward_val = float(probs[0, 1].item())
            left_val = float(probs[0, 2].item())
            right_val = float(probs[0, 3].item())
        
        # Cleanup
        del x, logits, probs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return as list of tuples: (direction, bool)
        print("probabilities:", {
            'forward': forward_val,
            'back': backward_val,
            'left': left_val,
            'right': right_val,
        })

        return [
            ('forward', bool(forward_val > threshold)),
            ('back', bool(backward_val > threshold)),
            ('left', bool(left_val > threshold)),
            ('right', bool(right_val > threshold)),
        ]
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"[+] Model loaded from {path}")
    
    def summary(self):
        print(self)

    def _create_dataset(self, record_files):
        class DrivingDataset(Dataset):
            def _calculate_reward(self, current, previous):
                reward = 0.0
                
                try:
                    current_pos = np.array(current.car_position, dtype=np.float32)
                    previous_pos = np.array(previous.car_position, dtype=np.float32)
                    
                    if np.isnan(current_pos).any() or np.isnan(previous_pos).any():
                        return -10.0
                    
                    distance_moved = np.linalg.norm(current_pos - previous_pos)
                    
                    if distance_moved > 100.0 or np.isnan(distance_moved):
                        return -50.0
                    
                    speed = float(current.car_speed)
                    rays = np.array(current.raycast_distances, dtype=np.float32)
                    rays = np.nan_to_num(rays, nan=100.0, posinf=100.0, neginf=0.0)
                    
                    if len(rays) == 0:
                        return -10.0
                    
                    min_ray = np.min(rays)
                    avg_ray = np.mean(rays)
                    
                    if speed < 1.5:
                        reward -= 5.0  
                    elif speed > 40.0:
                        reward -= (speed - 40.0) * 2.0
                    else:
                        reward += speed * 0.5
                    reward += distance_moved * 5.0
                    
                    reward += min_ray * 5.0
                    
                    reward += avg_ray * 2.0
                    
                    if min_ray <= 1.5:
                        reward -= 100.0
                    elif min_ray < 5.0:
                        reward -= 35.0
                    
                    controls = current.current_controls
                    if controls[0] and controls[1]:
                        reward -= 30.0
                    if controls[2] and controls[3]:
                        reward -= 30.0
                    
                    reward = np.clip(reward, -200.0, 100.0)
                    
                except Exception as e:
                    return -10.0
                
                return float(reward)
            
            def __init__(self, record_files):
                self.data = []
                self.samples = []
                
                print(f"\nLoading {len(record_files)} files...")
                
                for file in record_files:
                    print(f"  Loading {file}...")
                    try:
                        with lzma.open(file, "rb") as f:
                            records = pickle.load(f)
                            self.data.extend(records)
                    except Exception as e:
                        print(f"  Error loading {file}: {e}")
                        continue
                
                print(f"  Total records loaded: {len(self.data)}")
                
                for i in range(1, len(self.data)):
                    current = self.data[i]
                    previous = self.data[i-1]
                    
                    if not hasattr(current, 'raycast_distances') or not hasattr(current, 'car_speed'):
                        continue
                    if not hasattr(current, 'current_controls') or not hasattr(current, 'car_position'):
                        continue
                    if not hasattr(previous, 'car_position'):
                        continue
                    if len(current.raycast_distances) == 0:
                        continue
                    
                    features = np.array([
                        float(current.car_speed),
                        float(current.car_angle),
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
                    
                    reward = self._calculate_reward(current, previous)
                    
                    if np.isnan(reward) or np.isinf(reward):
                        continue
                    
                    self.samples.append({
                        'features': features,
                        'labels': labels,
                        'reward': reward
                    })
                
                print(f"  Valid samples created: {len(self.samples)}")
                
                if len(self.samples) > 0:
                    all_rewards = [s['reward'] for s in self.samples]
                    print(f"  Reward stats: min={min(all_rewards):.2f}, "
                          f"max={max(all_rewards):.2f}, mean={np.mean(all_rewards):.2f}")
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                return (
                    torch.from_numpy(sample['features']).float(),
                    torch.from_numpy(sample['labels']).float(),
                    torch.tensor(sample['reward'], dtype=torch.float32)
                )
        
        return DrivingDataset(record_files)
    
    def loss_function(self, outputs, labels, rewards):
        """Enhanced loss: reward-weighted imitation + safety penalties"""
        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        imitation_loss = bce_loss(outputs, labels)
        
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        weights = torch.exp(rewards_normalized * 0.3)
        weights = torch.clamp(weights, 0.1, 3.0)
        
        weighted_imitation_loss = imitation_loss * weights.unsqueeze(1)
        
        probs = torch.sigmoid(outputs)
        
        forward_back_penalty = (probs[:, 0] * probs[:, 1]) ** 2
        left_right_penalty = (probs[:, 2] * probs[:, 3]) ** 2
        coherence_loss = (forward_back_penalty + left_right_penalty).mean()
        
        forward_prob = probs[:, 0]
        back_prob = probs[:, 1]
        speed_penalty = (forward_prob - back_prob).clamp(min=0) * 0.5
        
        negative_rewards = (rewards < 0).float()
        forward_on_bad = forward_prob * negative_rewards
        risky_forward_loss = forward_on_bad.mean()
        
        total_loss = (
            weighted_imitation_loss.mean() + 
            coherence_loss * 3.0 + 
            risky_forward_loss * 2.0
        )
        
        return total_loss
    
    def train_model(self, record_files, epochs=10, batch_size=32, learning_rate=0.001, 
                    train_split=0.7, val_split=0.15):
        """Train the model on recorded driving data"""
        dataset = self._create_dataset(record_files)
        
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            total_train_loss = 0.0
            for features, labels, rewards in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                rewards = rewards.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.forward(features)
                loss = self.loss_function(outputs, labels, rewards)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for features, labels, rewards in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    rewards = rewards.to(self.device)
                    
                    outputs = self.forward(features)
                    loss = self.loss_function(outputs, labels, rewards)
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