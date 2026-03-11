import os
import glob
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import ChessNet

class ChessDataset(Dataset):
    def __init__(self, data_dir='data_v2', max_files=25):
        self.data = []
        
        # Sliding Window Replay Buffer
        all_files = glob.glob(os.path.join(data_dir, '*.pkl'))
        # Sort files by exact creation time (newest last)
        all_files.sort(key=os.path.getmtime)
        
        # Purge files that are older than the sliding window limit
        files_to_delete = all_files[:-max_files] if len(all_files) > max_files else []
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Replay Buffer Purge: Deleted old memory {f}")
            except Exception as e:
                pass
                
        # Load only the newest files
        files_to_load = all_files[-max_files:] if max_files > 0 else all_files
        
        for f in files_to_load:
            with open(f, 'rb') as file:
                self.data.extend(pickle.load(file))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return (torch.FloatTensor(state), 
                torch.FloatTensor(policy), 
                torch.FloatTensor([value]))

def train_model(net, epochs=5, batch_size=128, lr=5e-4, data_dir='data_v2'):
    dataset = ChessDataset(data_dir=data_dir)
    if len(dataset) == 0:
        print("No training data found in", data_dir)
        return
        
    device = next(net.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4) 
    
    def policy_loss_fn(pred_logits, target_probs):
        log_probs = nn.functional.log_softmax(pred_logits, dim=1)
        loss = -torch.sum(target_probs * log_probs, dim=1).mean()
        return loss

    value_loss_fn = nn.MSELoss()
    
    net.train()
    for epoch in range(epochs):
        total_p_loss = 0
        total_v_loss = 0
        for states, policies, values in dataloader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p_logits, v_pred = net(states)
                
                p_loss = policy_loss_fn(p_logits, policies)
                v_loss = value_loss_fn(v_pred, values)
                
                loss = p_loss + v_loss
                
            loss.backward()
            optimizer.step()
            
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Policy Loss: {total_p_loss/len(dataloader):.4f} | Value Loss: {total_v_loss/len(dataloader):.4f}")
        
    torch.save(net.state_dict(), "model.pt")
    print("Model saved to model.pt")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    net = ChessNet()
    if os.path.exists("model.pt"):
        try:
            net.load_state_dict(torch.load("model.pt", map_location=device))
            print("Loaded existing model weights.")
        except Exception:
            print("Could not load existing model weights. Starting fresh.")
            
    net.to(device)
        
    print("Starting training on default data path...")
    train_model(net, epochs=10)
