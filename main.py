import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import time
import pickle
from model import ChessNet
from train import train_model
from selfplay import generate_selfplay_data

def run_training_orchestrator(iterations=30, games_per_iteration=150, epochs_per_iteration=3):
    net = ChessNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Orchestrator using GPU device: {device}")
    
    if os.path.exists("model.pt"):
        try:
            net.load_state_dict(torch.load("model.pt", map_location=device))
            print("Loaded existing model weights.")
        except Exception:
            print("Could not load model weights. Starting fresh.")
            
    net.to(device)
        
    os.makedirs('data_v2', exist_ok=True)

    mcts_args = {
        'num_simulations': 50,
        'batch_size': 16,
        'virtual_loss': 3,
        'pb_c_base': 19652,
        'pb_c_init': 1.25,
        'root_dirichlet_alpha': 0.3,
        'root_exploration_fraction': 0.25
    }

    for i in range(1, iterations + 1):
        print(f"\n=== Iteration {i}/{iterations} ===")
        print(f"Generating {games_per_iteration} self-play games... This will heavily utilize the GPU.")
        
        # 1. Generate data via self-play 
        data = generate_selfplay_data(net, num_games=games_per_iteration, book_path='ph-gambitbook.bin', mcts_args=mcts_args)
        
        # 2. Save data to buffer
        timestamp = int(time.time())
        file_path = f"data_v2/selfplay_{timestamp}_iter_{i}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} training examples to {file_path}.")
        
        # 3. Train the network on historical replay buffer
        print("Training on historical Replay Buffer...")
        train_model(net, epochs=epochs_per_iteration, data_dir='data_v2')
        
    print("Finished massive training loop!")

if __name__ == "__main__":
    try:
        run_training_orchestrator(iterations=30, games_per_iteration=150, epochs_per_iteration=3)
    except KeyboardInterrupt:
        print("Training stopped safely.")
