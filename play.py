import sys
import chess
import torch
import numpy as np
from model import ChessNet
from mcts import MCTS
from env import ChessEnv, encode_action, decode_action

def play_human_vs_bot():
    print("Welcome to Aggressive Chess Bot!")
    color_choice = input("Do you want to play as White (w) or Black (b)? ").strip().lower()
    human_turn = chess.WHITE if color_choice == 'w' else chess.BLACK
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    net = ChessNet()
    try:
        net.load_state_dict(torch.load("model.pt", map_location=device))
        print("Loaded trained model weights.")
    except Exception as e:
        print("Could not load model.pt! Bot will play randomly.")
        
    net.to(device)
    net.eval()
    
    mcts_args = {
        'num_simulations': 200,
        'batch_size': 16,
        'virtual_loss': 3,
        'pb_c_base': 19652,
        'pb_c_init': 1.25,
        'root_dirichlet_alpha': 0.0, 
        'root_exploration_fraction': 0.0
    }
    mcts = MCTS(net, mcts_args)
    env = ChessEnv() 
    env.book_path = None
    env.reset()
    
    while not env.board.is_game_over(claim_draw=True):
        print("\n" + env.board.unicode() + "\n")
        
        if env.board.turn == human_turn:
            while True:
                move_str = input("Enter your move (e.g. e2e4): ")
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in env.board.legal_moves:
                        action_idx = encode_action(move, env.board)
                        env.step(action_idx)
                        break
                    else:
                        print("Illegal move. Try again.")
                except Exception:
                    print("Invalid format. Use UCI format like e2e4 or g1f3.")
        else:
            print("Bot is thinking...")
            target_policy = mcts.search(env)
            action = np.argmax(target_policy)
            move = decode_action(action, env.board)
            print(f"\n>>> Bot plays: {move.uci()}")
            env.step(action)
            
    print("\n" + env.board.unicode() + "\n")
    print("Game Over!")
    print("Result:", env.board.result(claim_draw=True))

if __name__ == "__main__":
    play_human_vs_bot()
