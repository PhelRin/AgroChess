import os
import time
import torch
import numpy as np
import random
import chess
import chess.pgn

from concurrent.futures import ThreadPoolExecutor
from env import ChessEnv, ACTION_SPACE_SIZE, decode_action
from mcts import MCTS
from model import ChessNet

def get_shaping_reward(board: chess.Board, move: chess.Move):
    reward = 0.0
    piece_moved = board.piece_at(move.from_square)
    is_capture = board.is_capture(move)
    captured_piece = board.piece_at(move.to_square)
    is_ep = board.is_en_passant(move)
    if is_ep:
        captured_piece = chess.Piece(chess.PAWN, not board.turn)
        
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    # Material Debt Penalty (punish player every move they spend down material to force urgency)
    my_mat = sum(len(board.pieces(pt, board.turn)) * val for pt, val in values.items() if pt != chess.KING)
    enemy_mat = sum(len(board.pieces(pt, not board.turn)) * val for pt, val in values.items() if pt != chess.KING)
    
    if my_mat < enemy_mat:
        reward -= 0.08 * (enemy_mat - my_mat)
    
    if piece_moved and piece_moved.piece_type != chess.KING:
        to_rank = chess.square_rank(move.to_square)
        if piece_moved.color == chess.WHITE and to_rank >= 4:
            reward += 0.02
        elif piece_moved.color == chess.BLACK and to_rank <= 3:
            reward += 0.02
            
    enemy_king_square = board.king(not board.turn)
    if enemy_king_square is not None and piece_moved and piece_moved.piece_type != chess.KING:
        enemy_king_rank = chess.square_rank(enemy_king_square)
        enemy_king_file = chess.square_file(enemy_king_square)
        my_piece_rank = chess.square_rank(move.to_square)
        my_piece_file = chess.square_file(move.to_square)
        
        distance = max(abs(enemy_king_rank - my_piece_rank), abs(enemy_king_file - my_piece_file))
        if distance <= 2:
            reward += 0.05
            
    board.push(move)
    if board.is_repetition(2):
        reward -= 0.25 # Repetition punishment to prevent shuffling
        
    if board.is_check():
        reward += 0.1
    if board.is_checkmate():
        reward += 0.5
    board.pop()
    
    if is_capture and captured_piece:
        val_captured = values.get(captured_piece.piece_type, 0)
        # Give a massive reward for taking valuable pieces!
        # Taking a Queen gives +0.45. Taking a pawn gives +0.05.
        reward += 0.05 * val_captured
            
    return reward

def play_single_game(game_idx, num_games, net, book_path, mcts_args):
    # Instantiate MCTS and Env INSIDE the thread so they don't share memory and crash
    mcts = MCTS(net, mcts_args)
    env = ChessEnv(book_path=book_path)
    game_history =[]
    step_idx = 0
    white_shaped_reward = 0.0
    black_shaped_reward = 0.0
    game_dataset =[]
    
    while True:
        state = env.get_state()
        target_policy = mcts.search(env)
        turn_now = env.board.turn
        
        game_history.append((state, target_policy, turn_now))
        
        # Change 30 to 10. Only play randomly for the first 5 full moves to create
        # opening variety, then play perfectly ruthlessly for the rest of the game.
        if step_idx < 10:
            action = np.random.choice(ACTION_SPACE_SIZE, p=target_policy)
        else:
            action = np.argmax(target_policy)
            
        turn_before_move = env.board.turn
        move = decode_action(action, env.board)
        reward = get_shaping_reward(env.board, move)
        
        new_state, _, done = env.step(action)
        
        if turn_before_move == chess.WHITE:
            white_shaped_reward += reward
        else:
            black_shaped_reward += reward
        
        step_idx += 1
        
        if done:
            result = env.board.result(claim_draw=True)
            white_shaped_advantage = white_shaped_reward - black_shaped_reward
            
            is_decisive = result in ['1-0', '0-1']
            
            if is_decisive:
                for state, t_pol, turn in game_history:
                    if result == '1-0':
                        rel_terminal_val = 1.0 if turn == chess.WHITE else -1.0
                    elif result == '0-1':
                        rel_terminal_val = -1.0 if turn == chess.WHITE else 1.0
                        
                    rel_shaping_adv = white_shaped_advantage if turn == chess.WHITE else -white_shaped_advantage
                    
                    speed_modifier = 0.0
                    if step_idx > 60:
                        # The old penalty for taking too long
                        speed_modifier = -0.05 * ((step_idx - 60) ** 1.3)
                    else:
                        # THE NEW SPEEDRUN BONUS!
                        # If it checkmates fast, it gets a massive terminal bonus.
                        # e.g., Mating on step 20 gives (60-20) * 0.01 = +0.40
                        speed_modifier = 0.01 * (60 - step_idx)
                        
                    scaled_shaping = np.tanh(rel_shaping_adv) * 0.4 + speed_modifier
                    
                    target_val = np.clip(rel_terminal_val + scaled_shaping, -1.0, 1.0)
                    game_dataset.append((state, t_pol, target_val))
            
            pgn = chess.pgn.Game.from_board(env.board)
            
            # Combine output into one string so threads don't jumble the console prints
            output = f"Game {game_idx+1}/{num_games} finished after {step_idx} steps. Result: {result}, Advantage: {white_shaped_advantage:.2f}\n"
            if is_decisive:
                output += f"PGN:\n{pgn}\n"
            else:
                output += f"Game {game_idx+1} was a draw. Discarding from dataset to save space.\n"
            print(output)
            
            # Drop cache to save RAM inside this thread
            mcts.nn_cache.clear()
            
            return game_dataset

def generate_selfplay_data(net, num_games, book_path=None, mcts_args=None):
    net.eval()
    datasets =[]
    
    print(f"Starting Multi-Threaded Self-Play (2 Workers) for {num_games} games...")
    
    # 2 threads running 2 MCTS environments simultaneously!
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all games to the thread pool
        futures =[executor.submit(play_single_game, i, num_games, net, book_path, mcts_args) for i in range(num_games)]
        
        # Gather the results as they finish
        for future in futures:
            try:
                game_data = future.result()
                if game_data: # If it wasn't a draw (which returns empty)
                    datasets.extend(game_data)
            except Exception as e:
                print(f"Game failed with an error: {e}")
                
    return datasets

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    net = ChessNet()
    if os.path.exists("model.pt"):
        try:
            net.load_state_dict(torch.load("model.pt", map_location=device))
        except Exception:
            pass
            
    net.to(device)
    print("Starting self-play generation...")
    data = generate_selfplay_data(net, num_games=3, book_path='ph-gambitbook.bin')
    
    import pickle
    os.makedirs('data_v2', exist_ok=True)
    timestamp = int(time.time())
    if data:
        with open(f"data_v2/selfplay_{timestamp}.pkl", "wb") as f:
            pickle.dump(data, f)
        print("Self-play data saved.")
    else:
        print("No decisive games generated.")