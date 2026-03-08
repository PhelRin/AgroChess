import chess
import chess.polyglot
import numpy as np
import random

ACTION_SPACE_SIZE = 4672

def encode_action(move: chess.Move, board: chess.Board):
    from_sq = move.from_square
    to_sq = move.to_square
    from_r, from_c = divmod(from_sq, 8)
    to_r, to_c = divmod(to_sq, 8)
    
    dr = to_r - from_r
    dc = to_c - from_c
    
    if move.promotion and move.promotion != chess.QUEEN:
        p_dir = dc + 1 
        p_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK].index(move.promotion)
        plane = 64 + p_dir * 3 + p_piece
        return plane * 64 + from_sq
        
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    if (dr, dc) in knight_moves:
        plane = 56 + knight_moves.index((dr, dc))
        return plane * 64 + from_sq
        
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
    
    dir_idx = -1
    for i, (d_r, d_c) in enumerate(directions):
        if dr * d_c == dc * d_r and np.sign(dr) == np.sign(d_r) and np.sign(dc) == np.sign(d_c):
            dir_idx = i
            break
            
    dist = max(abs(dr), abs(dc))
    plane = dir_idx * 7 + (dist - 1)
    return plane * 64 + from_sq

def decode_action(action_idx: int, board: chess.Board):
    plane, from_sq = divmod(action_idx, 64)
    from_r, from_c = divmod(from_sq, 8)
    
    to_r, to_c = -1, -1
    promotion = None
    
    if plane >= 64:
        p_idx = plane - 64
        p_dir = p_idx // 3 - 1
        p_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK][p_idx % 3]
        
        to_r = 7 if board.turn == chess.WHITE else 0
        to_c = from_c + p_dir
        promotion = p_piece
    elif plane >= 56:
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        dr, dc = knight_moves[plane - 56]
        to_r = from_r + dr
        to_c = from_c + dc
        
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and (to_r == 7 or to_r == 0):
            promotion = chess.QUEEN
    else:
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr, dc = directions[dir_idx]
        to_r = from_r + dr * dist
        to_c = from_c + dc * dist
        
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and (to_r == 7 or to_r == 0):
            promotion = chess.QUEEN
            
    to_sq = to_r * 8 + to_c
    return chess.Move(from_sq, to_sq, promotion)

class ChessEnv:
    def __init__(self, book_path=None):
        self.board = chess.Board()
        self.book_path = book_path
        self.history = []
        self.reset()

    def reset(self):
        self.board.reset()
        if self.book_path:
            try:
                with chess.polyglot.open_reader(self.book_path) as reader:
                    num_book_moves = random.randint(3, 8)
                    for _ in range(num_book_moves):
                        entries = list(reader.find_all(self.board))
                        if not entries:
                            break
                        weights = [entry.weight for entry in entries]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            probs = [w / total_weight for w in weights]
                            move = np.random.choice([e.move for e in entries], p=probs)
                        else:
                            move = random.choice([e.move for e in entries])
                        self.board.push(move)
            except Exception as e:
                print(f"Warning: Could not read book entries: {e}")
                
        self.history = [self.get_board_planes(self.board)]
        return self.get_state()

    def get_legal_actions(self):
        legal_moves = list(self.board.legal_moves)
        return [encode_action(move, self.board) for move in legal_moves]

    def get_board_planes(self, board):
        planes = np.zeros((14, 8, 8), dtype=np.float32)

        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for color, offset in [(chess.WHITE, 0), (chess.BLACK, 6)]:
                mask = board.pieces_mask(pt, color)
                pt_idx = pt - 1 + offset
                for sq in chess.SquareSet(mask):
                    planes[pt_idx, sq // 8, sq % 8] = 1.0
                    
        return planes

    def get_state(self):
        state = np.zeros((119, 8, 8), dtype=np.float32)
        
        hist_len = len(self.history)
        
        # 8 historical states * 14 planes = 112 planes
        for i in range(8):
            if i < hist_len:
                # History is ordered oldest to newest, so latest is at the end
                # self.history[-1] is the current board
                state[i*14 : (i+1)*14] = self.history[-(i+1)]
                
        # 112: Turn
        state[112, :, :] = 1.0 if self.board.turn == chess.WHITE else 0.0

        # 113-116: Castling rights
        state[113, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.WHITE) else 0.0
        state[114, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.WHITE) else 0.0
        state[115, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.BLACK) else 0.0
        state[116, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        # 117-118: En passant square mapping
        ep_square = self.board.ep_square
        if ep_square is not None:
            r, c = divmod(ep_square, 8)
            state[117, r, c] = 1.0 # Exact square
            state[118, :, c] = 1.0 # Entire file

        return state

    def push(self, move):
        self.board.push(move)
        self.history.append(self.get_board_planes(self.board))

    def pop(self):
        self.board.pop()
        self.history.pop()

    def step(self, action_idx):
        move = decode_action(action_idx, self.board)
        self.push(move)
        done = self.board.is_game_over(claim_draw=True)
        return self.get_state(), 0.0, done

    def render(self):
        print(self.board)
