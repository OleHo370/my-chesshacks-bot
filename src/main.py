from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import os

class ChessEvaluationNet(nn.Module):
    def __init__(self, input_size=840, hidden_sizes=[512, 256, 128]):
        super(ChessEvaluationNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class BoardEncoder:
    def __init__(self):
        self.piece_to_idx = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
    
    def encode_board(self, board):
        features = []

        piece_planes = np.zeros((12, 64), dtype=np.float32)
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                color_offset = 0 if piece.color else 6
                plane_idx = color_offset + piece_type - 1
                piece_planes[plane_idx, square] = 1.0
        
        features.extend(piece_planes.flatten())
        
        attack_defense = np.zeros(64, dtype=np.float32)
        for square in range(64):
            attackers_white = len(board.attackers(True, square))
            attackers_black = len(board.attackers(False, square))
            attack_defense[square] = (attackers_white - attackers_black) / 10.0
        
        features.extend(attack_defense)

        game_state = np.zeros(8, dtype=np.float32)
        game_state[0] = 1.0 if board.is_check() else 0.0
        game_state[1] = 1.0 if board.is_checkmate() else 0.0
        game_state[2] = 1.0 if board.is_stalemate() else 0.0
        game_state[3] = 1.0 if board.is_insufficient_material() else 0.0
        game_state[4] = 1.0 if board.can_claim_threefold_repetition() else 0.0
        game_state[5] = 1.0 if board.can_claim_fifty_moves() else 0.0
        game_state[6] = 1.0 if board.turn else 0.0
        game_state[7] = min(len(board.move_stack) / 100.0, 1.0)
        
        features.extend(game_state)
        
        return np.array(features, dtype=np.float32)

model = None
encoder = BoardEncoder()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_model():
    global model
    
    model_path = os.path.join(os.path.dirname(__file__), "weights", "model.pt")
    
    model = ChessEvaluationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {model_path}")
    
    return model


def evaluate_position(board, move):
    board.push(move)
    
    board_features = encoder.encode_board(board)
    board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        evaluation = model(board_tensor).item()
    
    board.pop()
    
    
    return evaluation


@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    global model
    
    if model is None:
        initialize_model()
    
    print(f"Move {len(ctx.board.move_stack) + 1}, Turn: {'White' if ctx.board.turn else 'Black'}")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    move_scores = {}
    for move in legal_moves:
        score = evaluate_position(ctx.board, move)
        move_scores[move] = score
    
    if ctx.board.turn:
        best_move = max(move_scores.items(), key=lambda x: x[1])[0]
    else:
        best_move = min(move_scores.items(), key=lambda x: x[1])[0]
    
    best_score = move_scores[best_move]
    
    temperature = 0.3
    scores_array = np.array(list(move_scores.values()))
    
    if not ctx.board.turn:
        scores_array = -scores_array
    
    exp_scores = np.exp(scores_array / temperature)
    probabilities = exp_scores / np.sum(exp_scores)
    
    move_probs = {move: prob for move, prob in zip(legal_moves, probabilities)}
    ctx.logProbabilities(move_probs)
    
    print(f"Selected: {best_move} (score: {best_score:.3f})")
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    global model
    
    if model is None:
        initialize_model()