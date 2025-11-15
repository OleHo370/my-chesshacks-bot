from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn as nn
import numpy as np
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
                plane_idx = (0 if piece.color else 6) + piece.piece_type - 1
                piece_planes[plane_idx, square] = 1.0

        features.extend(piece_planes.flatten())

        attack_defense = np.zeros(64, dtype=np.float32)

        for square in range(64):
            attackers_white = len(board.attackers(True, square))
            attackers_black = len(board.attackers(False, square))
            attack_defense[square] = (attackers_white - attackers_black) / 10.0

        features.extend(attack_defense)

        game_state = [
            1.0 if board.is_check() else 0.0,
            1.0 if board.is_checkmate() else 0.0,
            1.0 if board.is_stalemate() else 0.0,
            1.0 if board.is_insufficient_material() else 0.0,
            1.0 if board.can_claim_threefold_repetition() else 0.0,
            1.0 if board.can_claim_fifty_moves() else 0.0,
            1.0 if board.turn else 0.0,
            min(len(board.move_stack)/100.0, 1.0)
        ]
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

    return model

def order_moves(board, moves):
    move_scores = []

    for move in moves:
        score = 0

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)

            if captured_piece:
                victim_value = [0,1,3,3,5,9,0][captured_piece.piece_type]
                attacker = board.piece_at(move.from_square)
                attacker_value = [0,1,3,3,5,9,0][attacker.piece_type] if attacker else 0
                score += (victim_value - attacker_value*0.1)*10
        board.push(move)

        if board.is_check():
            score += 5

        board.pop()

        if move.promotion:
            score += 8
        move_scores.append((score, move))

    move_scores.sort(reverse=True, key=lambda x:x[0])
    return [move for _, move in move_scores]

def minimax_search(board, depth, alpha, beta, maximizing_player):

    if depth == 0 or board.is_game_over():
        board_features = encoder.encode_board(board)
        board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(device)

        with torch.no_grad():
            evaluation = model(board_tensor).item()

        if board.is_checkmate():
            return (10.0 if not board.turn else -10.0), None
        
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0, None
        
        return evaluation, None
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return 0.0, None
    legal_moves = order_moves(board, legal_moves)
    best_move = legal_moves[0]

    if maximizing_player:
        max_eval = float('-inf')

        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax_search(board, depth-1, alpha, beta, False)
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)

            if beta <= alpha:
                break
        return max_eval, best_move
    
    else:
        min_eval = float('inf')

        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax_search(board, depth-1, alpha, beta, True)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)

            if beta <= alpha:
                break

        return min_eval, best_move

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
    legal_moves = list(ctx.board.generate_legal_moves())

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    num_pieces = len(ctx.board.piece_map())
    
    if num_pieces <= 10:
        search_depth = 5
    elif num_pieces <= 20:
        search_depth = 4
    else:
        search_depth = 3

    maximizing = ctx.board.turn
    best_eval, best_move = minimax_search(ctx.board, search_depth, float('-inf'), float('inf'), maximizing)
    if best_move is None:
        best_move = legal_moves[0]
    move_scores = {move: evaluate_position(ctx.board, move) for move in legal_moves}
    scores_array = np.array(list(move_scores.values()))

    if not ctx.board.turn:
        scores_array = -scores_array

    exp_scores = np.exp(scores_array / 0.5)

    probabilities = exp_scores / np.sum(exp_scores)

    move_probs = {move: prob for move, prob in zip(legal_moves, probabilities)}
    ctx.logProbabilities(move_probs)
    print(f"Selected: {best_move} (eval: {best_eval:.3f})")
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    global model

    if model is None:
        initialize_model()