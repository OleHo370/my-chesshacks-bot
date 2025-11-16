from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn as nn
import numpy as np
import os
import time

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

nn_eval_cache = {}
MAX_CACHE_SIZE = 100000

game_start_time = None
total_time_budget = 57.0 
moves_made = 0

search_start_time = None
time_limit = None
nodes_searched = 0


def initialize_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "weights", "model.pt")
    model = ChessEvaluationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_time_for_move():
    global game_start_time, total_time_budget, moves_made
    
    if game_start_time is None:
        return 0.25
    
    elapsed = time.time() - game_start_time
    remaining = max(total_time_budget - elapsed, 0.1)
    
    estimated_moves_left = max(120 - moves_made, 8)
    time_per_move = remaining / estimated_moves_left
    

    return max(min(time_per_move * 1.2, 0.5), 0.05)


def should_stop_search():
    global search_start_time, time_limit
    if time_limit is None or search_start_time is None:
        return False
    return (time.time() - search_start_time) >= time_limit


def evaluate_with_nn(board):
    fen = board.fen().split(' ')[0]
    
    if fen in nn_eval_cache:
        return nn_eval_cache[fen]
    
    board_features = encoder.encode_board(board)
    board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(device)

    with torch.no_grad():
        evaluation = model(board_tensor).item()
    
    if len(nn_eval_cache) >= MAX_CACHE_SIZE:
        items_to_remove = MAX_CACHE_SIZE // 5
        for key in list(nn_eval_cache.keys())[:items_to_remove]:
            del nn_eval_cache[key]
    
    nn_eval_cache[fen] = evaluation
    return evaluation


def order_moves(board, moves):
    move_scores = []

    for move in moves:
        score = 0

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                victim_value = [0, 1, 3, 3, 5, 9, 0][captured_piece.piece_type]
                attacker = board.piece_at(move.from_square)
                attacker_value = [0, 1, 3, 3, 5, 9, 0][attacker.piece_type] if attacker else 0
                score += 1000 + (victim_value * 10 - attacker_value)
        
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()

        if move.promotion:
            score += 900
        
        move_scores.append((score, move))

    move_scores.sort(reverse=True, key=lambda x: x[0])
    return [move for _, move in move_scores]


def quiescence_search(board, alpha, beta, maximizing_player, depth=0):
    global nodes_searched
    nodes_searched += 1
    
    if nodes_searched % 1000 == 0 and should_stop_search():
        return 0.0
    
    if depth >= 6:
        return evaluate_with_nn(board)
    
    stand_pat = evaluate_with_nn(board)
    
    if maximizing_player:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
    
    tactical_moves = [m for m in board.legal_moves 
                      if board.is_capture(m) or board.gives_check(m)]
    
    if not tactical_moves:
        return stand_pat
    
    tactical_moves = order_moves(board, tactical_moves)
    
    for move in tactical_moves:
        board.push(move)
        
        if board.is_checkmate():
            score = -10.0
        else:
            score = -quiescence_search(board, -beta, -alpha, not maximizing_player, depth + 1)
        
        board.pop()
        
        if maximizing_player:
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        else:
            beta = min(beta, score)
            if beta <= alpha:
                break
    
    return alpha if maximizing_player else beta


def minimax_search(board, depth, alpha, beta, maximizing_player, use_quiescence=True):
    global nodes_searched
    nodes_searched += 1
    
    if nodes_searched % 100 == 0 and should_stop_search():
        return 0.0, None
    
    if board.is_checkmate():
        return -10.0, None
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0, None
    
    if depth == 0:
        if use_quiescence:
            eval_score = quiescence_search(board, alpha, beta, maximizing_player)
        else:
            eval_score = evaluate_with_nn(board)
        return eval_score, None
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return 0.0, None
    
    legal_moves = order_moves(board, legal_moves)
    best_move = legal_moves[0]

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax_search(board, depth - 1, alpha, beta, False, use_quiescence)
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
            eval_score, _ = minimax_search(board, depth - 1, alpha, beta, True, use_quiescence)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        
        return min_eval, best_move


def iterative_deepening_search(board, max_depth, maximizing, time_budget):
    global search_start_time, time_limit, nodes_searched
    
    search_start_time = time.time()
    time_limit = time_budget
    nodes_searched = 0
    
    best_move = None
    best_eval = 0.0
    
    for depth in range(1, max_depth + 1):
        if should_stop_search():
            break
        
        eval_score, move = minimax_search(
            board, depth, float('-inf'), float('inf'), maximizing
        )
        
        if move is not None:
            best_move = move
            best_eval = eval_score
        
        if abs(eval_score) > 9.5:
            break
    
    return best_eval, best_move


def batch_evaluate_moves(board, moves):
    if len(moves) == 0:
        return {}
    
    features_list = []
    moves_to_eval = []
    
    for move in moves:
        board.push(move)
        fen = board.fen().split(' ')[0]
        
        if fen not in nn_eval_cache:
            features = encoder.encode_board(board)
            features_list.append(features)
            moves_to_eval.append((move, fen))
        
        board.pop()
    
    if features_list:
        batch_tensor = torch.from_numpy(np.array(features_list)).to(device)
        with torch.no_grad():
            evaluations = model(batch_tensor).squeeze().cpu().numpy()
        
        if len(features_list) == 1:
            evaluations = [evaluations.item()]
        else:
            evaluations = evaluations.tolist()
        
        for (move, fen), eval_score in zip(moves_to_eval, evaluations):
            if len(nn_eval_cache) < MAX_CACHE_SIZE:
                nn_eval_cache[fen] = eval_score
    
    move_scores = {}
    for move in moves:
        board.push(move)
        fen = board.fen().split(' ')[0]
        move_scores[move] = nn_eval_cache.get(fen, 0.0)
        board.pop()
    
    return move_scores


@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    global model, game_start_time, moves_made

    if model is None:
        initialize_model()
    
    if game_start_time is None:
        game_start_time = time.time()
    
    legal_moves = list(ctx.board.generate_legal_moves())

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    if len(legal_moves) == 1:
        ctx.logProbabilities({legal_moves[0]: 1.0})
        moves_made += 1
        return legal_moves[0]
    
    num_pieces = len(ctx.board.piece_map())
    
    if num_pieces <= 6:
        max_depth = 5
    elif num_pieces <= 12:
        max_depth = 4
    elif num_pieces <= 20:
        max_depth = 3
    else:
        max_depth = 3

    time_for_move = get_time_for_move()
    maximizing = ctx.board.turn
    
    best_eval, best_move = iterative_deepening_search(
        ctx.board, max_depth, maximizing, time_for_move
    )
    
    if best_move is None:
        best_move = legal_moves[0]
    
    move_scores = batch_evaluate_moves(ctx.board, legal_moves)
    scores_array = np.array([move_scores[move] for move in legal_moves])

    if not ctx.board.turn:
        scores_array = -scores_array

    exp_scores = np.exp(scores_array / 0.5)
    probabilities = exp_scores / np.sum(exp_scores)

    move_probs = {move: prob for move, prob in zip(legal_moves, probabilities)}
    ctx.logProbabilities(move_probs)
    
    moves_made += 1
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    global model, nn_eval_cache, game_start_time, moves_made
    global search_start_time, time_limit, nodes_searched

    if model is None:
        initialize_model()
    
    nn_eval_cache.clear()
    game_start_time = None
    moves_made = 0
    search_start_time = None
    time_limit = None
    nodes_searched = 0