from copy import deepcopy
import numpy as np
from game import Move, Game

def verifie_move(from_pos: tuple[int, int], slide: Move, game: Game) -> bool:
    from_pos = (from_pos[1], from_pos[0])
    player_id = game.get_current_player()
    
    acceptable_take: bool = (
        (from_pos[0] == 0 and from_pos[1] < 5)
        or (from_pos[0] == 4 and from_pos[1] < 5)
        or (from_pos[1] == 0 and from_pos[0] < 5)
        or (from_pos[1] == 4 and from_pos[0] < 5)
    ) and (game.get_board()[from_pos] < 0 or game.get_board()[from_pos] == player_id)
    
    if not acceptable_take:
        return False
    
    SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    if from_pos not in SIDES:
        acceptable_top: bool = from_pos[0] == 0 and (
            slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
        )
        acceptable_bottom: bool = from_pos[0] == 4 and (
            slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
        )
        acceptable_left: bool = from_pos[1] == 0 and (
            slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
        )
        acceptable_right: bool = from_pos[1] == 4 and (
            slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
        )
    else:
        acceptable_top: bool = from_pos == (0, 0) and (
            slide == Move.BOTTOM or slide == Move.RIGHT)
        acceptable_left: bool = from_pos == (4, 0) and (
            slide == Move.TOP or slide == Move.RIGHT)
        acceptable_right: bool = from_pos == (0, 4) and (
            slide == Move.BOTTOM or slide == Move.LEFT)
        acceptable_bottom: bool = from_pos == (4, 4) and (
            slide == Move.TOP or slide == Move.LEFT)
    
    acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
    return acceptable

def simulate_move(from_pos: tuple[int, int], slide: Move, game: 'Game') -> np.array:
    acceptable = verifie_move(from_pos, slide, game)
    
    # should not happen check move before entering here
    if not acceptable:
        return None
    
    from_pos = (from_pos[1], from_pos[0])
    
    copy_game = deepcopy(game)
    copy_board = copy_game.get_board()
    
    piece = copy_game.get_current_player()
    copy_board[from_pos] = piece
    
    if slide == Move.LEFT:
        for i in range(from_pos[1], 0, -1):
            copy_board[(from_pos[0], i)] = copy_board[(from_pos[0], i - 1)]
        copy_board[(from_pos[0], 0)] = piece
    elif slide == Move.RIGHT:
        for i in range(from_pos[1], copy_board.shape[1] - 1, 1):
            copy_board[(from_pos[0], i)] = copy_board[(from_pos[0], i + 1)]
        copy_board[(from_pos[0], copy_board.shape[1] - 1)] = piece
    elif slide == Move.TOP:
        for i in range(from_pos[0], 0, -1):
            copy_board[(i, from_pos[1])] = copy_board[(i - 1, from_pos[1])]
        copy_board[(0, from_pos[1])] = piece
    elif slide == Move.BOTTOM:
        for i in range(from_pos[0], copy_board.shape[0] - 1, 1):
            copy_board[(i, from_pos[1])] = copy_board[(i + 1, from_pos[1])]
        copy_board[(copy_board.shape[0] - 1, from_pos[1])] = piece
        
    return copy_board

def fitness(from_pos: tuple[int, int], slide: Move, game: 'Game') -> int:
    copy_board = simulate_move(from_pos, slide, game)
    if copy_board is None:
        return -1

    piece = game.get_current_player()
    opponent_piece = 0 if piece == 1 else 1

    def calculate_score(line: np.ndarray) -> int:
        # Calculate score for a row, column, or diagonal
        score = 0
        length = len(line)
        contiguous_count = 0

        for i in range(length):
            if line[i] == piece:
                contiguous_count += 1
                score += contiguous_count
            else:
                contiguous_count = 0
                if line[i] == opponent_piece:
                    # Penalty for opponent's piece in the line
                    score -= 1

        return score

    row_score = np.sum(np.apply_along_axis(calculate_score, 1, copy_board))
    col_score = np.sum(np.apply_along_axis(calculate_score, 0, copy_board))
    principal_diag_score = calculate_score(np.diag(copy_board))
    secondary_diag_score = calculate_score(np.diag(np.fliplr(copy_board)))

    # Summing up scores with penalties
    score = row_score + col_score + principal_diag_score + secondary_diag_score

    return score