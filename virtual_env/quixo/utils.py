from copy import deepcopy
import random
import numpy as np
from game import Move, Game

def verifie_move(from_pos: tuple[int, int], slide: Move, game: Game) -> bool:
    """
        It checks if the move is valid, an invalid move is useless, so there is no point in computing it
    """
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

def simulate_move(from_pos: tuple[int, int], slide: Move, game: 'Game') -> np.ndarray | None:
    """
        It computes the next board after the action is taken
    """
    copy_game = deepcopy(game)
    acceptable = verifie_move(from_pos, slide, copy_game)
    
    # should not happen check move before entering here
    if not acceptable:
        return None
    
    from_pos = (from_pos[1], from_pos[0])
    
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
    """
        It computes the next board after the action is taken
        and then calculates the score for each row, column, and diagonal
        and then takes the maximum score
    """
    copy_board = simulate_move(from_pos, slide, game)
    if copy_board is None:
        return -1

    piece = game.get_current_player()

    count_row = np.max(np.sum(copy_board == piece, axis=1))
    count_col = np.max(np.sum(copy_board == piece, axis=0))
    count_diag = np.max(np.sum(np.diag(copy_board) == piece))
    count_diag_2 = np.max(np.sum(np.diag(np.fliplr(copy_board)) == piece))
    
    score = max(count_row, count_col, count_diag, count_diag_2)
    return score