import numpy as np

N_ROWS = 3


class TicTacToe:

    def __init__(self):
        self.board = np.zeros((N_ROWS, N_ROWS), dtype=np.str_)
        self.player_x_turn = True
        self.player_x_started = True
        self.winner = None  # winner is in ["x", "o", "t", None] t -> tie, None -> not finished yet

    def play_again(self):
        self.player_x_started = not self.player_x_started
        self.player_x_turn = self.player_x_started
        self.board = np.zeros((N_ROWS, N_ROWS), dtype=np.str_)

    def is_free(self, logical_position):
        return self.board[tuple(logical_position)] == ""

    def get_winner(self):
        for i in range(N_ROWS):
            row_unique_elements = np.unique(self.board[i, :])
            col_unique_elements = np.unique(self.board[:, i])

            if len(row_unique_elements) == 1 and row_unique_elements.item() != "":
                return row_unique_elements.item()
            if len(col_unique_elements) == 1 and col_unique_elements.item() != "":
                return col_unique_elements.item()

        diagonal_unique_elements = np.unique(np.diagonal(self.board))
        antidiagonal_unique_elements = np.unique(np.diagonal(np.flipud(self.board)))
        if len(diagonal_unique_elements) == 1 and diagonal_unique_elements.item() != "":
            return diagonal_unique_elements.item()

        if len(antidiagonal_unique_elements) == 1 and antidiagonal_unique_elements.item() != "":
            return antidiagonal_unique_elements.item()

        if np.all(self.board != ""):
            return "t"

        return ""

    def move(self, logical_position):
        new_symbol = ""
        if not self.is_free(logical_position):
            return new_symbol

        if self.player_x_turn:
            self.board[tuple(logical_position)] = new_symbol = "x"
        else:
            self.board[tuple(logical_position)] = new_symbol = "o"

        self.player_x_turn = not self.player_x_turn
        return new_symbol

    def available_moves(self):
        return np.argwhere((self.board == ""))
