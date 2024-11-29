from abc import ABC, abstractmethod

import numpy as np

from game import TicTacToe


def build_player(player_config, game):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, player_config)


class Player(ABC):
    def __init__(self, game):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game: TicTacToe, config):
        super().__init__(game)
        self.pruning_depth = config["pruning_depth"]
        self.player_sign = ""

    def get_move(self, event_position):
        if self.player_sign == "":
            if self.game.player_x_turn:
                self.player_sign = "x"
            else:
                self.player_sign = "o"
        board = self.game.board
        board_successors_with_moves = self.make_successors_with_moves(board, True)
        moves_with_values = []
        for successor, move in board_successors_with_moves:
            moves_with_values.append(
                (
                    move,
                    self.alfa_beta(
                        successor, self.pruning_depth, False, -np.inf, np.inf
                    ),
                )
            )

        best_move = max(moves_with_values, key=lambda x: x[1])[0]

        return best_move

    def alfa_beta(self, board: np.ndarray, depth, move_max, alfa, beta):
        winner = self.get_winner(board)
        if winner:
            return self.evaluate_terminal_state(winner)
        # if depth == 0:
        #     return self.evaluate_state(board, move_max)

        successors = self.make_successors_with_moves(board, move_max)
        if move_max:
            for successor, move in successors:
                alfa = max(
                    alfa, self.alfa_beta(successor, depth - 1, not move_max, alfa, beta)
                )
                if alfa >= beta:
                    return alfa
            return alfa
        else:
            for successor, move in successors:
                beta = min(
                    beta, self.alfa_beta(successor, depth - 1, not move_max, alfa, beta)
                )
                if beta <= alfa:
                    return beta
            return beta

    def evaluate_terminal_state(self, winner):
        if winner == "t":
            return 0
        if winner == self.player_sign:
            return 1000
        else:
            return -1000

    def get_winner(self, board: np.ndarray):
        for i in range(len(board[0])):
            row_unique_elements = np.unique(board[i, :])
            col_unique_elements = np.unique(board[:, i])

            if len(row_unique_elements) == 1 and row_unique_elements.item() != "":
                return row_unique_elements.item()
            if len(col_unique_elements) == 1 and col_unique_elements.item() != "":
                return col_unique_elements.item()

        diagonal_unique_elements = np.unique(np.diagonal(board))
        antidiagonal_unique_elements = np.unique(np.diagonal(np.flipud(board)))
        if len(diagonal_unique_elements) == 1 and diagonal_unique_elements.item() != "":
            return diagonal_unique_elements.item()

        if (
            len(antidiagonal_unique_elements) == 1
            and antidiagonal_unique_elements.item() != ""
        ):
            return antidiagonal_unique_elements.item()

        if np.all(board != ""):
            return "t"

        return ""

    def make_successors_with_moves(self, board, move_max):
        if move_max:
            current_sign = "x" if self.player_sign == "x" else "o"
        else:
            current_sign = "o" if self.player_sign == "x" else "x"
        successors = []

        available_moves = np.argwhere(board == "")

        for move in available_moves:
            new_board = board.copy()
            new_board[tuple(move)] = current_sign
            successors.append((new_board, tuple(move)))

        return successors
