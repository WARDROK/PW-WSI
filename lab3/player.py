from abc import ABC, abstractmethod

import numpy as np


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
    def __init__(self, game, config):
        super().__init__(game)
        self.pruning_depth = config["pruning_depth"]
        self.player_sign = ""
        self.opponent_sign = ""

    def get_move(self, event_position):
        if self.player_sign == "":
            if self.game.player_x_turn:
                self.player_sign = "x"
                self.opponent_sign = "o"
            else:
                self.player_sign = "o"
                self.opponent_sign = "x"
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

        return np.array(best_move)

    def alfa_beta(self, board, depth, move_max, alfa, beta):

        # 1000 - Player win, -1000 - Opponent win
        state_score = self.evaluate_state(board)
        is_terminal_state = np.all(board != "")
        if is_terminal_state or depth == 0 or abs(state_score) == 1000:
            return state_score

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

    def evaluate_state(self, board):
        scores = []

        # Evaluate score for each of the 8 lines (3 rows, 3 columns, 2 diagonals)
        # +1000, +10, +1 for each 3, 2, 1 signs in-a-line for player
        # -1000, -10, -1 for each 3, 2, 1 signs in-a-line for opponent
        scores.append(self.evaluate_line((0, 0), (0, 1), (0, 2), board))  # Row 0
        scores.append(self.evaluate_line((1, 0), (1, 1), (1, 2), board))  # Row 1
        scores.append(self.evaluate_line((2, 0), (2, 1), (2, 2), board))  # Row 2
        scores.append(self.evaluate_line((0, 0), (1, 0), (2, 0), board))  # Column 0
        scores.append(self.evaluate_line((0, 1), (1, 1), (2, 1), board))  # Column 1
        scores.append(self.evaluate_line((0, 2), (1, 2), (2, 2), board))  # Column 2
        scores.append(self.evaluate_line((0, 0), (1, 1), (2, 2), board))  # Diagonal 0
        scores.append(self.evaluate_line((0, 2), (1, 1), (2, 0), board))  # Diagonal 1

        # Player win
        if 100 in scores:
            return 1000
        # Opponent win
        elif -100 in scores:
            return -1000
        else:
            return sum(scores)

    def evaluate_line(self, cell1, cell2, cell3, board):
        score = 0

        # First cell
        if board[cell1] == self.player_sign:
            score = 1
        elif board[cell1] == self.opponent_sign:
            score = -1

        # Second cell
        if board[cell2] == self.player_sign:
            if score == 1:  # Cell 1 is player_sign
                score = 10
            elif score == -1:  # Cell 1 is opponent_sign
                return 0
            else:  # Cell 1 is empty
                score = 1
        elif board[cell2] == self.opponent_sign:
            if score == -1:  # Cell 1 is opponent_sign
                score = -10
            elif score == 1:  # Cell 1 is player_sign
                return 0
            else:  # Cell 1 is empty
                score = -1

        # Third cell
        if board[cell3] == self.player_sign:
            if score > 0:  # Cell 1 and/or Cell 2 are player_sign
                score *= 10
            elif score < 0:  # Cell 1 and/or Cell 2 are opponent_sign
                return 0
            else:  # Cell 1 and Cell 2 are empty
                score = 1
        elif board[cell3] == self.opponent_sign:
            if score < 0:  # Cell 1 and/or Cell 2 are opponent_sign
                score *= 10
            elif score > 0:  # Cell 1 and/or Cell 2 are player_sign
                return 0
            else:  # Cell 1 and Cell 2 are empty
                score = -1

        return score
