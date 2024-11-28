import os
from game import TicTacToe
from player import Player


class GameSimulation:
    def __init__(
        self,
        game: TicTacToe,
        player_x: Player,
        player_o: Player,
        simulation_amount: int,
    ):
        self.player_x = player_x
        self.player_o = player_o
        self.ties = 0
        self.game = game
        self.simulation_amount = simulation_amount

    def mainloop(self):
        while self.simulation_amount > 0:
            os.system("cls")

            logical_position = (
                self.player_x.get_move((0, 0))
                if self.game.player_x_turn
                else self.player_o.get_move((0, 0))
            )
            self.game.move(logical_position)

            if self.game.get_winner() in ["x", "o", "t"]:
                self.simulation_amount -= 1
                print("Remaining simulations:", self.simulation_amount)
                self.print_board()
                self.add_point()
                if self.simulation_amount > 0:
                    self.game.play_again()
        print("X: ", self.player_x.score)
        print("O: ", self.player_o.score)
        print("T: ", self.ties)

    def print_board(self):
        for i, row in enumerate(self.game.board):
            row_copy = [cell if cell != "" else " " for cell in row]
            print((" | ").join(row_copy))
            if i < len(self.game.board) - 1:
                print("---------")

    def add_point(self):
        winning_player = self.game.get_winner()
        if winning_player == "x":
            self.player_x.score += 1
        elif winning_player == "o":
            self.player_o.score += 1
        else:
            self.ties += 1
