from tkinter import Canvas, Tk

import numpy as np

N_BOARD_PIXELS = 900
N_ROWS = 3
SYMBOL_SIZE = (N_BOARD_PIXELS / N_ROWS - N_BOARD_PIXELS / 8) / 2
SYMBOL_THICKNESS = 50
X_COLOR = "#EE4035"
O_COLOR = "#0492CF"
GREEN_COLOR = "#7BC043"


def _logical_to_grid(logical_position):
    return (N_BOARD_PIXELS / 3) * logical_position + N_BOARD_PIXELS / 6


class GameGUI:
    def __init__(self, game, player_x, player_o):
        self.window = Tk()
        self.window.title("Tic Tac Toe")
        self.canvas = Canvas(self.window, width=N_BOARD_PIXELS, height=N_BOARD_PIXELS)
        self.canvas.pack()
        self.window.bind("<Button-1>", self.click)
        self.initialize_board()
        self.player_x = player_x
        self.player_o = player_o

        self.ties = 0

        self.game = game
        self.reset_board = False

    def mainloop(self):
        self.window.mainloop()

    def initialize_board(self):
        for i in range(2):
            self.canvas.create_line((i + 1) * N_BOARD_PIXELS / 3, 0, (i + 1) * N_BOARD_PIXELS / 3, N_BOARD_PIXELS)
            self.canvas.create_line(0, (i + 1) * N_BOARD_PIXELS / 3, N_BOARD_PIXELS, (i + 1) * N_BOARD_PIXELS / 3)

    def draw_naught(self, logical_position):
        grid_position = _logical_to_grid(logical_position)
        self.canvas.create_oval(
            grid_position[0] - SYMBOL_SIZE,
            grid_position[1] - SYMBOL_SIZE,
            grid_position[0] + SYMBOL_SIZE,
            grid_position[1] + SYMBOL_SIZE,
            width=SYMBOL_THICKNESS,
            outline=O_COLOR,
        )

    def draw_cross(self, logical_position):
        grid_position = _logical_to_grid(logical_position)
        self.canvas.create_line(
            grid_position[0] - SYMBOL_SIZE,
            grid_position[1] - SYMBOL_SIZE,
            grid_position[0] + SYMBOL_SIZE,
            grid_position[1] + SYMBOL_SIZE,
            width=SYMBOL_THICKNESS,
            fill=X_COLOR,
        )
        self.canvas.create_line(
            grid_position[0] - SYMBOL_SIZE,
            grid_position[1] + SYMBOL_SIZE,
            grid_position[0] + SYMBOL_SIZE,
            grid_position[1] - SYMBOL_SIZE,
            width=SYMBOL_THICKNESS,
            fill=X_COLOR,
        )

    def click(self, grid_position):
        grid_position = np.array([grid_position.x, grid_position.y], dtype=np.int_)
        logical_position = np.array(grid_position // (N_BOARD_PIXELS / N_ROWS), dtype=np.int_)
        if self.reset_board:
            self.play_again()
            return

        logical_position = (
            self.player_x.get_move(logical_position)
            if self.game.player_x_turn
            else self.player_o.get_move(logical_position)
        )

        new_symbol = self.game.move(logical_position)

        if new_symbol == "x":
            self.draw_cross(logical_position)
        elif new_symbol == "o":
            self.draw_naught(logical_position)

        if self.game.get_winner() in ["x", "o", "t"]:
            self.display_gameover()
            self.reset_board = True

    def play_again(self):
        self.reset_board = False
        self.canvas.delete("all")
        self.initialize_board()
        self.game.play_again()

    def display_gameover(self):
        winning_player = self.game.get_winner()
        if winning_player == "x":
            self.player_x.score += 1
            text = "Winner: (X)"
            color = X_COLOR
        elif winning_player == "o":
            self.player_o.score += 1
            text = "Winner: (O)"
            color = O_COLOR
        else:
            self.ties += 1
            text = "Its a tie"
            color = "gray"

        self.canvas.delete("all")
        self.canvas.create_text(N_BOARD_PIXELS / 2, N_BOARD_PIXELS / 3, font="cmr 40 bold", fill=color, text=text)

        score_text = "Scores \n"
        self.canvas.create_text(
            N_BOARD_PIXELS / 2, 5 * N_BOARD_PIXELS / 8, font="cmr 30 bold", fill=GREEN_COLOR, text=score_text
        )

        score_text = "(X)  : " + str(self.player_x.score) + "\n"
        score_text += "(O)  : " + str(self.player_o.score) + "\n"
        score_text += "Tie  : " + str(self.ties)
        self.canvas.create_text(
            N_BOARD_PIXELS / 2, 3 * N_BOARD_PIXELS / 4, font="cmr 20 bold", fill=GREEN_COLOR, text=score_text
        )

        score_text = "Click to play again \n"
        self.canvas.create_text(
            N_BOARD_PIXELS / 2, 15 * N_BOARD_PIXELS / 16, font="cmr 10 bold", fill="gray", text=score_text
        )
