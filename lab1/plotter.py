# Author: Łukasz Szydlik

import matplotlib.pyplot as plt
import numpy as np
from math import sin


def plot_function2D(x_start: float, x_end: float, variables: list, values: list, save_path=None) -> None:
    # Function graph
    x_values = np.linspace(x_start, x_end, 1000)
    y_values = [6 * x + 4 * sin(x) for x in x_values]
    plt.plot(x_values, y_values, label='6x + 4sin(x)', color='blue')

    # Dradient values
    gradient_x_values = [var[0] for var in variables]
    plt.scatter(gradient_x_values, values, color='red', label='Gradient descent points')
    plt.scatter(gradient_x_values[-1], values[-1], color='black', label='Extremum')

    plt.title('Function 2D with gradient descent points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, format='png')
    else:
        plt.show()
    plt.close()


def plot_function3D(variables_min: list, variables_max: list, variables: list, values: list, save_path=None) -> None:
    # Przygotowanie do wykresu 3D dla GradientFunction2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Siatka wartości X i Y
    x_values = np.linspace(variables_min[0], variables_max[0], 100)
    y_values = np.linspace(variables_min[1], variables_max[1], 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z = (4 * X * Y) / np.exp(X**2 + Y**2)

    # Tworzenie powierzchni funkcji
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, label="4xy/e^(x^2 + y^2)")

    # Wykres konkretnych punktów uzyskanych z gradientu
    gradient_x_values = [var[0] for var in variables]
    gradient_y_values = [var[1] for var in variables]
    ax.scatter(gradient_x_values, gradient_y_values, values, color='red', label='Gradient descent points')
    ax.scatter(gradient_x_values[-1], gradient_y_values[-1], values[-1], color='black', label='Extremum')

    # Wyświetlanie
    ax.set_title('Function 3D with gradient descent points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.legend()

    if save_path:
        plt.savefig(save_path, format='png')
    else:
        plt.show()
    plt.close()
