# Author: Łukasz Szydlik

from plotter import plot_function2D, plot_function3D
from gradient_descent import GradientFunction1, GradientFunction2
from math import pi


if (__name__ == "__main__"):
    variables_min = [-4*pi]
    variables_max = [4*pi]
    B = 0.5
    variables_start = [4]

    function1 = GradientFunction1(variables_min, variables_max, B, variables_start)
    extremum, variables, values, iterations = function1.grad_descent()

    plot_function2D(variables_min[0], variables_max[0], variables, values, "Scores/Function1.png")
    print("Function1 extremum:", extremum)

    variables_min = [-2, -2]
    variables_max = [2, 2]
    B = 0.1
    variables_start = [0.2, 0.7]

    function2 = GradientFunction2(variables_min, variables_max, B, variables_start)
    extremum, variables, values, iterations = function2.grad_descent()

    plot_function3D(variables_min, variables_max, variables, values, "Scores/Function2.png")
    print("Function2 extremum:", extremum)
