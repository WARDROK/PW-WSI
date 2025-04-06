# Author: Łukasz Szydlik

import abc
from math import e, sin, cos, exp, log, sqrt
from typing import Tuple, List


class GradientFunction(abc.ABC):
    def __init__(self, variables_min: list, variables_max: list, B: float, variables_start: list,
                 precision=1e-6, B_reductions_ratio=0.5, B_reductions_limit=None,
                 max_iterations=10000) -> None:
        if (len(variables_min) == len(variables_start) and len(variables_min) == len(variables_start) and B > 0):
            self.variables_min = variables_min
            self.variables_max = variables_max
            self.B = B
            self.variables_start = variables_start
            self.precision = precision
            self.B_reductions_ratio = B_reductions_ratio
            self.B_reductions_limit = B_reductions_limit
            self.max_iterations = max_iterations
        else:
            raise ValueError("Entered wrong data. Variable lists are not the same length or B <= 0")

    @abc.abstractmethod
    def function_value(self, variables: list) -> float:
        """Returns function value of variables."""
        pass

    @abc.abstractmethod
    def next_variables(self, variables: list) -> list:
        """
        Returns list of next variables computed by gradient metod.\n
        [x - self.B*(d/dx(function)), ...]
        """
        pass

    def gradient_descent_speed(self, variables_actual: list, variables_next: list) -> float:
        return sqrt(sum((actual - next) ** 2 for actual, next in zip(variables_actual, variables_next)))

    def is_variables_in_domain(self, variables: list) -> bool:
        return (all(v1 > v2 for v1, v2 in zip(variables, self.variables_min))
                and all(v1 < v2 for v1, v2 in zip(variables, self.variables_max)))

    def is_finding_maximum(self, find_maximum, value_next, value_actual) -> bool:
        if (find_maximum):
            return value_next > value_actual
        else:
            return value_next < value_actual

    def grad_descent(self, find_maximum=False) -> List[Tuple[float, List[float], List[float], int]]:
        """
        Algorithm for computing extremum for function.\n
        Returns list of [extremum, list of variables, list of values, amounts of iterations]
        """
        if (self.is_variables_in_domain(self.variables_start)):
            variables_actual = self.variables_start
            variables_list = [variables_actual]
            value_actual = self.function_value(variables_actual)
            values_list = [value_actual]
        else:
            raise ValueError("Entered variables outside the domain of the function")

        B = self.B
        B_reduction_actual = 0
        iterations = 0

        if (find_maximum):
            self.B = -self.B

        extremum_finded = False
        while (not extremum_finded):
            iterations += 1
            if (iterations > self.max_iterations):
                raise TimeoutError("Number of iterations exceeded")

            variables_next = self.next_variables(variables_actual)
            value_next = self.function_value(variables_next)

            if (self.gradient_descent_speed(variables_actual, variables_next) <= self.precision):
                extremum_finded = True

            elif (self.is_variables_in_domain(variables_next)
                    and self.is_finding_maximum(find_maximum, value_next, value_actual)):  # check next value

                variables_list.append(variables_next)
                values_list.append(value_next)
                variables_actual = variables_next
                value_actual = value_next

            elif (self.B_reductions_limit is None or B_reduction_actual < self.B_reductions_limit):
                self.B *= self.B_reductions_ratio
                B_reduction_actual += 1

            elif (B_reduction_actual == self.B_reductions_limit):
                raise Exception("Reached B_reductions_limit")

        extremum = value_actual
        self.B = B
        return [extremum, variables_list, values_list, iterations]


class GradientFunction1(GradientFunction):
    def function_value(self, variables: list) -> float:
        x = variables[0]
        return 6*x + 4*sin(x)

    def next_variables(self, variables: list) -> list:
        x = variables[0]
        return [x - self.B*(6 + 4*cos(x))]


class GradientFunction2(GradientFunction):
    def function_value(self, variables: list) -> float:
        x = variables[0]
        y = variables[1]
        return (4*x*y)/(exp(x**2 + y**2))

    def next_variables(self, variables: list) -> list:
        x = variables[0]
        y = variables[1]
        dx = (4 * y * exp(x**2 + y**2) - 8 * x**2 * y * exp(x**2 + y**2) * log(e))/(exp(2 * (x**2 + y**2)))
        dy = (4 * x * exp(x**2 + y**2) - 8 * y**2 * x * exp(x**2 + y**2) * log(e))/(exp(2 * (x**2 + y**2)))
        return [x - self.B*(dx), y - self.B*(dy)]
