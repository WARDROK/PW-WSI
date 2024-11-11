from typing import Tuple
import numpy as np


class evolutionary_algorithm:
    def __init__(self,
                 evaluation_func: callable,
                 data,  # data for evaluation function
                 starting_pops: list,
                 pops_amount: int,
                 p_mutation: float,
                 p_crossing: float,
                 t_max: int) -> None:
        self.evaluation_func = evaluation_func
        self.data = data
        self.starting_pops = starting_pops
        self.pops_amount = pops_amount
        self.p_mutation = p_mutation
        self.p_crossing = p_crossing
        self.t_max = t_max

    def compute_solution(self) -> Tuple[list, float]:
        """
        Return evolutionary algorithm solution as tuple of best_invidual and its value
        """
        # iteration
        t = 0

        # set actual population
        actual_pops = self.starting_pops

        # list of actual_individuals and their values - tuple(invidual, value)
        actual_individuals_values = self.compute_individuals_values(actual_pops)

        # find best invidual
        best_invidual, best_value = self.find_best_invidual(actual_individuals_values)

        while t < self.t_max:
            temp_pops = self.selection(actual_individuals_values)
            new_pops = self.crossing_and_mutations(temp_pops)
            new_individuals_values = self.compute_individuals_values(new_pops)
            new_best_invidual, new_best_value = self.find_best_invidual(new_individuals_values)
            if new_best_value > best_value:
                best_invidual = new_best_invidual
                best_value = new_best_value
            actual_individuals_values = self.succession(actual_individuals_values, new_individuals_values)

            t += 1

        return (best_invidual, best_value)

    def compute_individuals_values(self, pops: list[list]) -> list[Tuple[list, float]]:
        """
        Function to compute value for each invidaul\n
        Return tuple (invidual, value)
        """
        individuals_values = []
        for invidual in pops:
            individuals_values.append((invidual, self.evaluation_func(self.data, invidual)))

        return individuals_values

    def find_best_invidual(self, individuals_values: list[Tuple[list, float]]) -> Tuple[list, float]:
        """
        Return tuple of best_individuals and its value
        """
        return max(individuals_values, key=lambda x: x[1])

    def selection(self, individuals_values: list[Tuple[list, float]]) -> list[list]:
        """
        Return list of selected individuals with specific selection variant
        """
        return self.roulette_wheel_selection(individuals_values)

    def roulette_wheel_selection(self, individuals_values: list[Tuple[list, float]]) -> list[list]:
        """
        Return list of selected individuals
        """
        sum_of_individuals_values = sum(value[1] for value in individuals_values)

        # probability ranges for individuals
        individuals_probability = []
        range_start = 0.0
        for invidual, value in individuals_values:
            probability_value = value/sum_of_individuals_values
            individuals_probability.append((invidual, range_start,
                                            range_start + probability_value))
            range_start += probability_value

        # drawing individuals
        choices = np.random.uniform(0.0, 1.0, self.pops_amount)
        selected_individuals = []
        for choice in choices:
            for individual, start in individuals_probability:
                if start <= choice:
                    selected_individuals.append(individual)
                    break

        return selected_individuals

    def crossing_and_mutations(self, pops: list[list]) -> list[list]:
        new_pops = []
        # crossing
        # mutations
        return new_pops

    def succession(self,
                   actual_individuals_values: list[Tuple[list, float]],
                   new_individuals_values: list[Tuple[list, float]]
                   ) -> list[Tuple[list, float]]:
        """
        Return list of selected individuals with theirs values and specific succession variant
        """
        return self.generational_succession(actual_individuals_values, new_individuals_values)

    def generational_succession(actual_individuals_values: list[Tuple[list, float]],
                                new_individuals_values: list[Tuple[list, float]]
                                ) -> list[Tuple[list, float]]:
        """
        Return list of selected individuals with theirs values
        """
        return new_individuals_values
