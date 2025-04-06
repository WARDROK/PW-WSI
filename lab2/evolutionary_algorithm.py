from typing import Tuple
import numpy as np
import random


class evolutionary_algorithm:
    """
    The `evolutionary_algorithm` class implements an evolutionary algorithm
    used for optimizing an objective function.
    It utilizes a population of individuals that are modified over
    several iterations to find either the minimum or maximum of the given function.

    Initialization Parameters:
    - `evaluation_func: callable` — A fitness function used to
    evaluate the quality of each individual in the population.
    - `data` — Additional data required for computations in the evaluation function.
    - `starting_pops: list` — A list of initial populations (individuals)
    from which the evolution process starts.
    - `pops_amount: int` — The target number of individuals
    to maintain in each generation of the algorithm.
    - `p_crossing: float` — The probability of crossover between individuals,
    determining the creation of offspring.
    - `p_mutation: float` — The probability of mutating an individual,
    allowing random changes to explore the solution space.
    - `t_max: int` — The maximum number of iterations (generations)
    over which the algorithm will evolve the population.
    - `find_minimum: bool` — A flag indicating whether the goal is to find the minimum
    (`True`) or maximum (`False`) of the objective function.

    Class Attributes:
    - `self.evaluation_func_counter` — Times of the evaluation function has been called
    during the algorithm's execution.
"""

    def __init__(self,
                 evaluation_func: callable,
                 data,  # data for evaluation function
                 starting_pops: list,
                 pops_amount: int,
                 p_crossing: float,  # probability
                 p_mutation: float,  # probability
                 t_max: int,  # iterations
                 find_minimum=True) -> None:
        self.evaluation_func = evaluation_func
        self.data = data
        self.starting_pops = starting_pops
        self.pops_amount = pops_amount
        self.p_crossing = p_crossing
        self.p_mutation = p_mutation
        self.t_max = t_max
        self.find_minimum = find_minimum
        self.evaluation_func_counter = 0

    def compute_solution(self) -> Tuple[list, float]:
        """
        Return evolutionary algorithm solution as tuple of best_individual and its value
        """
        # iteration
        t = 0
        # amout of evaluation_func iteration
        self.evaluation_func_counter = 0

        # set actual population
        actual_pops = self.starting_pops

        # list of actual_individuals and their values - tuple(individual, value)
        actual_individuals_values = self.compute_individuals_values(actual_pops)

        # find best individual
        best_individual, best_value = self.find_best_individual(actual_individuals_values)
        best_individual = best_individual.copy()  # copy for data protection

        while t < self.t_max:
            temp_pops = self.selection(actual_individuals_values)
            new_pops = self.crossing(temp_pops)
            self.mutation(new_pops)
            new_individuals_values = self.compute_individuals_values(new_pops)
            new_best_individual, new_best_value = self.find_best_individual(new_individuals_values)
            if self.find_minimum:
                if new_best_value < best_value:
                    best_individual = new_best_individual.copy()
                    best_value = new_best_value
            else:
                if new_best_value > best_value:
                    best_individual = new_best_individual.copy()
                    best_value = new_best_value
            actual_individuals_values = self.succession(actual_individuals_values, new_individuals_values)

            t += 1

        return (best_individual, best_value)

    def compute_individuals_values(self, pops: list[list]) -> list[Tuple[list, float]]:
        """
        Function to compute value for each invidaul\n
        Return tuple (individual, value)
        """
        individuals_values = []
        for individual in pops:
            individuals_values.append((individual, self.evaluation_func(self.data, individual)))
            self.evaluation_func_counter += 1

        return individuals_values

    def find_best_individual(self, individuals_values: list[Tuple[list, float]]) -> Tuple[list, float]:
        """
        Return tuple of best_individuals and its value
        """
        if self.find_minimum:
            return min(individuals_values, key=lambda x: x[1])
        else:
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

        # For findind minimum invert values
        if self.find_minimum:
            individuals_values = [(individual, 1/value) for individual, value in individuals_values]

        sum_of_individuals_values = sum(value[1] for value in individuals_values)

        # probability ranges for individuals
        individuals_probability = []
        range_start = 0.0
        for individual, value in individuals_values:
            probability_value = value/sum_of_individuals_values
            individuals_probability.append(tuple([individual, range_start,
                                                  range_start + probability_value]))
            range_start += probability_value

        # drawing self.pops_amount individuals
        choices = np.random.uniform(0.0, 1.0, self.pops_amount)
        selected_individuals = []
        for choice in choices:
            for individual, start, end in individuals_probability:
                if start <= choice < end:
                    selected_individuals.append(individual)
                    break

        return selected_individuals

    def crossing(self, pops: list[list]) -> list[list]:
        new_pops = []
        individual_lenght = len(pops[0])

        if individual_lenght <= 3:
            raise ValueError("Algorithm is useless for 3 or less elements in individual")

        # Make children for each pair in pops, self.pops_amount % 2 it's for make pair when last don't have
        for i in range(int(self.pops_amount/2) + self.pops_amount % 2):
            # Set parents
            parent1 = pops[2*i]
            # Create pair if last haven't got
            if i == int(self.pops_amount/2):
                parent2 = random.choice(pops[:-1])
            else:
                parent2 = pops[2*i+1]

            # Probability of crossing for pair
            if random.random() < self.p_crossing:
                # Random selected range to crossing
                choices = sorted(np.random.randint(1, individual_lenght, size=2))
                while choices[0] == choices[1]:
                    choices = sorted(np.random.randint(1, individual_lenght, size=2))

                # Slice parents
                start1 = parent1[0:choices[0]]
                selected_part1 = parent1[choices[0]:choices[1]]
                end1 = parent1[choices[1]:]

                start2 = parent1[0:choices[0]]
                selected_part2 = parent2[choices[0]:choices[1]]
                end2 = parent1[choices[1]:]

                # Make children
                child1 = start1 + selected_part2 + end1
                child2 = start2 + selected_part1 + end2

                # Make maps for setting unique values
                map1 = {}
                map2 = {}
                for f1, f2 in zip(selected_part1, selected_part2):
                    map1[f2] = f1
                    map2[f1] = f2

                # Set unique values
                offset = len(start1)
                for j in range(len(selected_part1) + len(end1) - 1):  # not checking first and last (consts)
                    while child1[offset + j] in child1[1:offset + j]:
                        child1[offset + j] = map1[child1[offset + j]]
                    while child2[offset + j] in child2[1:offset + j]:
                        child2[offset + j] = map2[child2[offset + j]]

                # For last parent without pair add only one child
                if i == int(self.pops_amount/2):
                    new_pops.append(child1)
                else:
                    new_pops.extend([child1, child2])
            else:
                # Add parent if crosing not occur
                if i == int(self.pops_amount/2):
                    new_pops.append(parent1)
                else:
                    new_pops.extend([parent1, parent2])

        return new_pops

    def mutation(self, pops: list[list]) -> None:
        """
        Mutations individuals in list by specific mutation variant
        """
        self.swap_mutation(pops)

    def swap_mutation(self, pops: list[list]) -> None:
        """
        Mutations individuals in list by swapping two random values.
        """
        for individual in pops:
            if random.random() < self.p_mutation:
                i, j = random.sample(range(1, len(individual)-1), 2)
                individual[i], individual[j] = individual[j], individual[i]

    def insertion_mutation(self, pops: list[list]) -> None:
        """
        Mutations individuals in the list by inserting one value at a random position.
        """
        for individual in pops:
            if random.random() < self.p_mutation:
                i, j = random.sample(range(1, len(individual)-1), 2)
                value = individual.pop(i)
                individual.insert(j, value)

    def inversion_mutation(self, pops: list[list]) -> None:
        """
        Mutations individuals in the list by reverse random selected range.
        """
        for individual in pops:
            if random.random() < self.p_mutation:
                i, j = random.sample(range(1, len(individual)-1), 2)
                individual[i:j+1] = reversed(individual[i:j+1])

    def succession(self,
                   actual_individuals_values: list[Tuple[list, float]],
                   new_individuals_values: list[Tuple[list, float]]
                   ) -> list[Tuple[list, float]]:
        """
        Return list of selected individuals with theirs values and specific succession variant
        """
        return self.generational_succession(actual_individuals_values, new_individuals_values)

    def generational_succession(self,
                                actual_individuals_values: list[Tuple[list, float]],
                                new_individuals_values: list[Tuple[list, float]]
                                ) -> list[Tuple[list, float]]:
        """
        Return list of selected individuals with theirs values
        """
        return new_individuals_values
