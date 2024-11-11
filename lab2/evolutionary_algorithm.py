from typing import Tuple
import numpy as np
import random


class evolutionary_algorithm:
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

    def compute_solution(self) -> Tuple[list, float]:
        """
        Return evolutionary algorithm solution as tuple of best_individual and its value
        """
        # iteration
        t = 0

        # set actual population
        actual_pops = self.starting_pops

        # list of actual_individuals and their values - tuple(individual, value)
        actual_individuals_values = self.compute_individuals_values(actual_pops)

        # find best individual
        best_individual, best_value = self.find_best_individual(actual_individuals_values)

        while t < self.t_max:
            temp_pops = self.selection(actual_individuals_values)
            new_pops = self.crossing_and_mutations(temp_pops)
            new_individuals_values = self.compute_individuals_values(new_pops)
            new_best_individual, new_best_value = self.find_best_individual(new_individuals_values)
            if new_best_value > best_value:
                best_individual = new_best_individual
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

        return individuals_values

    def find_best_individual(self, individuals_values: list[Tuple[list, float]]) -> Tuple[list, float]:
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

        # For findind minimum invert values
        if self.find_minimum:
            individuals_values = [(individual, 1/value) for individual, value in individuals_values]

        sum_of_individuals_values = sum(value[1] for value in individuals_values)

        # probability ranges for individuals
        individuals_probability = []
        range_start = 0.0
        for individual, value in individuals_values:
            probability_value = value/sum_of_individuals_values
            individuals_probability.append((individual, range_start,
                                            range_start + probability_value))
            range_start += probability_value

        # drawing self.pops_amount individuals
        choices = np.random.uniform(0.0, 1.0, self.pops_amount)
        selected_individuals = []
        for choice in choices:
            for individual, start in individuals_probability:
                if start <= choice:
                    selected_individuals.append(individual)
                    break

        return selected_individuals

    def crossing_and_mutations(self, pops: list[list]) -> list[list]:
        new_pops = [[]]
        # Crossing
        individual_lenght = len(pops[0])

        if individual_lenght <= 3:
            raise ValueError("Algorithm is useless for 3 or less elements in individual")

        # Make children for each pair in pops, self.pops_amount % 2 it's for make pair when last don't have
        for i in range(int(self.pops_amount/2) + self.pops_amount % 2):
            # Probability of crossing for pair
            if random.random() < self.p_crossing:
                # Random selected range to crossing
                choices = sorted(np.random.randint(0, individual_lenght-1, size=2))
                while choices[0] == choices[1]:
                    choices = sorted(np.random.randint(0, individual_lenght-1, size=2))

                # Set parents
                parent1 = pops[2*i]
                # Create pair if last haven't got
                if i == int(self.pops_amount/2):
                    parent2 = random.choice(pops[:-1])
                else:
                    parent2 = pops[2*i+1]

                # Slice parents
                start1 = parent1[0:choices[0]]
                selected_part1 = parent1[choices[0]+1:choices[1]]
                end1 = parent1[choices[1]+1:]

                start2 = parent1[0:choices[0]]
                selected_part2 = parent2[choices[0]+1:choices[1]]
                end2 = parent1[choices[1]+1:]

                # Make children
                child1 = start1 + selected_part2 + end1
                child2 = start2 + selected_part1 + end2

                # Make maps for setting unique values
                map1 = {}
                map2 = {}
                for f1, f2 in selected_part1, selected_part2:
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

        # Mutations
        self.swap_mutation(new_pops)

        return new_pops

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

    def generational_succession(actual_individuals_values: list[Tuple[list, float]],
                                new_individuals_values: list[Tuple[list, float]]
                                ) -> list[Tuple[list, float]]:
        """
        Return list of selected individuals with theirs values
        """
        return new_individuals_values
