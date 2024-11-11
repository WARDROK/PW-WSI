from typing import Tuple


class evolutionary_algorithm:
    def __init__(self,
                 evaluation_func: callable,
                 starting_pops: list,
                 pops_amount: int,
                 p_mutation: float,
                 p_crossing: float,
                 t_max: int
                 ) -> None:
        self.evaluation_func = evaluation_func
        self.starting_pops = starting_pops
        self.pops_amount = pops_amount
        self.p_mutation = p_mutation
        self.p_crossing = p_crossing
        self.t_max = t_max

    def compute_solution(self) -> Tuple[list, float]:  # best_solution, value
        # iteration
        t = 0

        # set actual population
        actual_pops = self.starting_pops

        # tab of actual_inviduals - tuple(invidual, value)
        inviduals_values = compute_inviduals_values(actual_pops)

        # find best invidual
        best_invidual, best_value = find_best_invidual(inviduals_values)

        while t < self.t_max:
            temp_pops = selection()
            new_pops = crossing_and_mutations(temp_pops)
            inviduals_values = compute_inviduals_values(actual_pops)
            new_best_invidual, new_best_value = find_best_invidual(inviduals_values)
            if new_best_value > best_value:
                best_invidual = new_best_invidual
                best_value = new_best_value
            self.pops = succession(new_pops)

            t += 1

        return (best_invidual, best_value)
