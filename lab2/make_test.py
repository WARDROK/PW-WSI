import numpy as np
import time

from solution_utils import read_from_json, validate_solution, evaluate_solution
from evolutionary_algorithm import evolutionary_algorithm


def make_tests(data, amount: int):
    data_from_file = read_from_json("data/pops.json")
    pops = data_from_file.get("individuals", [])

    parameters = read_from_json("data/parameters.json")
    crossing_probability = parameters.get('crossing_probability')
    mutation_probability = parameters.get('mutation_probability')
    max_iterations = parameters.get('max_iterations')

    scores = []
    algorithm = evolutionary_algorithm(evaluate_solution, data, pops, len(pops),
                                       crossing_probability, mutation_probability, max_iterations)
    print("Pops:", len(pops),
          "Crossing_p:", crossing_probability,
          "Mutation_p:", mutation_probability,
          "Iterations:", max_iterations)

    total_time = 0
    for x in range(amount):
        loop_start = time.time()
        solution, value = algorithm.compute_solution()
        loop_end = time.time()
        total_time += (loop_end - loop_start)

        # print(solution, value, evaluate_solution(data, solution), algorithm.evaluation_func_counter)
        print(x+1, "/", amount, sep="")
        validate_solution(data, solution)
        scores.append(value)
    print(np.mean(scores))
    print(np.std(scores))
    print("Time:", total_time/amount)

    # m_p = [0.9, 0.6, 0.3, 0.2, 0.1, 0.05, 0.01]
    # for x in m_p:
    #     scores = []
    #     algorithm = evolutionary_algorithm(evaluate_solution, data, pops, len(pops),
    #                                        0.8, x, 100)
    #     for y in range(amount):
    #         solution, value = algorithm.compute_solution()
    #         print(y+1, "/", amount, sep="")
    #         validate_solution(data, solution)
    #         scores.append(value)
    #     print("Mutation:", x, "Crossing:", 0.8)
    #     print(np.mean(scores))
    #     print(np.std(scores))