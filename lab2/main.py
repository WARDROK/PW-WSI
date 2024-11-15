import argparse
import pathlib
import random
import json

import numpy as np
import pandas as pd
from solution_utils import generate_solution, decode_solution, validate_solution, evaluate_solution
from evolutionary_algorithm import evolutionary_algorithm

MINI_CITIES_NUM = 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities-path", type=pathlib.Path, required=True, help="Path to cities csv file")
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="mini",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Ciechanów")
    parser.add_argument("--finish", type=str, default="Olsztyn")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--make-new-pops", type=bool, default=False)
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[~((data.index == args.finish) | (data.index == args.start))]
    if args.problem_size == "mini":
        city_names = (
            [args.start] + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist() + [args.finish]
        )
    else:
        city_names = [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]

    return data[city_names].loc[city_names]


def make_pops_and_save_as_json(individual: list, number: int, file_path: str) -> None:
    '''
    Funtction fo make population from one individual.\n
    Save list of individuals in .json file
    '''
    pops = []
    for x in range(number):
        shuffled_range = random.sample(individual[1:-1], len(individual)-2)
        pops.append([0] + shuffled_range + [individual[-1]])

    with open(file_path, 'w') as file:
        json.dump({"individuals": pops}, file)


def read_from_json(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            parameters = json.load(file)
        return parameters
    except FileNotFoundError:
        print(f"Plik '{file_path}' nie został znaleziony.")
        return None
    except json.JSONDecodeError:
        print(f"Plik '{file_path}' zawiera błędny format JSON.")
        return None


def make_tests(solution: list, data, amount: int):
    data_from_file = read_from_json("data/pops.json")
    pops = data_from_file.get("individuals", [])

    parameters = read_from_json("data/parameters.json")
    crossing_probability = parameters.get('crossing_probability')
    mutation_probability = parameters.get('mutation_probability')
    max_iterations = parameters.get('max_iterations')

    print(solution, evaluate_solution(data, solution))
    scores = []
    algorithm = evolutionary_algorithm(evaluate_solution, data, pops, len(pops),
                                       crossing_probability, mutation_probability, max_iterations)
    for x in range(amount):
        solution, value = algorithm.compute_solution()
        print(solution, value, evaluate_solution(data, solution), algorithm.evaluation_func_counter)
        validate_solution(data, solution)
        scores.append(value)
    print(np.mean(scores))
    print(np.std(scores))


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)

    if args.make_new_pops:
        solution = generate_solution(data)
        make_pops_and_save_as_json(solution, 100, "data/pops.json")

    data_from_file = read_from_json("data/pops.json")
    pops = data_from_file.get("individuals", [])

    parameters = read_from_json("data/parameters.json")
    crossing_probability = parameters.get('crossing_probability')
    mutation_probability = parameters.get('mutation_probability')
    max_iterations = parameters.get('max_iterations')

    algorithm = evolutionary_algorithm(evaluate_solution, data, pops, len(pops),
                                       crossing_probability, mutation_probability, max_iterations)
    solution, value = algorithm.compute_solution()
    print(solution, value)
    print(decode_solution(data, solution))


if __name__ == "__main__":
    main()
