import argparse
import pathlib

import numpy as np
import pandas as pd
from solution_utils import (decode_solution, validate_solution,
                            evaluate_solution, read_from_json,
                            make_pops_and_save_as_json)
from evolutionary_algorithm import evolutionary_algorithm
# from make_test import make_tests

MINI_CITIES_NUM = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities-path", type=pathlib.Path, required=True, help="Path to cities csv file")
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="full",
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


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)

    if args.make_new_pops:
        make_pops_and_save_as_json(data, 100, "data/pops.json")

    data_from_file = read_from_json("data/pops.json")
    pops = data_from_file.get("individuals", [])

    parameters = read_from_json("data/parameters.json")
    crossing_probability = parameters.get('crossing_probability')
    mutation_probability = parameters.get('mutation_probability')
    max_iterations = parameters.get('max_iterations')

    algorithm = evolutionary_algorithm(evaluate_solution, data, pops, len(pops),
                                       crossing_probability, mutation_probability, max_iterations)
    solution, value = algorithm.compute_solution()
    validate_solution(data, solution)
    print("Length:", round(value, 2), "km")
    print(decode_solution(data, solution))

    # make_tests(data, 10)


if __name__ == "__main__":
    main()
