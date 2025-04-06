# Author: Łukasz Szydlik

import json
from typing import List, Tuple
from math import pi
from gradient_descent import GradientFunction1, GradientFunction2


def read_data_from_json(json_file: str) -> List[dict]:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_results_to_txt_function1(results: List[Tuple[float, List[float], int, float]], txt_file: str) -> None:
    with open(txt_file, "w") as f:
        for B, variables_start, iterations, extremum in results:
            f.write(f"{B}, {variables_start[0]}, {iterations}, {extremum}\n")


def write_results_to_txt_function2(results: List[Tuple[float, List[float], int, float]], txt_file: str) -> None:
    with open(txt_file, "w") as f:
        for B, variables_start, iterations, extremum in results:
            f.write(f"{B}, {variables_start[0]}, {variables_start[1]}, {iterations}, {extremum}\n")


def process_gradient_descent_function1(json_file: str, txt_file: str) -> None:
    data = read_data_from_json(json_file)

    results = []

    for example in data:
        B = example["B"]
        variables_start = example["variables_start"]

        variables_min = [-4*pi]
        variables_max = [4*pi]

        function1 = GradientFunction1(variables_min, variables_max, B, variables_start)

        try:
            extremum = function1.grad_descent()[0]
            iterations = function1.grad_descent()[3]
            results.append((B, variables_start, iterations, extremum))
        except Exception as e:
            print(f"Exception for B={B} and variables_start={variables_start}: {str(e)}")

    write_results_to_txt_function1(results, txt_file)


def process_gradient_descent_function2(json_file: str, txt_file: str) -> None:
    data = read_data_from_json(json_file)

    results = []

    for example in data:
        B = example["B"]
        variables_start = example["variables_start"]

        variables_min = [-2, -2]
        variables_max = [2, 2]

        function1 = GradientFunction2(variables_min, variables_max, B, variables_start)

        try:
            extremum = function1.grad_descent()[0]
            iterations = function1.grad_descent()[3]
            results.append((B, variables_start,  iterations, extremum))
        except Exception as e:
            print(f"Exception for B={B} and variables_start={variables_start}: {str(e)}")

    write_results_to_txt_function2(results, txt_file)


if __name__ == "__main__":
    process_gradient_descent_function1("Data/data1.json", "Scores/results1.txt")
    process_gradient_descent_function2("Data/data2.json", "Scores/results2.txt")
