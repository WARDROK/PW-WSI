import numpy as np
import json


def decode_solution(cities_matrix, solution):
    return list(map(lambda city_id: cities_matrix.index[city_id], solution))


def validate_solution(cities_matrix, solution):
    # check if each city is visited exactly one time
    assert len(list(solution)) == len(set(solution))
    assert sorted(solution) == list(range(len(cities_matrix)))
    # check if start and finish cities are in the correct place
    assert solution[0] == 0 and solution[-1] == len(cities_matrix) - 1


def evaluate_solution(cities_matrix, solution):
    total_distance = 0
    for city_id in range(len(solution) - 1):
        total_distance += cities_matrix.iloc[solution[city_id], solution[city_id + 1]]
    return total_distance


def generate_solution(cities_matrix):
    return [0] + np.random.permutation(np.arange(1, len(cities_matrix) - 1)).tolist() + [len(cities_matrix) - 1]


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


def make_pops_and_save_as_json(data, number: int, file_path: str) -> None:
    '''
    Funtction fo make population from one individual.\n
    Save list of individuals in .json file
    '''
    pops = []
    for x in range(number):
        pops.append(generate_solution(data))

    with open(file_path, 'w') as file:
        json.dump({"individuals": pops}, file)
