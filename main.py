#EVERYTHING IS IN ONE FILE FOR SIMPLICITY

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Type, TextIO, Iterable
from enum import Enum





###############     TEST FUNCTIONS    ###############

# Interface for test functions:
class FunctionInterface(ABC):
    max_iterations = [50, 100, 500, 1000]
    population_sizes = [30, 50, 100]
    trials = 10

    # Abstract members:
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def minimum(self):
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Iterable[int]:
        pass

    @property
    @abstractmethod
    def lower_boundary(self):
        pass

    @property
    @abstractmethod
    def upper_boundary(self):
        pass

    @staticmethod
    @abstractmethod
    def function(x):
        pass


# Functions itself:
class Rastrigin(FunctionInterface):
    name = "Rastrigin"
    minimum = 0.0
    dimensions = [2, 5, 10, 30]
    lower_boundary = -5.12
    upper_boundary = 5.12

    @staticmethod
    def function(x):
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


class Rosenbrock(FunctionInterface):
    name = "Rosenbrock"
    minimum = 0.0
    dimensions = [2, 5, 10, 30]
    lower_boundary = -5.0
    upper_boundary = 5.0

    @staticmethod
    def function(x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Sphere(FunctionInterface):
    name = "Sphere"
    minimum = 0.0
    dimensions = [2, 5, 10, 30, 50]
    lower_boundary = -5.12
    upper_boundary = 5.12

    @staticmethod
    def function(x):
        return np.sum(x ** 2)


class Beale(FunctionInterface):
    name = "Beale"
    minimum = 0.0
    dimensions = [2]
    lower_boundary = -4.5
    upper_boundary = 4.5

    @staticmethod
    def function(x):
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return term1 + term2 + term3


class Bukin(FunctionInterface):
    name = "Bukin"
    minimum = 0.0
    dimensions = [2]
    lower_boundary = -15.0
    upper_boundary = 3.0

    @staticmethod
    def function(x):
        x1, x2 = x[0], x[1]
        x1 = max(min(x1, -5.0), -15.0)
        x2 = max(min(x2, 3.0), -3.0)
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
        term2 = 0.01 * np.abs(x1 + 10)
        return term1 + term2


class Himmelblaus(FunctionInterface):
    name = "Himmelblaus"
    minimum = 0.0
    dimensions = [2]
    lower_boundary = -5.0
    upper_boundary = 5.0

    @staticmethod
    def function(x):
        x1, x2 = x[0], x[1]
        term1 = (x1 ** 2 + x2 - 11) ** 2
        term2 = (x1 + x2 ** 2 - 7) ** 2
        return term1 + term2





###############     BENCHMARK CLASSES     ###############
class ResultType(Enum):
    BEST = 0,
    WORST = 1


class Benchmark:
    # Properties:
    current_benchmark_function: FunctionInterface = None
    current_dimension = None
    current_max_iterations = None
    current_population_size = None
    current_best_solution = None
    current_best_fitness: float = np.inf
    current_worst_solution = None
    current_worst_fitness: float = -np.inf


    # Benchmark methods:
    @staticmethod
    def run(benchmark_functions: List[Type[FunctionInterface]]):
        file_path = 'benchmarkJSO.csv'
        with open(file_path, 'a', encoding='utf-8') as file:
            if os.stat(file_path).st_size != 0:
                file.write('\n\n')
            file.write('Algorithm;'
                       'FunctionName;'
                       'Trials;'
                       'Dimension;'
                       'MaxIterations;'
                       'PopulationSize;'
                       'LowerBoundary;'
                       'UpperBoundary;'
                       'SearchedMinimum;'
                       'BestFitness;'
                       'BestSolution;'
                       'ResultType\n')
            Benchmark.benchmark(benchmark_functions, file)

    @staticmethod
    def benchmark(benchmark_functions: List[Type[FunctionInterface]], file: TextIO):
        for benchmark_function in benchmark_functions:
            Benchmark.current_benchmark_function = benchmark_function
            for dimension in benchmark_function.dimensions:
                Benchmark.current_dimension = dimension
                for max_iterations in benchmark_function.max_iterations:
                    Benchmark.current_max_iterations = max_iterations
                    for population_size in benchmark_function.population_sizes:
                        Benchmark.current_population_size = population_size
                        for _ in range(benchmark_function.trials):
                            trial_solution, trial_fitness = jellyfish_search(benchmark_function.function,
                                                                             dimension,
                                                                             benchmark_function.lower_boundary,
                                                                             benchmark_function.upper_boundary,
                                                                             population_size,
                                                                             max_iterations)

                            if Benchmark.current_best_fitness > trial_fitness:
                                Benchmark.current_best_solution = trial_solution
                                Benchmark.current_best_fitness = trial_fitness

                            if Benchmark.current_worst_fitness < trial_fitness:
                                Benchmark.current_worst_solution = trial_solution
                                Benchmark.current_worst_fitness = trial_fitness

                        Benchmark.write_to_file(file, ResultType.BEST)
                        Benchmark.write_to_file(file, ResultType.WORST)
                        Benchmark.print_on_screen(ResultType.BEST)
                        Benchmark.print_on_screen(ResultType.WORST)

                        Benchmark.current_best_solution = None
                        Benchmark.current_best_fitness = np.inf
                        Benchmark.current_worst_solution = None
                        Benchmark.current_worst_fitness = -np.inf


    # Write to file methods:
    @staticmethod
    def write_to_file(file: TextIO, result_type: ResultType):
        file.write(f'JSO;'
                   f'{Benchmark.current_benchmark_function.name};'
                   f'{Benchmark.current_benchmark_function.trials};'
                   f'{Benchmark.current_dimension};'
                   f'{Benchmark.current_max_iterations};'
                   f'{Benchmark.current_population_size};'
                   f'{Benchmark.current_benchmark_function.lower_boundary};'
                   f'{Benchmark.current_benchmark_function.upper_boundary};'
                   f'{Benchmark.current_benchmark_function.minimum};')

        if result_type == ResultType.BEST:
            Benchmark.write_best_to_file(file)
        else:
            Benchmark.write_worst_to_file(file)

    @staticmethod
    def write_best_to_file(file: TextIO):
        current_best_solution_str = f"({', '.join(map(str, Benchmark.current_best_solution))})"
        file.write(f'{Benchmark.current_best_fitness};'
                   f'{current_best_solution_str};'
                   f'BEST\n')

    @staticmethod
    def write_worst_to_file(file: TextIO):
        current_worst_solution_str = f"({', '.join(map(str, Benchmark.current_worst_solution))})"
        file.write(f'{Benchmark.current_worst_fitness};'
                   f'{current_worst_solution_str};'
                   f'WORST\n')


    # Print on screen methods:
    @staticmethod
    def print_on_screen(result_type: ResultType):
        print(f'Algorithm: JSO\n'
              f'Function: {Benchmark.current_benchmark_function.name}\n'
              f'Trials = {Benchmark.current_benchmark_function.trials}\n'
              f'Dimensions = {Benchmark.current_dimension}\n'
              f'Max iterations = {Benchmark.current_max_iterations}\n'
              f'Population size = {Benchmark.current_population_size}\n'
              f'Domain: From {Benchmark.current_benchmark_function.lower_boundary} to {Benchmark.current_benchmark_function.upper_boundary}\n'
              f'Searched minimum = {Benchmark.current_benchmark_function.minimum}')

        if result_type == ResultType.BEST:
            Benchmark.print_best_on_screen()
        else:
            Benchmark.print_worst_on_screen()

    @staticmethod
    def print_best_on_screen():
        print(f'Best fitness = {Benchmark.current_best_fitness}\n'
              f'Best solution = {Benchmark.current_best_solution}\n'
              f'Result type = BEST\n')

    @staticmethod
    def print_worst_on_screen():
        print(f'Worst fitness = {Benchmark.current_worst_fitness}\n'
              f'Worst solution = {Benchmark.current_worst_solution}\n'
              f'Result type = WORST\n')




###############     JELLYFISH SEARCH IMPLEMENTATION     ###############
def jellyfish_search(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
    # Initialize the population:
    jellyfish = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([objective_function(position) for position in jellyfish])

    # Calculate start values:
    best_jellyfish = jellyfish[np.argmin(fitness)].copy()
    best_fitness = np.min(fitness)

    beta = 3.0
    gamma = 0.1

    for iteration in range(max_iterations):
        for current in range(population_size):
            # Time control function:
            time_control = abs((1 - iteration / max_iterations) * (2 * np.random.rand() - 1))

            if time_control >= 0.5:
                # Jellyfish follows ocean current:
                micro = np.mean(jellyfish, axis=0)
                ocean_current = best_jellyfish - beta * np.random.rand() * micro
                jellyfish[current] = jellyfish[current] + np.random.rand() * ocean_current
            else:
                if np.random.rand() > 1 - time_control:
                    # Passive jellyfish motion:
                    jellyfish[current] = jellyfish[current] + gamma * np.random.rand() * (
                            upper_boundary - lower_boundary)
                else:
                    # Active jellyfish motion:
                    random_number = np.random.randint(population_size)
                    direction = jellyfish[random_number] - jellyfish[current] if fitness[current] >= fitness[random_number] else jellyfish[current] - jellyfish[random_number]
                    step = np.random.rand() * direction
                    jellyfish[current] += step

            # Boundary control:
            jellyfish[current] = np.clip(jellyfish[current], lower_boundary, upper_boundary)

            # Evaluate new position:
            new_fitness = objective_function(jellyfish[current])

            if new_fitness <= fitness[current]:
                fitness[current] = new_fitness
                if new_fitness <= best_fitness:
                    best_jellyfish = jellyfish[current].copy()
                    best_fitness = new_fitness

        # print(f'Iteration {iteration + 1} -> best_fitness = {best_fitness}')

    return best_jellyfish, best_fitness





###############     MAIN     ###############
if __name__ == '__main__':
    benchmark_functions: List[Type[FunctionInterface]] = [Rastrigin, Rosenbrock, Sphere, Beale, Bukin, Himmelblaus]

    Benchmark.run(benchmark_functions)
