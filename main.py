import numpy as np

def jellyfish_search(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
    #Initialize the population:
    jellyfish = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([objective_function(position) for position in jellyfish])

    #Calculate start values:
    best_jellyfish = jellyfish[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    beta = 3.0
    gamma = 0.1

    for iteration in range(max_iterations):
        for current in range(population_size):
            #Time control function:
            time_control = abs((1 - iteration / max_iterations) * (2 * np.random.rand() - 1))

            if time_control >= 0.5:
                #Jellyfish follows ocean current:
                micro = np.mean(jellyfish, axis=0)
                ocean_current = best_jellyfish - beta * np.random.rand() * micro
                jellyfish[current] = jellyfish[current] + np.random.rand() * ocean_current
            else:
                if np.random.rand() > 1 - time_control:
                    #Passive jellyfish motion:
                    jellyfish[current] = jellyfish[current] + gamma * np.random.rand() * (upper_boundary - lower_boundary)
                else:
                    #Active jellyfish motion:
                    random_number = np.random.randint(population_size)
                    direction = jellyfish[random_number] - jellyfish[current] if fitness[random_number] >= fitness[current] else jellyfish[current] - jellyfish[random_number]
                    step = np.random.rand() * direction
                    jellyfish[current] += step

            #Boundary control:
            jellyfish[current] = np.clip(jellyfish[current], lower_boundary, upper_boundary)

            #Evaluate new position:
            new_fitness = objective_function(jellyfish[current])

            if new_fitness <= fitness[current]:
                fitness[current] = new_fitness
                if new_fitness <= best_fitness:
                    best_jellyfish = jellyfish[current].copy()
                    best_fitness = new_fitness

        print(f'Iteration {iteration + 1} -> best_fitness = {best_fitness}')

    return best_jellyfish, best_fitness


#Example usage with some sample functions:
def shubert(x):
    sum1 = np.sum([i * np.cos((i + 1) * x[0] + i) for i in range(1, 6)])
    sum2 = np.sum([i * np.cos((i + 1) * x[1] + i) for i in range(1, 6)])
    return sum1 * sum2


def sixhump_camel_back(x):
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3


def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def michalewiczN(x, m=10):
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m))


easom = lambda x: -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))
sphere = lambda x: np.sum(x**2)

dimensions = 2
lower_boundary = -10.0
upper_boundary = 10.0
population_size = 50
max_iterations = 1000

best_solution, best_fitness = jellyfish_search(shubert,
                                               dimensions,
                                               lower_boundary,
                                               upper_boundary,
                                               population_size,
                                               max_iterations)
print(f'Best solution: {best_solution}')
print(f'Best fitness: {best_fitness}')
