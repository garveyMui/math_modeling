import numpy as np
import random
import math

# Parameters for Simulated Annealing
INITIAL_TEMPERATURE = 1000.0
COOLING_RATE = 0.999
NUM_ITERATIONS = 1000

# Distance matrix between cities (example, replace with your data)
distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Function to calculate the total distance of a tour
def calculate_tour_distance(tour, distance_matrix):
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities):
        total_distance += distance_matrix[tour[i]][tour[(i + 1) % num_cities]]
    return total_distance

# Function to generate a random initial solution
def generate_random_solution(num_cities):
    return list(range(num_cities))

# Function to generate a neighboring solution by swapping two cities
def generate_neighbor_solution(solution):
    i, j = random.sample(range(len(solution)), 2)
    neighbor_solution = solution.copy()
    neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
    return neighbor_solution

# Simulated Annealing algorithm
def simulated_annealing():
    num_cities = distance_matrix.shape[0]
    current_solution = generate_random_solution(num_cities)
    current_distance = calculate_tour_distance(current_solution, distance_matrix)

    best_solution = current_solution.copy()
    best_distance = current_distance

    temperature = INITIAL_TEMPERATURE

    for iteration in range(NUM_ITERATIONS):
        new_solution = generate_neighbor_solution(current_solution)
        new_distance = calculate_tour_distance(new_solution, distance_matrix)

        # If the new solution is better, always accept it
        if new_distance < current_distance:
            current_solution = new_solution
            current_distance = new_distance

            # Update the best solution if needed
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance
        else:
            # If the new solution is worse, accept it with a certain probability
            acceptance_probability = math.exp((current_distance - new_distance) / temperature)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_distance = new_distance

        # Cool down the temperature
        temperature *= COOLING_RATE

    return best_solution

if __name__ == "__main__":
    best_tour = simulated_annealing()
    print("Best Tour:", best_tour)
    print("Best Distance:", calculate_tour_distance(best_tour, distance_matrix))
