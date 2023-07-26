import numpy as np

# Parameters for Tabu Search
TABU_TENURE = 10  # Tabu list tenure
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
    i, j = np.random.choice(len(solution), 2, replace=False)
    neighbor_solution = solution.copy()
    neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
    return neighbor_solution

# Tabu Search algorithm
def tabu_search():
    num_cities = distance_matrix.shape[0]
    current_solution = generate_random_solution(num_cities)
    current_distance = calculate_tour_distance(current_solution, distance_matrix)

    best_solution = current_solution.copy()
    best_distance = current_distance

    tabu_list = np.zeros((num_cities, num_cities), dtype=int)

    for iteration in range(NUM_ITERATIONS):
        # Generate neighboring solutions
        neighbors = [generate_neighbor_solution(current_solution) for _ in range(10)]

        # Choose the best non-tabu neighbor
        best_neighbor = None
        best_neighbor_distance = float('inf')

        for neighbor in neighbors:
            neighbor_distance = calculate_tour_distance(neighbor, distance_matrix)
            i, j = np.where(tabu_list > 0)
            if not np.any(np.logical_and(np.equal(i, neighbor[0]), np.equal(j, neighbor[1]))):
                if neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance

        if best_neighbor is None:
            # If all neighbors are tabu, choose the best neighbor
            for neighbor in neighbors:
                neighbor_distance = calculate_tour_distance(neighbor, distance_matrix)
                if neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance

        # Update the current solution
        current_solution = best_neighbor
        current_distance = best_neighbor_distance

        # Update the best solution if needed
        if current_distance < best_distance:
            best_solution = current_solution.copy()
            best_distance = current_distance

        # Update the tabu list
        tabu_list -= 1
        i, j = np.where(tabu_list < 0)
        tabu_list[i, j] = TABU_TENURE

    return best_solution

if __name__ == "__main__":
    best_tour = tabu_search()
    print("Best Tour:", best_tour)
    print("Best Distance:", calculate_tour_distance(best_tour, distance_matrix))
