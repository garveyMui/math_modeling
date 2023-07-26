import numpy as np

# Parameters for the ant colony algorithm
ALPHA = 1.0  # Pheromone importance
BETA = 3.0   # Heuristic importance
RHO = 0.5    # Evaporation rate
Q = 100      # Pheromone deposit factor
NUM_ANTS = 10
NUM_ITERATIONS = 100
TABU_TENURE = 10  # Tabu list tenure

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

# Function to perform ant colony optimization with a tabu list
def ant_colony_optimization_tabu():
    num_cities = distance_matrix.shape[0]
    pheromone_matrix = np.ones((num_cities, num_cities))
    tabu_list = np.zeros((num_cities, num_cities), dtype=int)

    best_tour = None
    best_distance = float('inf')

    for iteration in range(NUM_ITERATIONS):
        # Initialize ants at random cities
        ants = [np.random.permutation(num_cities) for _ in range(NUM_ANTS)]

        # Perform ant tours
        for ant in ants:
            for _ in range(num_cities - 1):
                current_city = ant[-1]
                next_city = None

                # Calculate probabilities of choosing the next city based on pheromone and distance
                probabilities = []
                for city in range(num_cities):
                    if city not in ant:
                        pheromone = pheromone_matrix[current_city][city]
                        distance = distance_matrix[current_city][city]
                        probabilities.append((pheromone ** ALPHA) * ((1.0 / distance) ** BETA))
                    else:
                        probabilities.append(0.0)

                # Choose the next city based on probabilities
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()

                # Apply tabu list
                for city in range(num_cities):
                    if city in ant:
                        probabilities[city] = 0.0
                    else:
                        tabu_duration = tabu_list[current_city][city]
                        if tabu_duration > 0:
                            probabilities[city] = 0.0
                        else:
                            probabilities[city] *= (1.0 + tabu_duration)

                next_city = np.random.choice(range(num_cities), p=probabilities)

                ant = np.append(ant, next_city)

            # Update tabu list
            for i in range(num_cities):
                j = (i + 1) % num_cities
                if tabu_list[ant[i]][ant[j]] > 0:
                    tabu_list[ant[i]][ant[j]] = TABU_TENURE
                else:
                    tabu_list[ant[i]][ant[j]] -= 1

        # Update pheromone levels on the edges
        delta_pheromone_matrix = np.zeros((num_cities, num_cities))
        for ant in ants:
            distance = calculate_tour_distance(ant, distance_matrix)
            for i in range(num_cities):
                j = (i + 1) % num_cities
                delta_pheromone_matrix[ant[i]][ant[j]] += Q / distance

        pheromone_matrix = (1 - RHO) * pheromone_matrix + delta_pheromone_matrix

        # Update the best tour and distance if needed
        current_best_tour = ants[np.argmin([calculate_tour_distance(ant, distance_matrix) for ant in ants])]
        current_best_distance = calculate_tour_distance(current_best_tour, distance_matrix)
        if current_best_distance < best_distance:
            best_tour = current_best_tour
            best_distance = current_best_distance

        print(f"Iteration {iteration + 1}, Best Distance: {best_distance}")

    return best_tour

if __name__ == "__main__":
    best_tour = ant_colony_optimization_tabu()
    print("Best Tour:", best_tour)
    print("Best Distance:", calculate_tour_distance(best_tour, distance_matrix))
