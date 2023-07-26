import random

# Parameters for the genetic algorithm
TARGET_STRING = "HELLO, GENETIC ALGORITHM!"
POPULATION_SIZE = 200
MUTATION_RATE = 0.005
GENERATIONS = 3000

# Function to calculate the fitness of an individual (string)
def calculate_fitness(individual):
    score = 0
    for i in range(len(individual)):
        if individual[i] == TARGET_STRING[i]:
            score += 1
    return score / len(TARGET_STRING)

# Function to create a random individual (string)
def create_individual():
    return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ, !") for _ in range(len(TARGET_STRING)))

# Function to perform crossover between two parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(TARGET_STRING) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Function to perform mutation on an individual
def mutate(individual):
    mutated_individual = ""
    for char in individual:
        if random.random() < MUTATION_RATE:
            mutated_individual += random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ, !")
        else:
            mutated_individual += char
    return mutated_individual

# Genetic algorithm main function
def genetic_algorithm():
    # Initialize the population with random individuals
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # Calculate the fitness of each individual
        fitness_scores = [calculate_fitness(individual) for individual in population]

        # Check for a perfect match
        if max(fitness_scores) == 1.0:
            break

        # Select parents for the next generation using roulette wheel selection
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        parents = random.choices(population, weights=probabilities, k=POPULATION_SIZE)

        # Create the next generation through crossover and mutation
        new_generation = []
        while len(new_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_generation.append(child)

        population = new_generation

        # Display progress for every 100 generations
        if generation % 100 == 0:
            best_individual = population[fitness_scores.index(max(fitness_scores))]
            print(f"Generation {generation}, Best Match: {best_individual}")

    # Display final result
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    print(f"\nTarget String: {TARGET_STRING}")
    print(f"Best Match: {best_individual}")

if __name__ == "__main__":
    genetic_algorithm()
