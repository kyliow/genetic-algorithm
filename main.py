import numpy
from matplotlib import patches, pyplot
from matplotlib.animation import FuncAnimation

import constants as c
from genetic.crossover import Crossover
from genetic.mutation import Mutation
from genetic.selection import Selection
from genetic.fitness import Fitness
from physics import Physics
from animation import Animation

pyplot.style.use("seaborn-pastel")

numpy.random.seed(c.random_seed)


def compute_fitness_probability(distances, times):
    distances = numpy.array(distances)

    # Compute fitness probability and find best/worst m chromosomes
    p_fitness = distances / numpy.sum(distances)
    return p_fitness


def plot_best_fitness_per_generation(fitness, max_g):
    f, ax = pyplot.subplots(figsize=(10, 5))
    ax.plot(range(max_g), fitness)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    pyplot.show()


def main(animate=False, save=False):
    """
    Main function to simulate genetic iteration
    """
    # Initiate animation class
    animation_class = Animation()

    # Compute initial accelerations
    accelerations = Physics.compute_initial_accelerations()

    best_fitness = []
    max_g = c.N_generation
    for g in range(c.N_generation):
        # Calculate the positions for this generation
        positions = Physics.compute_positions(accelerations)

        # Get distance travelled and time taken for each chromosome
        distances_times = [
            Physics.distance_travelled(positions[m], animation_class.ys)
            for m in range(c.N_chromosome)
        ]
        distances, times = zip(*distances_times)

        # Calculate fitness for all chromosomes
        fitness_class = Fitness(distances, times)
        fitnesses = fitness_class.with_time_penalty()

        # Get the best performing chromosome
        best_fitness_index = numpy.argmax(fitnesses)
        best_fitness.append(fitnesses[best_fitness_index])
        print(
            f"----- Generation #{g} -----\n"
            + f"Distance travelled: {distances[best_fitness_index]:.3f}\n"
            + f"Time: {times[best_fitness_index]}\n"
            + f"Fitness score: {fitnesses[best_fitness_index]:.3f}"
        )

        # Animation
        if (
            animate == True
            and any(t > 0 for t in times)
            and (g % 10 == 0 or g == c.N_generation - 1)
        ):
            animation_class.animate(g, times, positions, best_fitness_index, save)

        # Stop simulation if the car reaches the maximum distance
        if max(distances) > c.max_distance:
            print("Maximum distance reached")
            # animation_class.animate(g, times, positions, best_fitness_index, save)
            max_g = g + 1
            break

        # Run core genetic algorithms when last generation has not reached
        if g != c.N_generation - 1:
            p_fitness = fitness_class.fitness_probabilities(fitnesses)
            accelerations = Selection(p_fitness, accelerations).simple_sort()
            accelerations = Crossover(accelerations).simple_crossover()
            accelerations = Mutation(accelerations).simple_mutation()

    plot_best_fitness_per_generation(best_fitness, max_g)

    print("Simulation complete!")


if __name__ == "__main__":
    # f, ax = pyplot.subplots()
    # ax.plot(range(N_frame), acceleration)
    # ax.plot(range(N_frame), velocity)
    # ax.plot(range(N_frame), position)
    # pyplot.show()
    main(animate=True, save=True)
