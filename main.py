import numpy
from matplotlib import patches, pyplot
from matplotlib.animation import FuncAnimation

import constants as c
from genetic.crossover import Crossover
from genetic.mutation import Mutation
from genetic.selection import Selection
from physics import Physics
from animation import Animation

pyplot.style.use("seaborn-pastel")

numpy.random.seed(c.random_seed)


def compute_fitness_probability(distances, times):
    distances = numpy.array(distances)

    # Compute fitness probability and find best/worst m chromosomes
    p_fitness = distances / numpy.sum(distances)
    return p_fitness


def plot_distances(fitness, max_g):
    f, ax = pyplot.subplots(figsize=(10, 5))
    ax.plot(range(max_g), fitness)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best distance")
    pyplot.show()


def main(animate=False, save=False):
    animation_class = Animation()
    accelerations = Physics.compute_initial_accelerations()

    best_distances = []
    max_g = c.N_generation
    for g in range(c.N_generation):
        # Calculate for this generation
        positions = Physics.compute_positions(accelerations)

        distances_times = [
            Physics.distance_travelled(positions[m], animation_class.ys)
            for m in range(c.N_chromosome)
        ]
        distances, times = zip(*distances_times)
        max_distance_index = distances.index(max(distances))

        best_distances.append(distances[max_distance_index])
        print(
            f"--- Generation #{g} ---\n"
            + f"Distance travelled: {distances[max_distance_index]:.3f}\n"
            + f"Time: {times[max_distance_index]}"
        )

        # Animation
        if animate == True and any(t > 0 for t in times):
            animation_class.animate(g, times, positions, max_distance_index)

        if max(distances) > c.max_distance:
            print("Best solution reached")
            max_g = g + 1
            break

        if g != c.N_generation - 1:
            p_fitness = compute_fitness_probability(distances, times)

            # Core genetic algorithms
            accelerations = Selection(p_fitness, accelerations).roulette_wheel()
            accelerations = Crossover(accelerations).simple_crossover()
            accelerations = Mutation(accelerations).simple_mutation()

    plot_distances(best_distances, max_g)

    print("Simulation complete!")


if __name__ == "__main__":
    # f, ax = pyplot.subplots()
    # ax.plot(range(N_frame), acceleration)
    # ax.plot(range(N_frame), velocity)
    # ax.plot(range(N_frame), position)
    # pyplot.show()
    main(animate=True)
