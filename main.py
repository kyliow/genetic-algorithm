import numpy
from matplotlib import patches, pyplot
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid

import constants as c
from genetic.crossover import Crossover
from genetic.mutation import Mutation
from genetic.selection import Selection

pyplot.style.use("seaborn-pastel")

numpy.random.seed(c.random_seed)

# Initialise figure
fig, ax = pyplot.subplots(1, 1, figsize=(10, 5))
ax.set_xlim(-c.plot_axis_offset, c.N_rectangle - 1 + c.plot_axis_offset)
ylim = (c.N_rectangle - 1) / 2 + c.plot_axis_offset
ylim = 1
ax.set_ylim(-ylim, ylim)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

# Initialise blocks
ys, blocks = ([] for _ in range(2))
for n in range(c.N_rectangle):
    # Shift from y=0
    if n == 0:
        shift = 0.25 * numpy.pi
    else:
        shift = 2 * numpy.pi * numpy.random.rand()
    # Sine movement
    ys.append(
        numpy.sin(numpy.linspace(0 + shift, 2 * numpy.pi + shift, c.N_frame))
        * (1 - c.block_height / 2)
        - c.block_height / 2
    )
    blocks.append(patches.Rectangle((0, 0), c.block_width, c.block_height, fc="y"))

# Initialise car: car width and height are inverse of blocks
car = patches.Rectangle((0, 0), c.block_height, c.block_width)

# Initialise text
label = ax.text(-0.05, 0, "", transform=ax.transAxes)


def distance_travelled(position):
    """
    Distance from start point. Greater distance is preferred.

    Return distance travelled and time taken
    """
    # If the car goes backward and hit the back wall, return zero
    # distance travelled
    if numpy.any(position <= -c.plot_axis_offset):
        return 0.0, 0

    # For simplicity, assume car and blocks are circles
    car_radius = c.block_height * 1.2
    for n in range(c.N_rectangle):
        within_car_radius_indices = numpy.where(
            (position > n - car_radius) & (position < n + car_radius)
        )[0]
        car_x = position[within_car_radius_indices]
        block_y = ys[n][within_car_radius_indices]
        car_block_distance = numpy.sqrt((car_x - n) ** 2 + block_y**2)
        if numpy.any(car_block_distance <= car_radius):
            collision_index = (
                numpy.where(car_block_distance <= car_radius)[0][0]
                + within_car_radius_indices[0]
            )
            return position[collision_index], collision_index

    # Code reaches here if there is no collision.
    # If the final position is beyond maximum distance of simulation, get the position
    # and time when maximum distance is reached.
    if position[-1] > c.max_distance:
        end_index = numpy.where(position > c.max_distance)[0][0]
        return position[end_index], end_index

    # If the car ends within simulation box
    else:
        return position[-1], c.N_frame - 1


def compute_initial_accelerations():
    """
    Compute N_chromosome number of initial accelerations.
    """
    accelerations = numpy.random.randint(
        -c.max_acceleration, c.max_acceleration + 1, (c.N_chromosome, c.N_frame)
    )
    accelerations = accelerations * c.slowdown_factor
    return accelerations


def compute_positions(accelerations):
    """
    Compute positions from accelerations
    """
    velocities = cumulative_trapezoid(accelerations, initial=0)
    positions = cumulative_trapezoid(velocities, initial=0)
    return positions


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


class Animation:
    def init():
        for block in blocks:
            ax.add_patch(block)
        ax.add_patch(car)
        label.set_text("")
        all_patches = blocks + [car] + [label]
        return all_patches

    def animate(i, position, generation):
        for n, block in enumerate(blocks):
            block.set_xy([n - c.block_width / 2, ys[n][i]])

        car.set_xy([position[i] - c.block_height / 2, 0 - c.block_width / 2])

        text = f"""
            - Generation #{generation} -
            Time    : {i}
            Position: {position[i]:.3f}
        """
        label.set_text(text)
        all_patches = blocks + [car] + [label]
        return all_patches


def main(animate=False, save=False):
    accelerations = compute_initial_accelerations()
    best_distances = []
    max_g = c.N_generation
    for g in range(c.N_generation):
        # Calculate for this generation
        positions = compute_positions(accelerations)
        distances_times = [
            distance_travelled(positions[m]) for m in range(c.N_chromosome)
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
        # if g == c.N_generation - 1:
        #     if animate == True and any(t > 0 for t in times):
        #         anim = FuncAnimation(
        #             fig,
        #             Animation.animate,
        #             init_func=Animation.init,
        #             frames=times[max_distance_index],
        #             interval=20,
        #             blit=True,
        #             fargs=(
        #                 positions[max_distance_index],
        #                 g,
        #             ),
        #         )
        #         if save == True:
        #             anim.save(f"./animation/car-{g}.gif")
        #         pyplot.show()

        if max(distances) > c.max_distance:
            print("Best solution reached")
            max_g = g + 1
            anim = FuncAnimation(
                fig,
                Animation.animate,
                init_func=Animation.init,
                frames=times[max_distance_index],
                interval=20,
                blit=True,
                fargs=(
                    positions[max_distance_index],
                    g,
                ),
            )

            break

        if g != c.N_generation - 1:
            p_fitness = compute_fitness_probability(distances, times)

            # Replace worst chromosomes with new chromosomes via sorting
            accelerations = Selection(p_fitness, accelerations).roulette_wheel()

            # Crossover and mutation
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
    main()
