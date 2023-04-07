import numpy
from matplotlib import patches, pyplot
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid

import constants as c

pyplot.style.use("seaborn-pastel")

numpy.random.seed(150)

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
    car_radius = c.block_height
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

    # If no collision at all, return last position and max time
    return position[-1], c.N_frame


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


def fitness_calculation(distances, accelerations):
    distances = numpy.array(distances)

    # Compute fitness probability and find best/worst m chromosomes
    p_fitness = distances / numpy.sum(distances)
    p_fitness_sorted_indices = numpy.argsort(p_fitness)
    N_chromosome_to_be_removed = int(c.N_chromosome * c.remove_proportion)
    worse_m_chromosome_indices = p_fitness_sorted_indices[:N_chromosome_to_be_removed]
    best_m_chromosome_indices = p_fitness_sorted_indices[
        c.N_chromosome - N_chromosome_to_be_removed :
    ]

    # Replace worst chromosomes with best chromosomes
    accelerations[worse_m_chromosome_indices] = accelerations[best_m_chromosome_indices]

    return accelerations


def crossover(accelerations):
    """
    Simple non-binary crossover
    """
    N_crossover = int(c.p_crossover * c.N_chromosome)
    if N_crossover % 2 != 0:
        N_crossover -= 1

    random_indices = numpy.random.choice(
        c.N_chromosome, size=N_crossover, replace=False
    )

    # Crossover front and rear chromosome pairs
    for n in range(int(N_crossover / 2)):
        a, b = random_indices[n], random_indices[N_crossover - n - 1]
        front = accelerations[a].copy()
        rear = accelerations[b].copy()
        portion = numpy.random.choice(c.N_frame, size=2, replace=False)
        accelerations[a][portion[0] : portion[1]] = rear[portion[0] : portion[1]]
        accelerations[b][portion[0] : portion[1]] = front[portion[0] : portion[1]]

    return accelerations


def mutation(accelerations):
    """
    Simple non-binary mutation
    """
    N_mutation = int(c.N_chromosome * c.N_frame * c.p_mutation)
    mutation_x = numpy.random.randint(c.N_frame, size=N_mutation)
    mutation_y = numpy.random.randint(c.N_chromosome, size=N_mutation)

    for i in range(N_mutation):
        x, y = mutation_x[i], mutation_y[i]
        accelerations[y, x] = (
            numpy.random.randint(-c.max_acceleration, c.max_acceleration + 1)
            * c.slowdown_factor
        )

    return accelerations


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


def main(animate=True, save=True):
    accelerations = compute_initial_accelerations()
    for g in range(c.N_generation):
        # Calculate for this generation
        positions = compute_positions(accelerations)
        distances_times = [
            distance_travelled(positions[m]) for m in range(c.N_chromosome)
        ]
        distances, times = zip(*distances_times)
        max_distance_index = distances.index(max(distances))

        print(
            f"--- Generation #{g} ---\n"
            + f"Distance travelled: {distances[max_distance_index]:.3f}\n"
            + f"Time: {times[max_distance_index]}"
        )

        # Animation
        if animate == True and any(t > 0 for t in times):
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
            if save == True:
                anim.save(f"car-{g}.gif")
            pyplot.show()

        if max(distances) > c.max_distance:
            print("Best solution reached")
            break

        if g != c.N_generation - 1:
            # Replace worst chromosomes with new chromosomes via sorting
            accelerations = fitness_calculation(distances, accelerations)

            # Crossover and mutation
            accelerations = crossover(accelerations)
            accelerations = mutation(accelerations)

    print("Simulation complete!")


if __name__ == "__main__":
    # f, ax = pyplot.subplots()
    # ax.plot(range(N_frame), acceleration)
    # ax.plot(range(N_frame), velocity)
    # ax.plot(range(N_frame), position)
    # pyplot.show()
    main()
