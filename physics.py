from typing import List, Tuple

import numpy
from scipy.integrate import cumulative_trapezoid

import constants as c


class Physics:
    """
    Physics class
    """

    def compute_initial_accelerations(
        N_chromosome_to_spawn: int = c.N_chromosome,
    ) -> numpy.ndarray:
        """
        Compute initial accelerations of the chromosomes
        """
        accelerations = numpy.random.randint(
            -c.max_acceleration,
            c.max_acceleration + 1,
            (N_chromosome_to_spawn, c.N_frame),
        )
        accelerations = accelerations * c.slowdown_factor
        return accelerations

    def compute_positions(accelerations: numpy.ndarray) -> numpy.ndarray:
        """
        Compute positions from accelerations
        """
        velocities = cumulative_trapezoid(accelerations, initial=0)
        positions = cumulative_trapezoid(velocities, initial=0)
        return positions

    def distance_travelled(
        position: numpy.ndarray, ys: List[float]
    ) -> Tuple[float, int]:
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
