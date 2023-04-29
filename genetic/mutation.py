import numpy

import constants as c


class Mutation:
    def __init__(self, accelerations: numpy.ndarray):
        self.accelerations = accelerations

    def simple_mutation(self):
        """
        Simple non-binary mutation
        """
        N_mutation = int(c.N_chromosome * c.N_frame * c.p_mutation)
        mutation_x = numpy.random.randint(c.N_frame, size=N_mutation)
        mutation_y = numpy.random.randint(c.N_chromosome, size=N_mutation)

        for i in range(N_mutation):
            x, y = mutation_x[i], mutation_y[i]
            self.accelerations[y, x] = (
                numpy.random.randint(-c.max_acceleration, c.max_acceleration + 1)
                * c.slowdown_factor
            )

        return self.accelerations
