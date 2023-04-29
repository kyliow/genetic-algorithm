import numpy

import constants as c


class Crossover:
    """
    Class to handle crossover related algorithm
    """

    def __init__(self, accelerations: numpy.ndarray):
        self.accelerations = accelerations

    def simple_crossover(self) -> numpy.ndarray:
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
            front = self.accelerations[a].copy()
            rear = self.accelerations[b].copy()
            portion = numpy.random.choice(c.N_frame, size=2, replace=False)
            self.accelerations[a][portion[0] : portion[1]] = rear[
                portion[0] : portion[1]
            ]
            self.accelerations[b][portion[0] : portion[1]] = front[
                portion[0] : portion[1]
            ]

        return self.accelerations
