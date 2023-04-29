import numpy

import constants as c
from physics import Physics


class Selection:
    """
    Class to handle selection related algorithm
    """

    def __init__(self, p_fitness: numpy.ndarray, accelerations: numpy.ndarray):
        self.p_fitness = p_fitness
        self.accelerations = accelerations

    def simple_sort(self) -> numpy.ndarray:
        """
        Selection via simple sorting
        """
        p_fitness_sorted_indices = numpy.argsort(self.p_fitness)
        N_chromosome_to_be_removed = int(c.N_chromosome * c.remove_proportion)
        worse_m_chromosome_indices = p_fitness_sorted_indices[
            :N_chromosome_to_be_removed
        ]
        best_m_chromosome_indices = p_fitness_sorted_indices[
            c.N_chromosome - N_chromosome_to_be_removed :
        ]

        # Replace worst chromosomes with best chromosomes
        self.accelerations[worse_m_chromosome_indices] = self.accelerations[
            best_m_chromosome_indices
        ]

        return self.accelerations

    def roulette_wheel(self) -> numpy.ndarray:
        """
        Selection via roulette wheel
        """
        chromosome_to_keep_indices = numpy.random.choice(
            c.N_chromosome, c.N_chromosome, True, self.p_fitness
        )

        return self.accelerations[chromosome_to_keep_indices]

    def simple_sort_with_new_chromosomes(self) -> numpy.ndarray:
        """
        Selection via simple sorting, then introduce new chromosomes to replace
        bad chromosomes
        """
        p_fitness_sorted_indices = numpy.argsort(self.p_fitness)
        N_chromosome_to_be_removed = int(c.N_chromosome * c.remove_proportion)
        worse_m_chromosome_indices = p_fitness_sorted_indices[
            :N_chromosome_to_be_removed
        ]

        # Replace worst chromosomes with new chromosomes
        self.accelerations[
            worse_m_chromosome_indices
        ] = Physics.compute_initial_accelerations(N_chromosome_to_be_removed)

        return self.accelerations
