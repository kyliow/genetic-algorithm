import numpy

import constants as c


class Fitness:
    """
    Fitness class. Shape of `fitnesses` is (N_chromosome,).
    """

    def __init__(self, distances, times):
        self.distances = numpy.asarray(distances)
        self.times = numpy.asarray(times)

    def normalised_distance(self) -> numpy.ndarray:
        """
        Use travel distances over maximum distance as fitnesses
        """
        return self.distances / c.max_distance

    def with_time_penalty(self, penalty_coefficient: float = 0.5) -> numpy.ndarray:
        """ 
        Add penalty to chromosome which takes too long
        """
        time_penalty = self.times / c.N_frame * penalty_coefficient
        result = self.normalised_distance() - time_penalty 

        # Make sure there is no negative fitness
        result = numpy.where(result > 0, result, 0)
        return result

    @staticmethod
    def fitness_probabilities(fitnesses: numpy.ndarray) -> numpy.ndarray:
        """
        Compute fitness probability
        """
        p_fitness = numpy.asarray(fitnesses) / numpy.sum(fitnesses)
        return p_fitness
