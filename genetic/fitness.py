import numpy
from typing import List, Union

import constants as c


class Fitness:
    """
    Fitness class. Shape of `fitnesses` is (N_chromosome,).
    """

    def __init__(
        self,
        distances: Union[List[float], numpy.ndarray],
        times: Union[List[float], numpy.ndarray],
        collided: Union[List[bool], numpy.ndarray],
    ):
        self.distances = numpy.asarray(distances)
        self.times = numpy.asarray(times)
        self.collided = numpy.asarray(collided)

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

    def time_penalty_on_non_collided_chromosomes(self, penalty_coefficient: float = 0.2):
        """ 
        Only add penalty to non collided chromosomes
        """
        time_penalty = self.times / c.N_frame * penalty_coefficient

        # Only add time penalty to those non-collided chromosomes
        result = self.normalised_distance() - time_penalty * ~self.collided

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
