from ..models import geometric_brownian_motion, discount_rate
import numpy as np
import unittest


class TestPathsGeneration(unittest.TestCase):
    def test_gbm(self):

        simulation_time = 1.0
        number_of_steps = 2500
        number_of_paths = 500_000
        risk_free_rate = 0.05
        std_deviation = 0.4
        initial_value = 100.0
        _, path_risk_free = geometric_brownian_motion(
            number_of_paths=number_of_paths,
            number_of_steps=number_of_steps,
            simulation_time=simulation_time,
            drift=risk_free_rate,
            diffusion=std_deviation,
            initial_value=initial_value,
        )
        dr = discount_rate(risk_free_rate, simulation_time)
        mean_discounted_end_value = np.mean(path_risk_free[:, -1] * dr)
        diff = np.abs(initial_value - mean_discounted_end_value)

        self.assertAlmostEqual(diff, 0.0, 0)
