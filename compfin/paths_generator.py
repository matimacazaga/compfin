import numpy as np
from abc import ABC, abstractmethod
from numbers import Number
from typing import Optional


class Path(ABC):

    def __init__(self, seed: Optional[Number] = None):

        self.random_number_generator = np.random.default_rng(seed)

    @abstractmethod
    def generate(
            self,
            number_of_paths: int,
            number_of_steps: int,
            simulation_time: Number,
            drift: float,
            volatility: float,
            initial_value: float):

        pass


class ABM(Path):

    def generate(
            self,
            number_of_paths: int,
            number_of_steps: int,
            simulation_time: Number,
            drift: float,
            volatility: float,
            initial_value: float):

        Z = self.random_number_generator.normal(
            0., 1., size=(number_of_paths, number_of_steps))

        paths = np.zeros(shape=(number_of_paths, number_of_steps+1))

        time = np.linspace(0., simulation_time, num=number_of_steps+1)

        paths[:, 0] = initial_value

        dt = simulation_time / number_of_steps

        for i in range(0, number_of_steps):

            if number_of_paths > 1:

                Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

            paths[:, i+1] = paths[:, i] + \
                (drift - 0.5 * volatility**2) * \
                dt + volatility * Z[:, i] * dt**0.5

        return time, paths


class GBM(ABM):

    def generate(self, number_of_paths: int, number_of_steps: int, simulation_time: Number, drift: float, volatility: float, initial_value: float):

        time, abm_paths = super().generate(number_of_paths, number_of_steps, simulation_time,
                                           drift, volatility, np.log(initial_value))
        return time, np.exp(abm_paths)

    def generate_from_abm(abm_paths: np.array):

        return np.exp(abm_paths)


class MoneySavingsAccount:

    @staticmethod
    def get_discount_rate(drift: float, time: Number):

        return np.exp(-drift*time)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # gbm = GBM(seed=42)
    # abm = ABM(seed=42)

    # simulation_time = 1
    # number_of_steps = 500
    # number_of_paths = 1
    # drift = 0.05
    # volatility = 0.4
    # initial_value = 10

    # time, gbm_path = gbm.generate(
    #     number_of_paths, number_of_steps, simulation_time, drift, volatility, initial_value)

    # _, abm_path = abm.generate(
    #     number_of_paths, number_of_steps, simulation_time, drift, volatility, np.log(initial_value))

    # fig, ax = plt.subplots()
    # ax.plot(time, gbm_path[0, :].ravel(), label="GBM")
    # ax.plot(time, abm_path[0, :].ravel(), label="ABM")
    # ax.plot(time, np.exp(abm_path).ravel(), label="GBM from ABM")
    # ax.legend()
    # plt.show()

    # # Martingale property
    # time, gbm_path = gbm.generate(
    #     100_000, number_of_steps, simulation_time, drift, volatility, initial_value)
    # discount_rate = MoneySavingsAccount.get_discount_rate(
    #     drift, simulation_time)

    # print(np.mean(gbm_path[:, -1]))

    # print(np.mean(gbm_path[:, -1] * discount_rate))

    # Q and P measure

    number_of_paths = 8

    number_of_steps = 1000

    simulation_time = 10

    drift_r = 0.05

    drift_mu = 0.15

    volatility = 0.1

    initial_value = 1.

    gbm = GBM(42)
    time, gbm_paths_q = gbm.generate(
        number_of_paths, number_of_steps, simulation_time, drift_r, volatility, initial_value)

    _, gbm_paths_p = gbm.generate(
        number_of_paths, number_of_steps, simulation_time, drift_mu, volatility, initial_value)

    discount_rate = MoneySavingsAccount.get_discount_rate(drift_r, time)

    fig, axs = plt.subplots(ncols=2)

    axs[0].plot(time, initial_value*np.exp(drift_r * time) *
                discount_rate, color="red", linestyle="--", label="Process under Q-Measure")
    axs[0].plot(time, (gbm_paths_q*discount_rate).T)

    axs[0].legend()

    axs[1].plot(time, initial_value*np.exp(drift_mu*time) *
                discount_rate, color="red", linestyle="--", label="Process under P-Measure")
    axs[1].plot(time, (gbm_paths_p*discount_rate).T)

    axs[1].legend()
    plt.show()
