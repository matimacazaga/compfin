import numpy as np
from abc import ABC, abstractmethod
from numbers import Number
from typing import Optional, Tuple, Union
from enum import Enum
from numba import njit, prange, float64, int32


class Measure(Enum):

    P = 0
    Q = 1


@njit(
    "float64[:, :](int32, int32, float64, int32, float64, float64)",
    parallel=True,
    cache=True,
)
def _generate_jumps(
    number_of_paths: int,
    number_of_steps: int,
    dt: float,
    poisson_rate: int,
    jump_magnitude_mean: float,
    jump_magnitude_volatility: float,
) -> np.ndarray:

    jumps = np.zeros((number_of_paths, number_of_steps))

    for i in prange(jumps.shape[0]):
        for j in range(jumps.shape[1]):
            jumps[i, j] = np.random.poisson(poisson_rate * dt) * np.random.normal(
                jump_magnitude_mean, jump_magnitude_volatility
            )

    return jumps


@njit(cache=True)
def nb_cumsum(array: np.ndarray, exp: bool = False):

    out = np.copy(array)
    for i in range(1, array.shape[1]):
        out[:, i] = out[:, i - 1] + out[:, i]

    if exp:
        out = np.exp(out)

    return out


@njit("float64(int32, float64)", cache=True)
def get_dt(number_of_steps: int, simulation_time: float) -> float:

    return simulation_time / number_of_steps


@njit(
    "Tuple((float64[:], float64[:, :]))(int32, int32, float64, float64, float64, float64)",
    parallel=True,
    cache=True,
)
def arithmetic_brownian_motion(
    number_of_paths: int,
    number_of_steps: int,
    simulation_time: float,
    drift: float,
    std_deviation: float,
    initial_value: float,
) -> Tuple[np.ndarray, np.ndarray]:

    paths = np.zeros((number_of_paths, number_of_steps + 1))

    time = np.linspace(0.0, simulation_time, number_of_steps + 1)

    paths[:, 0] = initial_value

    dt = get_dt(number_of_steps, simulation_time)

    for i in prange(number_of_paths):

        for j in range(number_of_steps):

            paths[i, j + 1] = (
                paths[i, j]
                + drift * dt
                + std_deviation * np.random.normal() * dt ** 0.5
            )

    return time, paths


@njit(
    "Tuple((float64[:], float64[:, :]))(int32, int32, float64, float64, float64, float64)",
    parallel=True,
    cache=True,
)
def geometric_brownian_motion(
    number_of_paths: int,
    number_of_steps: int,
    simulation_time: float,
    drift: float,
    std_deviation: float,
    initial_value: float,
) -> Tuple[np.ndarray, np.ndarray]:

    paths = np.zeros((number_of_paths, number_of_steps + 1))

    time = np.linspace(0.0, simulation_time, number_of_steps + 1)

    paths[:, 0] = initial_value

    dt = get_dt(number_of_steps, simulation_time)

    for i in prange(number_of_paths):

        for j in range(number_of_steps):

            paths[i, j + 1] = paths[i, j] * (
                1.0 + drift * dt + std_deviation * np.random.normal() * dt ** 0.5
            )

    return time, paths


def merton_jump_diffusion():
    pass


# class ArithmeticBrownianMotion(RandomPath):
#     def generate(
#         self,
#         number_of_paths: int,
#         number_of_steps: int,
#         simulation_time: Number,
#         drift: float,
#         volatility: float,
#         initial_value: float,
#     ) -> Tuple[np.ndarray, np.ndarray]:

#         paths = np.zeros(shape=(number_of_paths, number_of_steps + 1))

#         time = np.linspace(0.0, simulation_time, num=number_of_steps + 1)

#         paths[:, 0] = initial_value

#         dt = self._get_dt(number_of_steps, simulation_time)

#         paths[:, 1:] = _generate_abm(
#             number_of_paths, number_of_steps, drift, volatility, dt
#         )

#         paths = nb_cumsum(paths)

#         return time, paths


# class GeometricBrownianMotion(ArithmeticBrownianMotion):
#     def generate(
#         self,
#         number_of_paths: int,
#         number_of_steps: int,
#         simulation_time: Number,
#         risk_free_rate: float,
#         drift: float,
#         volatility: float,
#         initial_value: float,
#         measure: Measure,
#     ) -> Tuple[np.ndarray, np.ndarray]:

#         paths = np.zeros(shape=(number_of_paths, number_of_steps + 1))

#         time = np.linspace(0.0, simulation_time, num=number_of_steps + 1)

#         dt = self._get_dt(number_of_steps, simulation_time)

#         drift_ = drift if measure == Measure.P else risk_free_rate

#         drift_ -= 0.5 * volatility ** 2

#         paths[:, 0] = np.log(initial_value)

#         paths[:, 1:] = _generate_abm(
#             number_of_paths, number_of_steps, drift_, volatility, dt
#         )

#         paths = nb_cumsum(paths, exp=True)

#         return time, paths

#     def generate_from_abm(abm_paths: np.array) -> np.ndarray:

#         return np.exp(abm_paths)


# class MertonModel(ArithmeticBrownianMotion):
#     @staticmethod
#     def _get_expectation_jump_magnitude(
#         jump_magnitude_mean: float, jump_magnitude_volatility: float
#     ) -> float:
#         return np.exp(jump_magnitude_mean + 0.5 * jump_magnitude_volatility ** 2) - 1.0

#     def generate(
#         self,
#         number_of_paths: int,
#         number_of_steps: int,
#         simulation_time: Number,
#         risk_free_rate: float,
#         drift: float,
#         volatility: float,
#         initial_value: float,
#         poisson_rate: int,
#         jump_magnitude_mean: float,
#         jump_magnitude_volatility: float,
#         measure: Measure,
#     ):

#         dt = simulation_time / number_of_steps

#         drift_ = (
#             drift
#             if measure == Measure.P
#             else self.get_q_measure_drift(
#                 risk_free_rate,
#                 volatility,
#                 poisson_rate,
#                 jump_magnitude_mean,
#                 jump_magnitude_volatility,
#             )
#         )

#         paths = np.zeros(shape=(number_of_paths, number_of_steps + 1))

#         time = np.linspace(0.0, simulation_time, num=number_of_steps + 1)

#         paths[:, 0] = np.log(initial_value)

#         paths[:, 1:] = _generate_abm(
#             number_of_paths, number_of_steps, drift_, volatility, dt
#         ) + _generate_poisson(
#             number_of_paths,
#             number_of_steps,
#             dt,
#             jump_magnitude_mean,
#             jump_magnitude_volatility,
#         )

#         paths = nb_cumsum(paths, exp=True)

#         return time, paths

#     def get_q_measure_drift(
#         self,
#         risk_free_rate: float,
#         volatility: float,
#         poisson_rate: float,
#         jump_magnitude_mean: float,
#         jump_magnitude_volatility: float,
#     ):

#         return (
#             risk_free_rate
#             - poisson_rate
#             * self._get_expectation_jump_magnitude(
#                 jump_magnitude_mean, jump_magnitude_volatility
#             )
#             - 0.5 * volatility ** 2
#         )


@njit([float64[:](float64, float64[:]), float64(float64, float64)], cache=True)
def discount_rate(rate: float, time: Union[float, np.ndarray]):

    return np.exp(-rate * time)


# class MoneySavingsAccount:
#     @staticmethod
#     def get_discount_rate(drift: float, time: Number):

#         return np.exp(-drift * time)


# print(np.mean(gbm_path_drift[:, -1] * dr))
# print(np.mean(gbm_path_risk_free[:, -1] * dr))

# import time

# start = time.time()
# paths = geometric_brownian_motion(
#     number_of_paths=1000000,
#     number_of_steps=500,
#     simulation_time=10.0,
#     drift=0.12,
#     std_deviation=0.1,
#     initial_value=100.0,
# )
# end = time.time()

# print(f"Time elapsed: {end-start}")

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

# number_of_paths = 8

# number_of_steps = 1000

# simulation_time = 10

# drift_r = 0.05

# drift_mu = 0.15

# volatility = 0.1

# initial_value = 1.0

# gbm = GeometricBrownianMotion(42)
# time, gbm_paths_q = gbm.generate(
#     number_of_paths,
#     number_of_steps,
#     simulation_time,
#     drift_r,
#     volatility,
#     initial_value,
# )

# _, gbm_paths_p = gbm.generate(
#     number_of_paths,
#     number_of_steps,
#     simulation_time,
#     drift_mu,
#     volatility,
#     initial_value,
# )

# discount_rate = MoneySavingsAccount.get_discount_rate(drift_r, time)

# fig, axs = plt.subplots(ncols=2)

# axs[0].plot(
#     time,
#     initial_value * np.exp(drift_r * time) * discount_rate,
#     color="red",
#     linestyle="--",
#     label="Process under Q-Measure",
# )
# axs[0].plot(time, (gbm_paths_q * discount_rate).T)

# axs[0].legend()

# axs[1].plot(
#     time,
#     initial_value * np.exp(drift_mu * time) * discount_rate,
#     color="red",
#     linestyle="--",
#     label="Process under P-Measure",
# )
# axs[1].plot(time, (gbm_paths_p * discount_rate).T)

# axs[1].legend()
# plt.show()

## Merton Jump Diffusion model

# mjd = MertonModel()

# number_of_paths = 500000
# number_of_steps = 500
# simulation_time = 5
# poisson_rate = 1
# jump_magnitude_mean = 0.0
# jump_magnitude_volatility = 0.2
# volatility = 0.2
# risk_free_rate = 0.05
# initial_value = 100.0
# drift = 0.12

# time, mjd_paths = mjd.generate(
#     number_of_paths,
#     number_of_steps,
#     simulation_time,
#     risk_free_rate,
#     drift,
#     volatility,
#     initial_value,
#     poisson_rate,
#     jump_magnitude_mean,
#     jump_magnitude_volatility,
#     Measure.Q,
# )

# discount_rate = MoneySavingsAccount.get_discount_rate(
#     risk_free_rate, simulation_time
# )

# print(np.mean(mjd_paths[:, -1] * discount_rate))

# fig, ax = plt.subplots()

# ax.plot(time, mjd_paths.T)

# plt.show()

