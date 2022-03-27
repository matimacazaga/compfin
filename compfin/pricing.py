from enum import Enum
from numbers import Number
from dataclasses import dataclass
import numpy as np
import scipy.stats as st
from abc import ABC, abstractmethod
from root_finding import newton_method

@dataclass
class Option(ABC):

    underlying_value: float
    strike: float
    volatility: float
    time_to_maturity: Number
    risk_free_rate: float

    def bs_d1(self):

        return self._bs_d1(self.volatility)

    def _bs_d1(self, volatility):

        a = 1.0 / (volatility * np.sqrt(self.time_to_maturity))

        b = (
            np.log(self.underlying_value / self.strike)
            + (self.risk_free_rate + 0.5 * volatility ** 2) * self.time_to_maturity
        )

        return a * b

    def bs_d2(self):

        return self._bs_d2(self.volatility)

    def _bs_d2(self, volatility: float):

        return self._bs_d1(volatility) - volatility * np.sqrt(self.time_to_maturity)

    @abstractmethod
    def get_bs_price(self):
        pass

    @abstractmethod
    def _get_bs_price(self, volatility: float):
        pass

    def bs_vega(self):

        return self._bs_vega(self.volatility)

    def _bs_vega(self, volatility: float):

        value = (
            self.strike
            * np.exp(-self.risk_free_rate * self.time_to_maturity)
            * st.norm.pdf(self._bs_d2(volatility))
            * np.sqrt(self.time_to_maturity)
        )

        return value

    def get_bs_implied_volatility(
        self, market_price: float, initial_guess: float, verbose: bool = False
    ):
        def function(x):

            return self._get_bs_price(x) - market_price

        return newton_method(function, self._bs_vega, initial_guess, verbose)


class Call(Option):
    def _get_bs_price(self, volatility: float):

        return st.norm.cdf(
            self._bs_d1(volatility)
        ) * self.underlying_value - st.norm.cdf(
            self._bs_d2(volatility)
        ) * self.strike * np.exp(
            -self.risk_free_rate * self.time_to_maturity
        )

    def get_bs_price(self):
        return self._get_bs_price(self.volatility)


class Put(Option):
    def _get_bs_price(self, volatility: float):

        return (
            st.norm.cdf(-self._bs_d2(volatility))
            * self.strike
            * np.exp(-self.risk_free_rate * self.time_to_maturity)
            - st.norm.cdf(-self._bs_d1(volatility)) * self.underlying_value
        )

    def get_bs_price(self):
        return self._get_bs_price(self.volatility)


if __name__ == "__main__":

    call = Call(100.0, 120.0, 0.1614827288413938, 1, 0.05)

    print(call.get_bs_price())

    print(call.get_bs_implied_volatility(2.0, 0.1, False))
