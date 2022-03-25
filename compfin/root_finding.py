from ast import Call
from typing import Callable


def newton_method(function: Callable, derivative: Callable, initial_guess: float = 0.1, verbose: bool = False):

    error = 1e10

    n = 1

    root = initial_guess

    while error > 10e-10:

        g = function(root)
        g_prim = derivative(root)

        root -= g/g_prim

        error = abs(g)

        n += 1

        if verbose:

            print(f"Iteration {n} with error: {error:.2f}")

    return root
