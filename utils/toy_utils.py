import math
import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import openai

from llm_hpo import LLMOptimizer


def optimize_with_llm(
    toy_func, search_budget, llm_model, cot, temperature, prompt_num, verbose=False
):
    """Optimize a toy function using LLM."""
    llm = LLMOptimizer(model=llm_model, temperature=temperature)
    trajectory = []
    losses = []
    for iteration in range(search_budget):
        print(f"Iteration {iteration} out of {search_budget}")
        if iteration == 0:
            prompt = BlackBoxPrompts.make_inital_prompt(
                toy_func.search_space, prompt_num=prompt_num, budget=search_budget
            )
        else:
            prompt = BlackBoxPrompts.make_feedback_prompt(
                loss, iter=iteration, use_cot=cot, prompt_num=prompt_num
            )
        params = llm.ask(prompt)
        if isinstance(toy_func, QuadraticFunction):
            loss = toy_func(params)
        else:
            loss = toy_func(params[0], params[1])
        if verbose:
            print(f"----\nPrompt: {prompt}\nparams: {params}\n Loss: {loss}")
        trajectory.append(params)
        losses.append(loss)
    messages = llm.get_current_messages()
    return trajectory, losses, messages


# define toy functions
class ToyFunction:
    search_space: dict[str, tuple[float]]
    optima: dict[str, float]

    def __init__(self, search_space=None, optima=None) -> None:
        self.search_space = search_space
        self.optima = optima

    def __call__(self, *args):
        raise NotImplementedError


class QuadraticFunction:
    """Axis-aligned quadratic function"""

    def __init__(self, num_dimensions=5, eigenspectrum="uniform", optima_min=5):
        # Define optimum c within the range [-min_range, min_range]
        self.num_dimensions = num_dimensions
        self.min_range = optima_min  # dist from origin to optima

        if eigenspectrum == "uniform":
            self.eigenvalues = np.ones(self.num_dimensions)
        elif eigenspectrum == "random":
            self.eigenvalues = np.random.uniform(1, num_dimensions, self.num_dimensions)
        else:
            raise NotImplementedError

        self.A = np.diag(self.eigenvalues)
        self.c = np.random.uniform(-self.min_range, self.min_range, self.num_dimensions)
        self.search_space = {
            f"x{i+1}": (-self.min_range, self.min_range)
            for i in range(self.num_dimensions)
        }

    def __call__(self, x):
        x_minus_c = x - self.c
        return np.dot(x_minus_c.T, np.dot(self.A, x_minus_c))

    def set_A(self, A):
        self.A = A

    def set_c(self, c):
        self.c = c

    def evaluate_on_grid(self, X1, X2):
        Y = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                x = np.array([X1[i, j], X2[i, j]])
                Y[i, j] = self(x)
        return Y

    def load_dict(self, d):
        self.num_dimensions = d["num_dimensions"]
        self.eigenvalues = d["eigenvalues"]
        self.c = d["c"]
        self.A = np.diag(self.eigenvalues)

    def get_dict(self):
        # returns dictionary that can be used to reconstruct identical function
        return {
            "num_dimensions": self.num_dimensions,
            "eigenvalues": self.eigenvalues,
            "c": self.c,
        }

    def get_info(self):
        return self.A, self.c

    def __repr__(self) -> str:
        return f"QuadraticFunction(eigenspectrum={self.eigenvalues}, min={self.c}"


class ShiftedToyFunction:
    # shifts everything to mitigate overfitting
    def __init__(self, toy_function_instance, shift_value):
        self.toy_function_instance = toy_function_instance
        self.shift_value = shift_value
        self.search_space = self.toy_function_instance.search_space
        self.optima = self.toy_function_instance.optima
        self.optima["x1"] += self.shift_value[0]
        self.optima["x2"] += self.shift_value[1]
        # check if instance has optimas
        if getattr(self.toy_function_instance, "optimas", None) is not None:
            self.optimas = []
            for optima in self.toy_function_instance.optimas:
                self.optimas.append(
                    {
                        "x1": optima["x1"] + self.shift_value[0],
                        "x2": optima["x2"] + self.shift_value[1],
                    }
                )

    def __call__(self, x1, x2) -> float:
        return self.toy_function_instance(
            x1 - self.shift_value[0], x2 - self.shift_value[1]
        )


class RosenbrockFunction(ToyFunction):
    def __init__(self):
        search_space = {
            "x1": (-5, 10),
            "x2": (-5, 10),  # TODO: check if this is correct
        }
        optima = {
            "x1": 1.0,
            "x2": 1.0,
        }
        super().__init__(search_space=search_space, optima=optima)

    def __call__(self, x1: float, x2: float) -> float:
        # See https://en.wikipedia.org/wiki/Rosenbrock_function.
        # Search space: [-5, 10] x [5, 10]
        return (1 - x1) ** 2.0 + 100 * (x2 - x1**2.0) ** 2.0


class BraninFunction(ToyFunction):
    def __init__(self):
        search_space = {
            "x1": (-5, 10),
            "x2": (0, 15),
        }
        optima = {
            "x1": -math.pi,
            "x2": 12.275,
        }
        optimas = [
            {"x1": -math.pi, "x2": 12.275},
            {"x1": math.pi, "x2": 2.275},
            {"x1": 9.42478, "x2": 2.475},
        ]
        self.optimas = optimas
        super().__init__(search_space=search_space, optima=optima)

    def __call__(self, x1: float, x2: float) -> float:
        # Reference code:
        # https://github.com/automl/SMAC3/blob/main/benchmark/src/models/branin.py.
        # Search space: [-5, 10] x [0, 15].
        pi = math.pi
        a = 1
        b = 5.1 / ((2 * pi) ** 2)
        c = 5 / pi
        r = 6
        s = 10
        t = 1 / (8 * pi)
        return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


class HimmelblauFunction(ToyFunction):
    def __init__(self):
        search_space = {
            "x1": (-5, 5),
            "x2": (-5, 5),
        }
        optima = {
            "x1": 3.0,
            "x2": 2.0,
        }
        optimas = [
            {"x1": 3, "x2": 2},
            {"x1": -2.805118, "x2": 3.131312},
            {"x1": -3.779310, "x2": -3.283186},
            {"x1": 3.584428, "x2": -1.848126},
        ]
        self.optimas = optimas
        super().__init__(search_space=search_space, optima=optima)

    def __call__(self, x1: float, x2: float) -> float:
        # Reference code:
        # https://github.com/automl/SMAC3/blob/main/benchmark/src/models/himmelblau.py.
        # Search space: [-5, 5] x [-5, 5]
        return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


class AckleyFunction(ToyFunction):
    def __init__(self):
        search_space = {
            "x1": (-5, 5),
            "x2": (-5, 5),
        }
        optima = {
            "x1": 0.0,
            "x2": 0.0,
        }
        super().__init__(search_space=search_space, optima=optima)

    def __call__(self, x1: float, x2: float) -> float:
        # Reference code:
        return (
            -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
            - np.exp(0.5 * (np.cos(2 * math.pi * x1) + np.cos(2 * math.pi * x2)))
            + math.e
            + 20
        )


named_toy_functions = {
    "rosenbrock": RosenbrockFunction(),
    "branin": BraninFunction(),
    "himmelblau": HimmelblauFunction(),
    "ackley": AckleyFunction(),
}

function_names = tuple(named_toy_functions.keys())
toy_functions = namedtuple("_toy_funcs", named_toy_functions.keys())(
    **named_toy_functions
)

quadratic_uniform_2d = QuadraticFunction(
    num_dimensions=2, eigenspectrum="uniform", optima_min=5
)
quad_optima_x = -4.15
quad_optima_y = 3.35
quadratic_uniform_2d.set_c(np.array([quad_optima_x, quad_optima_y]))

quadratic_badcondition_2d = QuadraticFunction(
    num_dimensions=2, eigenspectrum="uniform", optima_min=5
)
quadratic_badcondition_2d.set_c(np.array([quad_optima_x, quad_optima_y]))
quadratic_badcondition_2d.set_A(np.array([[1, 0], [0, 10]]))

named_toy_functions.update(
    {
        "quadratic2d": quadratic_uniform_2d,
        "quadratic2d_10": quadratic_badcondition_2d,
    }
)

shift_value = (0.93, 0.59)  # we use these throughout experiments
named_toy_functions.update(
    {
        "shifted_rosenbrock": ShiftedToyFunction(RosenbrockFunction(), shift_value),
        "shifted_branin": ShiftedToyFunction(BraninFunction(), shift_value),
        "shifted_himmelblau": ShiftedToyFunction(HimmelblauFunction(), shift_value),
        "shifted_ackley": ShiftedToyFunction(AckleyFunction(), shift_value),
    }
)


def get_function_name(function: ToyFunction) -> str:
    return repr(function).split("Function")[0]


def get_function(name: str) -> ToyFunction:
    return named_toy_functions[name]


# Prompts for 2D Toy functions
class BlackBoxPrompts:
    def make_inital_prompt(search_space, prompt_num, budget=10):
        if prompt_num == 0:
            prompt = f"""You are optimizing a function with two inputs. x1 must be in range {search_space['x1']}. x2 must be in range {search_space['x2']}. I want you to predict values that minimize the loss of the function; I will tell you the value of the function before you try again. Do not put new lines or extra characters in your response. Format your output with json as follows: {{"x": [x1, x2]}}\n"""
        elif prompt_num == 1:
            prompt = f"""You are optimizing a function with two inputs. x1 must be in range {search_space['x1']}. x2 must be in range {search_space['x2']}. I want you to predict values that minimize the loss of the function; I will tell you the value of the function before you try again. Do not put new lines or extra characters in your response. We have a total of {budget} evaluations. Format your output with json as follows: {{"x": [x1, x2]}}\n"""
        elif prompt_num == 2:
            prompt = f"""You are helping tune hyperparameters to minimize loss. x1 must be in range {search_space['x1']}. x2 must be in range {search_space['x2']}. I want you to predict values that minimize the loss of the function; I will tell you the value of the function before you try again. Do not put new lines or extra characters in your response. We have a total of {budget} evaluations. Format your output with json as follows: {{"x": [x1, x2]}}\n"""
        elif prompt_num == 3:
            prompt = f"""You are helping tune hyperparameters to minimize loss. x1 must be in range {search_space['x1']}. x2 must be in range {search_space['x2']}. The training process is deterministic and yields a nonnegative loss. I want you to predict values that minimize the loss of the function; I will tell you the value of the function before you try again. Do not put new lines or extra characters in your response. We have a total of {budget} evaluations. Format your output with json as follows: {{"x": [x1, x2]}}\n"""
        else:
            raise NotImplementedError
        return prompt

    def make_feedback_prompt(loss, iter=1, use_cot=True, prompt_num=1):
        # prompt_num currently unused
        if iter > 0 and use_cot:
            return f"""Loss: {loss:.3e}. Write two lines as follows:\nAnalysis:(up to a few sentences describing what worked so far and what to choose next)\nOutput:{{json dict}}"""
        else:
            return f"""Loss: {loss:.3e}. Format your next output as before."""


def plot_trajectory(
    func: ToyFunction,
    trajectory: list[tuple[float]],
    log_scale=False,
    plot_type="standard",
    func_name="example",
):
    fig, ax = plt.subplots()
    # ax.set_title(f"Trajectory on {func_name}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if plot_type == "standard":
        x1_lower, x1_upper = func.search_space["x1"]
        x1_lower -= 0.5
        x1_upper += 0.5
        x2_lower, x2_upper = func.search_space["x2"]
        x2_lower -= 0.5
        x2_upper += 0.5
    else:
        x1_lower, x1_upper = -6, 6
        x2_lower, x2_upper = -6, 6
    ax.set_xlim(x1_lower, x1_upper)
    ax.set_ylim(x2_lower, x2_upper)

    # setup log scale if needed
    norm = LogNorm() if log_scale else None

    # Plot the function as a color map
    x1 = np.linspace(x1_lower, x1_upper, 150)
    x2 = np.linspace(x2_lower, x2_upper, 150)
    X1, X2 = np.meshgrid(x1, x2)
    if plot_type == "standard":
        Y = func(X1, X2)
    else:
        Y = func.evaluate_on_grid(X1, X2)
    c = ax.pcolormesh(X1, X2, Y, shading="auto", cmap="inferno", norm=norm)
    fig.colorbar(c, ax=ax)

    # Add contours
    contour = ax.contour(X1, X2, Y, levels=20, colors="salmon", norm=norm)
    ax.clabel(contour, inline=1, fontsize=8)

    # Plot the trajectory with arrows
    x1_vals, x2_vals = zip(*trajectory)
    ax.plot(x1_vals, x2_vals, marker="o", color="royalblue", linewidth=1, markersize=1)
    for i in range(1, len(x1_vals)):
        ax.quiver(
            x1_vals[i - 1],
            x2_vals[i - 1],
            x1_vals[i] - x1_vals[i - 1],
            x2_vals[i] - x2_vals[i - 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="royalblue",
        )

    # Indicate start and end of the trajectory
    ax.plot(x1_vals[0], x2_vals[0], marker="o", color="mediumseagreen", label="Start")
    ax.plot(x1_vals[-1], x2_vals[-1], marker="s", color="dodgerblue", label="End")

    # Mark the optima
    if plot_type == "standard":
        # check if func has optimas
        if getattr(func, "optimas", None) is not None:
            print("multiple optimas")
            # iterate through optimas
            for i, optima in enumerate(func.optimas):
                x1_opt, x2_opt = optima["x1"], optima["x2"]
                if i == 0:
                    ax.plot(
                        x1_opt,
                        x2_opt,
                        marker="*",
                        markersize=12,
                        color="limegreen",
                        label="Optima",
                    )
                else:
                    ax.plot(
                        x1_opt, x2_opt, marker="*", markersize=12, color="limegreen"
                    )
        else:
            print("single optima")
            x1_opt, x2_opt = func.optima["x1"], func.optima["x2"]
            ax.plot(
                x1_opt,
                x2_opt,
                marker="*",
                markersize=12,
                color="limegreen",
                label="Optima",
            )
    else:
        x1_opt, x2_opt = quad_optima_x, quad_optima_y
        ax.plot(
            x1_opt, x2_opt, marker="*", markersize=12, color="limegreen", label="Optima"
        )

    ax.legend()
    plt.show()
    return fig
