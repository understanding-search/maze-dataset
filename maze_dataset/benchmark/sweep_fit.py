from pathlib import Path
from typing import Callable

from pysr import PySRRegressor

# other imports after pysr since it has to be before torch?
from jaxtyping import Float
import numpy as np
import sympy as sp

from maze_dataset import MazeDatasetConfig
from maze_dataset.benchmark.config_sweep import (
    SweepResult,
    plot_grouped,
)


def extract_training_data(
    sweep_result: SweepResult,
) -> tuple[Float[np.ndarray, "num_rows 5"], Float[np.ndarray, " num_rows"]]:
    """Extract data (X, y) from a SweepResult.

    # Parameters:
     - `sweep_result : SweepResult`
        The sweep result holding configs and success arrays.

    # Returns:
     - `X : Float[np.ndarray, "num_rows 5"]`
        Stacked [p, grid_n, deadends, endpoints_not_equal, generator_func] for each config & param-value
     - `y : Float[np.ndarray, "num_rows"]`
        The corresponding success rate
    """
    X_list: list[list[float]] = []
    y_list: list[float] = []
    for cfg in sweep_result.configs:
        # success_arr is an array of success rates for param_values
        success_arr = sweep_result.result_values[cfg.to_fname()]
        for i, p in enumerate(sweep_result.param_values):
            # Temporarily override p in the config's array representation:
            arr = cfg._to_ps_array().copy()
            arr[0] = p  # index 0 is 'p'
            X_list.append(arr)
            y_list.append(success_arr[i])

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


DEFAULT_PYSR_KWARGS: dict = dict(
    niterations=50,
    unary_operators=[
        "exp",
        "log",
        "square(x) = x^2",
        "cube(x) = x^3",
        "sigmoid(x) = 1/(1 + exp(-x))",
    ],
    extra_sympy_mappings={
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sigmoid": lambda x: 1 / (1 + sp.exp(-x)),
    },
    binary_operators=["+", "-", "*", "/", "^"],
    # populations=50,
    progress=True,
    model_selection="best",
)


def train_pysr_model(
    data: SweepResult,
    **pysr_kwargs,
) -> PySRRegressor:
    # Convert to arrays
    X, y = extract_training_data(data)

    print(f"training data extracted: {X.shape = }, {y.shape = }")

    # Fit the PySR model
    model: PySRRegressor = PySRRegressor(**{**DEFAULT_PYSR_KWARGS, **pysr_kwargs})
    model.fit(X, y)

    return model


def plot_model(
    data: SweepResult,
    model: PySRRegressor,
    save_dir: Path,
    show: bool = True,
) -> None:
    # save all the equations
    save_dir.mkdir(parents=True, exist_ok=True)
    equations_file: Path = save_dir / "equations.txt"
    equations_file.write_text(repr(model))
    print(f"Equations saved to: {equations_file = }")

    # Create a callable that predicts from MazeDatasetConfig
    predict_fn: Callable = model.get_best()["lambda_format"]
    print(f"Best PySR Equation: {model.get_best()['equation'] = }")
    print(f"{predict_fn =}")

    def predict_config(cfg: MazeDatasetConfig) -> float:
        arr = cfg._to_ps_array()
        result = predict_fn(arr)[0]
        return float(result)  # pass the array as separate args

    plot_grouped(
        data,
        predict_fn=predict_config,
        save_dir=save_dir,
        show=show,
    )


def sweep_fit(
    data_path: Path,
    save_dir: Path,
    **pysr_kwargs,
) -> None:
    # Load the sweep result
    data: SweepResult = SweepResult.read(data_path)
    print(f"loaded data: {data.summary() = }")

    # Train the PySR model
    model: PySRRegressor = train_pysr_model(data, **pysr_kwargs)

    # Plot the model
    plot_model(data, model, save_dir, show=False)


if __name__ == "__main__":
    import argparse

    argparser: argparse.ArgumentParser = argparse.ArgumentParser()
    argparser.add_argument(
        "data_path",
        type=Path,
        help="Path to the sweep result file",
    )
    argparser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("docs/benchmarks/percolation_fractions/fit_plots/"),
        help="Path to save the plots",
    )
    argparser.add_argument(
        "--niterations",
        type=int,
        default=50,
        help="Number of iterations for PySR",
    )
    args: argparse.Namespace = argparser.parse_args()

    sweep_fit(
        args.data_path,
        args.save_dir,
        niterations=args.niterations,
        # add any additional kwargs here if running in CLI
        populations=50,
        # ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running.
        population_size=50,
        # ^ Generations between migrations.
        timeout_in_seconds=60 * 60 * 7,
        # ^ stop after 7 hours have passed.
        maxsize=50,
        # ^ Allow greater complexity.
        weight_randomize=0.01,
        # ^ Randomize the tree much more frequently
        turbo=True,
        # ^ Faster evaluation (experimental)
    )

    PySRRegressor()
