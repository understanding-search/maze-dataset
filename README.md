<!-- [![PyPI](https://img.shields.io/pypi/v/muutils)](https://pypi.org/project/muutils/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/muutils) -->
[![Checks](https://github.com/mivanit/muutils/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/muutils/actions/workflows/checks.yml)
[![Coverage](docs/coverage/coverage.svg)](docs/coverage/coverage.txt)
![code size, bytes](https://img.shields.io/github/languages/code-size/mivanit/muutils)
<!-- ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/mivanit/muutils)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/mivanit/muutils) -->


# Maze Dataset

Generation, solving and processing of various mazes. Primarily build for the [maze-transformer](https://github.com/AISC-understanding-search/maze-transformer) project, which this repo is an offshoot of.


|   |   |   |   |
|---|---|---|---|
| ![Maze generated via percolation](docs/maze_perc.png) |  ![Maze generated via constrained randomized depth first search](docs/maze_dfs_constrained.png)  |  ![Maze with random heatmap](docs/mazeplot_heatmap.png)  |  ![MazePlot with solution](docs/mazeplot_path.png)  |






# Installation
```
pip install git+ssh://git@github.com/aisc-understanding-search/maze-dataset.git
```

You can run all tests via
```bash
make test
```
(this will take a few minutes)


# Usage

Most of the functionality is demonstrated in the ipython notebooks in the `notebooks/` folder.

- `demo_dataset.ipynb` how to easily create a dataset of mazes, utilities for filtering the generates mazes via properties, and basic visualization
- `demo_latticemaze.ipynb` internals of the `LatticeMaze` and `SolvedMaze` objects, and advanced visualization

# Development

## Prerequisites

* Install [Poetry](https://python-poetry.org/docs/#installation)
* Install Python 3.10
    * It's a good idea to use [pyenv](https://github.com/pyenv/pyenv) to manage python versions
    * If using pyenv, you'll need to update your Poetry config for it to use the pyenv Python version: `poetry config virtualenvs.prefer-active-python true`
* Install dev dependencies
    ```
    poetry config virtualenvs.in-project true
    poetry install --with dev
    ```
* Run unit, integration, and notebook tests
    ```
    make test
    ```

* (Optional) If you want to work with the jupyter notebooks in VSCode
  * create a jupyter kernel with `poetry run ipython kernel install --user --name=maze-dataset`
  * Restart VSCode
  * In VSCode, select the python interpreter located in `maze-dataset/.venv/bin` as your juptyer kernel


## Testing & Static analysis

`make help` will print all available commands.

- all tests via `make test`
    - unit tests via `make unit`
    - notebook tests via `make test_notebooks`

- formatter (black, pycln, and isort) via `make format`
    - formatter in check-only mode via `make check-format`
