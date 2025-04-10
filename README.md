<div align="center">

  <!-- Header with logo to the left of the title -->
  <table style="margin: 0 auto;">
    <tr>
      <td style="vertical-align: middle;">
        <img src="docs/resources/logo.svg" alt="Maze Dataset Logo" width="50">
      </td>
      <td style="vertical-align: middle; padding-left: 10px;">
        <h1 style="margin: 0;">maze-dataset</h1>
      </td>
    </tr>
  </table>

<!-- Top prominent badges -->
  <p>
    <a href="https://pypi.org/project/maze-dataset/">
      <img src="https://img.shields.io/pypi/v/maze-dataset" alt="PyPI">
    </a>
    &nbsp;
    <a href="https://understanding-search.github.io/maze-dataset/">
      <img src="https://img.shields.io/badge/docs-available-brightgreen" alt="Docs">
    </a>
    &nbsp;
    <a href="https://understanding-search.github.io/maze-dataset/examples/maze_examples.html">
      <img src="https://img.shields.io/badge/examples-view-blue" alt="Examples">
    </a>
    &nbsp;
    <a href="http://arxiv.org/abs/2309.10498">
        <img src="https://img.shields.io/badge/arXiv-2309.10498-b31b1b.svg" alt="arXiv">
    </a>
  </p>
  

  <!-- Diagram -->
  <p>
    <img src="docs/resources/diagram.svg" alt="Diagram">
  </p>

  <!-- Additional badges -->
  <p>
    <a href="https://pypi.org/project/maze-dataset/">
      <img src="https://img.shields.io/pypi/v/maze-dataset" alt="PyPI">
    </a>
    &nbsp;
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version">
    &nbsp;
    <a href="https://github.com/understanding-search/maze-dataset/actions/workflows/checks.yml">
      <img src="https://github.com/understanding-search/maze-dataset/actions/workflows/checks.yml/badge.svg" alt="Checks">
    </a>
    &nbsp;
    <a href="docs/coverage/coverage.txt">
      <img src="docs/coverage/coverage.svg" alt="Coverage">
    </a>
    &nbsp;
    <img src="https://img.shields.io/github/languages/code-size/understanding-search/maze-dataset" alt="code size, bytes">
    &nbsp;
    <img src="https://img.shields.io/github/commit-activity/t/understanding-search/maze-dataset" alt="GitHub commit activity">
    &nbsp;
    <img src="https://img.shields.io/github/issues-closed/understanding-search/maze-dataset" alt="GitHub closed issues">
    &nbsp;
    <img src="https://img.shields.io/github/issues-pr-closed/understanding-search/maze-dataset" alt="GitHub closed pull requests">
    &nbsp;
    <img src="https://img.shields.io/pypi/dm/maze-dataset" alt="PyPI - Downloads">
    &nbsp;
    <a href="https://juleskreuer.eu/projects/citation-badge">
         <img alt="Citation Badge" src="https://api.juleskreuer.eu/citation-badge.php?doi=10.48550/arxiv.2309.10498">
    </a>
  </p>

</div>



<!-- [![PyPI](https://img.shields.io/pypi/v/maze-dataset)](https://pypi.org/project/maze-dataset/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/maze-dataset)
[![Checks](https://github.com/understanding-search/maze-dataset/actions/workflows/checks.yml/badge.svg)](https://github.com/understanding-search/maze-dataset/actions/workflows/checks.yml)
[![Coverage](docs/coverage/coverage.svg)](docs/coverage/coverage.txt)
![code size, bytes](https://img.shields.io/github/languages/code-size/understanding-search/maze-dataset)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/understanding-search/maze-dataset)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/understanding-search/maze-dataset) -->


# `maze-dataset`

This package provides utilities for generation, filtering, solving, visualizing, and processing of mazes for training or evaluating ML systems. Primarily built for the [maze-transformer interpretability](https://github.com/understanding-search/maze-transformer) project. You can find our paper on it here: http://arxiv.org/abs/2309.10498

This package includes a variety of maze generation algorithms, including randomized depth first search, Wilson's algorithm for uniform spanning trees, and percolation. Datasets can be filtered to select mazes of a certain length or complexity, remove duplicates, and satisfy custom properties. A variety of output formats for visualization and training ML models are provided.

|   |   |   |   |
|---|---|---|---|
| <img src="docs/assets/maze_perc.png" alt="Maze generated via percolation" width="95%"/> | <img src="docs/assets/maze_dfs_constrained.png" alt="Maze generated via constrained randomized depth first search" width="95%"/> | <img src="docs/assets/mazeplot_heatmap.png" alt="Maze with random heatmap" width="95%"/> | <img src="docs/assets/mazeplot_path.png" alt="MazePlot with solution" width="95%"/> |


You can view and search through a wide variety of example mazes here: [`understanding-search.github.io/maze-dataset/examples/maze_examples`](https://understanding-search.github.io/maze-dataset/examples/maze_examples.html)

# Citing

If you use this code in your research, please cite [our paper](http://arxiv.org/abs/2309.10498):

```
@misc{maze-dataset,
    title={A Configurable Library for Generating and Manipulating Maze Datasets}, 
    author={Michael Igorevich Ivanitskiy and Rusheb Shah and Alex F. Spies and Tilman Räuker and Dan Valentine and Can Rager and Lucia Quirke and Chris Mathwin and Guillaume Corlouer and Cecilia Diniz Behn and Samy Wu Fung},
    year={2023},
    eprint={2309.10498},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={http://arxiv.org/abs/2309.10498}
}
```


# Installation
This package is [available on PyPI](https://pypi.org/project/maze-dataset/), and can be installed via
```
pip install maze-dataset
```

# Docs

The full hosted documentation is available at [https://understanding-search.github.io/maze-dataset/](https://understanding-search.github.io/maze-dataset/).

Additionally, our [notebooks](https://understanding-search.github.io/maze-dataset/notebooks) serve as a good starting point for understanding the package.

# Usage

## Creating a dataset

To create a `MazeDataset`, which inherits from `torch.utils.data.Dataset`, you first create a `MazeDatasetConfig`:

```python
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
cfg: MazeDatasetConfig = MazeDatasetConfig(
	name="test", # name is only for you to keep track of things
	grid_n=5, # number of rows/columns in the lattice
	n_mazes=4, # number of mazes to generate
	maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)
```

and then pass this config to the `MazeDataset.from_config` method:

```python
dataset: MazeDataset = MazeDataset.from_config(cfg)
```

This method can search for whether a dataset with matching config hash already exists on your filesystem in the expected location, and load it if so. It can also generate a dataset on the fly if needed.

## Conversions to useful formats

The elements of the dataset are [`SolvedMaze`](maze_dataset/maze/lattice_maze.py) objects:
```python
>>> m = dataset[0]
>>> type(m)
maze_dataset.maze.lattice_maze.SolvedMaze
```

Which can be converted to a variety of formats:
```python
# visual representation as ascii art
m.as_ascii() 
# RGB image, optionally without solution or endpoints, suitable for CNNs
m.as_pixels() 
# text format for autoreregressive transformers
from maze_dataset.tokenization import MazeTokenizerModular, TokenizationMode
m.as_tokens(maze_tokenizer=MazeTokenizerModular(
    tokenization_mode=TokenizationMode.AOTP_UT_rasterized, max_grid_size=100,
))
# advanced visualization with many features
from maze_dataset.plotting import MazePlot
MazePlot(maze).plot()
```

<img src="docs/assets/output_formats.png" alt="textual and visual output formats" width="100%"/>


# Development

we use this [makefile template](https://github.com/mivanit/python-project-makefile-template) with slight modifications for our development workflow.

- clone with `git clone https://github.com/understanding-search/maze-dataset`
- `make dep` to install all dependencies
- `make help` will print all available commands
- `make test` will run basic tests to ensure the package is working
- `make format` will run ruff to format and check the code



