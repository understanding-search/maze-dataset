#!/usr/bin/env python3
"""
Generate HTML examples of various maze configurations.

This script generates a variety of maze examples with different configurations,
saves them as SVG files, and creates an HTML page to display them with searchable
configuration details.
"""

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import jinja2

from maze_dataset import LatticeMaze, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot

# Define the examples directory
EXAMPLES_DIR = pathlib.Path("docs/examples")
SVG_DIR = EXAMPLES_DIR / "svg"
HTML_PATH = EXAMPLES_DIR / "maze_examples.html"
TEMPLATE_DIR = EXAMPLES_DIR
TEMPLATE_FILE = "maze_examples.html.jinja2"

def generate_maze_configs() -> List[Dict[str, Any]]:
    """Generate a list of diverse maze configurations.
    
    # Returns:
     - `List[Dict[str, Any]]`
        List of maze configuration dictionaries
    """
    configs = []
    
    # Basic maze configurations with different algorithms
    for grid_n in [5, 8, 12]:
        # DFS with different options
        configs.append({
            "name": f"dfs_basic_g{grid_n}",
            "grid_n": grid_n,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {},
            "description": f"Basic DFS maze ({grid_n}x{grid_n})",
            "tags": ["dfs", "basic"]
        })
        
        configs.append({
            "name": f"dfs_no_forks_g{grid_n}",
            "grid_n": grid_n,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {"do_forks": False},
            "description": f"DFS without forks ({grid_n}x{grid_n})",
            "tags": ["dfs", "no_forks", "simple_path"]
        })
        
        # Wilson's algorithm
        configs.append({
            "name": f"wilson_g{grid_n}",
            "grid_n": grid_n,
            "maze_ctor": LatticeMazeGenerators.gen_wilson,
            "maze_ctor_kwargs": {},
            "description": f"Wilson's algorithm ({grid_n}x{grid_n}) - unbiased random maze",
            "tags": ["wilson", "uniform_random"]
        })
        
        # Percolation with different probabilities
        for p in [0.3, 0.5, 0.7]:
            configs.append({
                "name": f"percolation_p{int(p*10)}_g{grid_n}",
                "grid_n": grid_n,
                "maze_ctor": LatticeMazeGenerators.gen_percolation,
                "maze_ctor_kwargs": {"p": p},
                "description": f"Pure percolation (p={p}) ({grid_n}x{grid_n})",
                "tags": ["percolation", f"p={p}"]
            })
            
            configs.append({
                "name": f"dfs_percolation_p{int(p*10)}_g{grid_n}",
                "grid_n": grid_n,
                "maze_ctor": LatticeMazeGenerators.gen_dfs_percolation,
                "maze_ctor_kwargs": {"p": p},
                "description": f"DFS with percolation (p={p}) ({grid_n}x{grid_n})",
                "tags": ["dfs", "percolation", f"p={p}"]
            })
    
    # Additional specialized configurations
    configs.extend([
        {
            "name": "dfs_accessible_cells",
            "grid_n": 10,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {"accessible_cells": 50},
            "description": "DFS with limited accessible cells (50)",
            "tags": ["dfs", "limited_cells"]
        },
        {
            "name": "dfs_accessible_cells_ratio",
            "grid_n": 10,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {"accessible_cells": 0.6},
            "description": "DFS with 60% accessible cells",
            "tags": ["dfs", "limited_cells", "ratio"]
        },
        {
            "name": "dfs_max_tree_depth",
            "grid_n": 10,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {"max_tree_depth": 10},
            "description": "DFS with max tree depth of 10",
            "tags": ["dfs", "limited_depth"]
        },
        {
            "name": "dfs_max_tree_depth_ratio",
            "grid_n": 10,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {"max_tree_depth": 0.3},
            "description": "DFS with max tree depth 30% of grid size",
            "tags": ["dfs", "limited_depth", "ratio"]
        },
        {
            "name": "dfs_start_coord",
            "grid_n": 10,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {"start_coord": (5, 5)},
            "description": "DFS starting from center of grid",
            "tags": ["dfs", "custom_start"]
        },
        {
            "name": "dfs_combined_constraints",
            "grid_n": 15,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {
                "accessible_cells": 100,
                "max_tree_depth": 25,
                "start_coord": (7, 7)
            },
            "description": "DFS with multiple constraints",
            "tags": ["dfs", "combined_constraints"]
        }
    ])
    
    # Add endpoint options for some configurations
    for deadend_start, deadend_end in [(True, False), (False, True), (True, True)]:
        configs.append({
            "name": f"dfs_deadend_start{deadend_start}_end{deadend_end}",
            "grid_n": 8,
            "maze_ctor": LatticeMazeGenerators.gen_dfs,
            "maze_ctor_kwargs": {},
            "endpoint_kwargs": {
                "deadend_start": deadend_start,
                "deadend_end": deadend_end,
                "endpoints_not_equal": True
            },
            "description": f"DFS with {'deadend start' if deadend_start else ''}{' and ' if deadend_start and deadend_end else ''}{'deadend end' if deadend_end else ''}",
            "tags": ["dfs", "deadend_endpoints"]
        })
    
    # Add some percolation examples with deadend endpoints
    configs.append({
        "name": "dfs_percolation_deadends",
        "grid_n": 8,
        "maze_ctor": LatticeMazeGenerators.gen_dfs_percolation,
        "maze_ctor_kwargs": {"p": 0.3},
        "endpoint_kwargs": {
            "deadend_start": True,
            "deadend_end": True,
            "endpoints_not_equal": True,
            "except_on_no_valid_endpoint": False
        },
        "description": "DFS percolation with deadend endpoints",
        "tags": ["dfs", "percolation", "deadend_endpoints"]
    })
    
    return configs

def generate_maze_svg(config_src: Dict[str, Any], seed: Optional[int] = None) -> Tuple[str, MazeDatasetConfig]:
    """Generate a maze from the given configuration and save it as an SVG file.
    
    # Parameters:
     - `config : Dict[str, Any]`
        Maze configuration dictionary
     - `seed : Optional[int]`
        Random seed for reproducibility
        (defaults to `None`)
    
    # Returns:
     - `Tuple[str, str, str]`
        Path to the SVG file, the JSON string of the configuration, and the serialized config
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Extract maze config parameters
    name = config_src["name"]
    grid_n = config_src["grid_n"]
    maze_ctor = config_src["maze_ctor"]
    maze_ctor_kwargs = config_src["maze_ctor_kwargs"]
    endpoint_kwargs = config_src.get("endpoint_kwargs", {})
    
    # Create a MazeDatasetConfig
    maze_config = MazeDatasetConfig(
        name=name,
        grid_n=grid_n,
        n_mazes=1,
        maze_ctor=maze_ctor,
        maze_ctor_kwargs=maze_ctor_kwargs,
        endpoint_kwargs=endpoint_kwargs
    )
       
    # Generate the maze directly
    maze: LatticeMaze = maze_ctor(
        grid_shape=np.array([grid_n, grid_n]),
        **maze_ctor_kwargs
    )
    
    # Generate a solution if endpoint_kwargs are provided
    if endpoint_kwargs:
        solution = maze.generate_random_path(**endpoint_kwargs)
        if solution is not None:
            from maze_dataset import SolvedMaze
            maze = SolvedMaze.from_lattice_maze(maze, solution)
    
    # Create the plot
    maze_plot = MazePlot(maze)
    fig: Figure = maze_plot.plot().fig
    
    # Save as SVG
    svg_path = SVG_DIR / f"{name}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    
    # Prepare JSON configuration
    metadata_json = {k: v for k, v in config_src.items() if k not in ["maze_ctor"]}
    metadata_json["maze_ctor"] = config_src["maze_ctor"].__name__
    metadata_json["config"] = maze_config.serialize()
    
    # Create a corresponding config file
    config_path = SVG_DIR / f"{name}.json"
    with open(config_path, 'w') as f:
        f.write(json.dumps(metadata_json, indent=2))
    
    return svg_path.relative_to(EXAMPLES_DIR).as_posix(), metadata_json

def setup_jinja_env() -> jinja2.Environment:
    """Set up the Jinja2 environment.
    
    # Returns:
     - `jinja2.Environment`
        Configured Jinja2 environment
    """
    # Ensure the template directory exists
    TEMPLATE_DIR.mkdir(exist_ok=True)
    
    # Set up Jinja2 environment
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )

def generate_html_from_template(
    env: jinja2.Environment, 
    maze_examples: List[Dict[str, Any]], 
    all_tags: Set[str]
) -> None:
    """Generate an HTML file using a Jinja2 template.
    
    # Parameters:
     - `env : jinja2.Environment`
        Jinja2 environment
     - `maze_examples : List[Dict[str, Any]]`
        List of maze example data
     - `all_tags : Set[str]`
        Set of all unique tags
    """
    template = env.get_template(TEMPLATE_FILE)
    
    # Prepare the context for the template
    context = {
        'maze_examples': maze_examples,
        'all_tags': sorted(all_tags)
    }
    
    # Render the template and write to file
    with open(HTML_PATH, 'w') as f:
        f.write(template.render(**context))

def main() -> None:
    """Main function to generate maze examples."""
    print(f"Generating maze examples in {EXAMPLES_DIR}")
    
    # Make sure directories exist
    EXAMPLES_DIR.mkdir(exist_ok=True)
    SVG_DIR.mkdir(exist_ok=True)
    TEMPLATE_DIR.mkdir(exist_ok=True)
    
    # Copy the template file from the repository or create it if it doesn't exist
    template_path = TEMPLATE_DIR / TEMPLATE_FILE
    if not template_path.exists():
        print(f"Creating template file at {template_path}")
        # You would need to have the template content here or copy it from somewhere
    
    # Get configurations
    configs = generate_maze_configs()
    print(f"Found {len(configs)} maze configurations")
    
    # Generate maze examples
    maze_examples = []
    all_tags: Set[str] = set()
    
    for i, metadata in enumerate(configs):
        print(f"Generating maze {i+1}/{len(configs)}: {metadata['name']}")
        svg_path, metadata = generate_maze_svg(metadata, seed=i)  # Use index as seed for reproducibility
        
        # Update the set of all tags
        all_tags.update(metadata["tags"])
        
        # Prepare the example data for the template
        maze_examples.append({
            "name": metadata["name"],
            "grid_n": metadata["grid_n"],
            "algorithm_name": metadata["maze_ctor"],
            "description": metadata["description"],
            "tags": metadata["tags"],
            "svg_path": svg_path,
            "config": metadata
        })
    
    # Set up Jinja2 and generate HTML
    jinja_env = setup_jinja_env()
    generate_html_from_template(jinja_env, maze_examples, all_tags)
    print(f"Generated HTML at {HTML_PATH}")

if __name__ == "__main__":
    main()