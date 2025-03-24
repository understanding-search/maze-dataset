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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from maze_dataset import LatticeMaze, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot

# Define the examples directory
EXAMPLES_DIR = pathlib.Path("docs/examples")
SVG_DIR = EXAMPLES_DIR / "svg"
HTML_PATH = EXAMPLES_DIR / "maze_examples.html"

# Make sure directories exist
EXAMPLES_DIR.mkdir(exist_ok=True)
SVG_DIR.mkdir(exist_ok=True)

# Define a variety of maze configurations to showcase different features
def generate_maze_configs() -> List[Dict[str, Any]]:
    """Generate a list of diverse maze configurations."""
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

def generate_maze_svg(config: Dict[str, Any], seed: Optional[int] = None) -> Tuple[str, str]:
    """Generate a maze from the given configuration and save it as an SVG file.
    
    Returns:
        Tuple[str, str]: Path to the SVG file and the JSON string of the configuration
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Extract maze config parameters
    name = config["name"]
    grid_n = config["grid_n"]
    maze_ctor = config["maze_ctor"]
    maze_ctor_kwargs = config["maze_ctor_kwargs"]
    endpoint_kwargs = config.get("endpoint_kwargs", {})
    
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
    config_json = {k: v for k, v in config.items() if k not in ["maze_ctor"]}
    config_json["maze_ctor"] = config["maze_ctor"].__name__
    config_json_str = json.dumps(config_json, indent=2)
    
    # Create a corresponding config file
    config_path = SVG_DIR / f"{name}.json"
    with open(config_path, 'w') as f:
        f.write(config_json_str)
    
    return str(svg_path.relative_to(EXAMPLES_DIR)), config_json_str

def generate_html(maze_examples: List[Dict[str, Any]]) -> None:
    """Generate an HTML file to display the maze examples."""
    all_tags = set()
    for example in maze_examples:
        all_tags.update(example["tags"])
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maze Dataset Examples</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .search-container {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        .filter-section {
            margin-bottom: 15px;
        }
        .filter-section h3 {
            margin-bottom: 8px;
        }
        .tag-filters {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }
        .tag-filter {
            background-color: #e0e0e0;
            padding: 5px 10px;
            border-radius: 15px;
            cursor: pointer;
            user-select: none;
        }
        .tag-filter.active {
            background-color: #007bff;
            color: white;
        }
        input[type="text"], select {
            padding: 8px;
            width: 100%;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .maze-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .maze-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .maze-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .maze-image {
            width: 100%;
            height: 300px;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f9f9f9;
            padding: 10px;
            box-sizing: border-box;
        }
        .maze-image img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .maze-details {
            padding: 15px;
        }
        .maze-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .maze-description {
            margin-bottom: 10px;
            color: #555;
        }
        .maze-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 10px;
        }
        .maze-tag {
            background-color: #e0e0e0;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
        }
        .config-toggle {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .config-code {
            display: none;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .copy-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 5px;
        }
        .no-results {
            grid-column: 1 / -1;
            text-align: center;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        footer {
            margin-top: 40px;
            text-align: center;
            color: #777;
            font-size: 14px;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Maze Dataset Examples</h1>
    
    <div class="search-container">
        <div class="filter-section">
            <h3>Search by Keywords</h3>
            <input type="text" id="search-input" placeholder="Search by name, description, or configuration...">
        </div>
        
        <div class="filter-section">
            <h3>Filter by Tags</h3>
            <div class="tag-filters">
"""

    # Add tags
    for tag in sorted(all_tags):
        html_content += f'                <div class="tag-filter" data-tag="{tag}">{tag}</div>\n'

    html_content += """
            </div>
        </div>
        
        <div class="filter-section">
            <h3>Sort By</h3>
            <select id="sort-select">
                <option value="name">Name</option>
                <option value="grid_n">Grid Size</option>
                <option value="algorithm">Algorithm</option>
            </select>
        </div>
    </div>
    
    <div class="maze-grid" id="maze-grid">
"""

    # Add maze cards
    for example in maze_examples:
        html_content += f"""
        <div class="maze-card" 
             data-name="{example['name']}" 
             data-grid-n="{example['grid_n']}" 
             data-algorithm="{example['maze_ctor'].__name__}" 
             data-tags="{','.join(example['tags'])}">
            <div class="maze-image">
                <img src="{example['svg_path']}" alt="{example['name']}">
            </div>
            <div class="maze-details">
                <div class="maze-title">{example['description']}</div>
                <div class="maze-description">Grid size: {example['grid_n']}x{example['grid_n']}</div>
                <div class="maze-tags">
"""
        
        for tag in example["tags"]:
            html_content += f'                    <span class="maze-tag">{tag}</span>\n'
            
        html_content += f"""
                </div>
                <button class="config-toggle">Show Configuration</button>
                <div class="config-code">{example['config_json']}</div>
                <button class="copy-button">Copy Config</button>
            </div>
        </div>
"""

    html_content += """
        <div class="no-results" style="display: none;">No mazes match your search criteria.</div>
    </div>

    <footer>
        Generated using <a href="https://github.com/understanding-search/maze-dataset">maze-dataset</a>
    </footer>

    <script>
        // Show/hide configuration
        document.querySelectorAll('.config-toggle').forEach(button => {
            button.addEventListener('click', function() {
                const codeBlock = this.nextElementSibling;
                if (codeBlock.style.display === 'none' || codeBlock.style.display === '') {
                    codeBlock.style.display = 'block';
                    this.textContent = 'Hide Configuration';
                } else {
                    codeBlock.style.display = 'none';
                    this.textContent = 'Show Configuration';
                }
            });
        });

        // Copy configuration
        document.querySelectorAll('.copy-button').forEach(button => {
            button.addEventListener('click', function() {
                const codeBlock = this.previousElementSibling;
                
                // Create a temporary textarea element
                const textarea = document.createElement('textarea');
                textarea.value = codeBlock.textContent;
                document.body.appendChild(textarea);
                
                // Copy the text
                textarea.select();
                document.execCommand('copy');
                
                // Remove the textarea
                document.body.removeChild(textarea);
                
                // Give feedback to the user
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 2000);
            });
        });

        // Filter functionality
        const searchInput = document.getElementById('search-input');
        const sortSelect = document.getElementById('sort-select');
        const mazeCards = document.querySelectorAll('.maze-card');
        const tagFilters = document.querySelectorAll('.tag-filter');
        const noResults = document.querySelector('.no-results');
        
        let activeTagFilters = [];
        
        function filterMazes() {
            const searchTerm = searchInput.value.toLowerCase();
            let visibleCount = 0;
            
            mazeCards.forEach(card => {
                const name = card.getAttribute('data-name').toLowerCase();
                const description = card.querySelector('.maze-description').textContent.toLowerCase();
                const config = card.querySelector('.config-code').textContent.toLowerCase();
                const tags = card.getAttribute('data-tags').split(',');
                
                const matchesSearch = name.includes(searchTerm) || 
                                    description.includes(searchTerm) || 
                                    config.includes(searchTerm);
                
                const matchesTags = activeTagFilters.length === 0 || 
                                  activeTagFilters.every(tag => tags.includes(tag));
                
                if (matchesSearch && matchesTags) {
                    card.style.display = 'block';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });
            
            noResults.style.display = visibleCount === 0 ? 'block' : 'none';
        }
        
        function sortMazes() {
            const sortBy = sortSelect.value;
            const mazeGrid = document.getElementById('maze-grid');
            const mazeArray = Array.from(mazeCards);
            
            mazeArray.sort((a, b) => {
                if (sortBy === 'name') {
                    return a.getAttribute('data-name').localeCompare(b.getAttribute('data-name'));
                } else if (sortBy === 'grid_n') {
                    return parseInt(a.getAttribute('data-grid-n')) - parseInt(b.getAttribute('data-grid-n'));
                } else if (sortBy === 'algorithm') {
                    return a.getAttribute('data-algorithm').localeCompare(b.getAttribute('data-algorithm'));
                }
                return 0;
            });
            
            mazeArray.forEach(card => {
                mazeGrid.appendChild(card);
            });
        }
        
        searchInput.addEventListener('input', filterMazes);
        sortSelect.addEventListener('change', sortMazes);
        
        tagFilters.forEach(tag => {
            tag.addEventListener('click', function() {
                const tagValue = this.getAttribute('data-tag');
                
                if (this.classList.contains('active')) {
                    // Remove tag from active filters
                    this.classList.remove('active');
                    activeTagFilters = activeTagFilters.filter(t => t !== tagValue);
                } else {
                    // Add tag to active filters
                    this.classList.add('active');
                    activeTagFilters.push(tagValue);
                }
                
                filterMazes();
            });
        });
        
        // Initial sort
        sortMazes();
    </script>
</body>
</html>
    """
    
    with open(HTML_PATH, 'w') as f:
        f.write(html_content)

def main() -> None:
    """Main function to generate maze examples."""
    print(f"Generating maze examples in {EXAMPLES_DIR}")
    
    # Get configurations
    configs = generate_maze_configs()
    print(f"Found {len(configs)} maze configurations")
    
    # Generate maze examples
    maze_examples = []
    for i, config in enumerate(configs):
        print(f"Generating maze {i+1}/{len(configs)}: {config['name']}")
        svg_path, config_json = generate_maze_svg(config, seed=i)  # Use index as seed for reproducibility
        
        maze_examples.append({
            **config,
            "svg_path": svg_path,
            "config_json": config_json
        })
    
    # Generate HTML
    generate_html(maze_examples)
    print(f"Generated HTML at {HTML_PATH}")

if __name__ == "__main__":
    main()