#!/bin/bash
# Bash script for generating custom maze datasets with full configuration
# This script provides a template with all available parameters explicitly documented

set -e  # Exit on error

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Output directory where all generated mazes will be stored
OUTPUT_DIR="data/maze_output"

# Total number of mazes to generate across all grid sizes
TOTAL_MAZES=32

# Grid sizes to generate (space-separated list of integers)
# Each size means an NxN grid
# Example: "3 5 7 9" generates equal numbers across 3x3, 5x5, 7x7, and 9x9 grids
GRID_SIZES="3 5 7 9"

# Maze generation algorithms to use
# Available generators from maze_dataset.generation.GENERATORS_MAP:
#   - gen_dfs: Depth-first search
#   - gen_wilson: Wilson's algorithm
#   - gen_kruskal: Kruskal's algorithm
#   - gen_prim: Prim's algorithm
#   - gen_binary_tree: Binary tree algorithm
# You can list multiple generators (space-separated), and they will be cycled through
GENERATORS="gen_dfs"

# Output image resolution in pixels (square images)
# Larger values = higher quality renders
# Example: 256, 512, 1024
RESOLUTION=256

# Number of intermediate frames to generate per maze
# These show the solution path progressively from start to end
# Set to 0 to skip frame generation
FRAMES=24

# Frames per second for video encoding
# Only used when --write-video is enabled
FPS=12

# Whether to encode frames into a video file
# Set to true to enable, false to disable
WRITE_VIDEO=true

# Video format to use (only applies when WRITE_VIDEO=true)
# Options: "mp4" or "gif"
# Note: MP4 requires imageio-ffmpeg to be installed
#       GIF works with just imageio
VIDEO_FORMAT="mp4"

# Random seed for reproducible generation
# Use the same seed to regenerate identical datasets
SEED=1234

# ==============================================================================
# USAGE DOCUMENTATION
# ==============================================================================
# 
# Generated Output Structure:
# ├── data/maze_output/
# │   ├── size_3/
# │   │   ├── gen_dfs/
# │   │   │   ├── maze_000001/
# │   │   │   │   ├── reference.png      (start state image)
# │   │   │   │   ├── goal.png           (solved state image)
# │   │   │   │   ├── metadata.json      (configuration metadata)
# │   │   │   │   ├── frames/            (if FRAMES > 0)
# │   │   │   │   │   ├── frame_000.png
# │   │   │   │   │   ├── frame_001.png
# │   │   │   │   │   └── ...
# │   │   │   │   └── solution.mp4       (if WRITE_VIDEO=true)
# │   │   │   ├── maze_000002/
# │   │   │   └── ...
# │   │   └── ...
# │   ├── size_5/
# │   └── ...
#
# Metadata JSON contents:
#   - grid_n: Grid size (e.g., 5 for 5x5)
#   - generator: Generation algorithm used
#   - path_length: Number of steps in solution
#   - start_shape: Marker shape at start (circle or star)
#   - end_shape: Marker shape at end (circle or star)
#   - resolution: Image resolution in pixels
#   - n_frames: Number of frames generated
#   - fps: Frames per second for video
#   - video: Whether video was generated
#   - seed: Random seed used
#   - maze_index: Sequential index of this maze
#
# ==============================================================================
# INSTALLATION NOTES
# ==============================================================================
#
# For basic functionality (PNG images only):
#   pip install pillow numpy imageio
#
# For video output (MP4):
#   pip install pillow numpy imageio imageio-ffmpeg
#
# Or using the project's optional dependencies:
#   pip install -e ".[video]"
#   uv pip install -e ".[video]"
#
# ==============================================================================

# Function to print configuration
print_config() {
	echo "======================================================================"
	echo "MAZE DATASET GENERATION CONFIGURATION"
	echo "======================================================================"
	echo "Output Directory:        $OUTPUT_DIR"
	echo "Total Mazes:             $TOTAL_MAZES"
	echo "Grid Sizes:              $GRID_SIZES"
	echo "Generators:              $GENERATORS"
	echo "Resolution (pixels):     ${RESOLUTION}x${RESOLUTION}"
	echo "Frames per Maze:         $FRAMES"
	echo "Video FPS:               $FPS"
	echo "Write Video:             $WRITE_VIDEO"
	if [ "$WRITE_VIDEO" = true ]; then
		echo "Video Format:            $VIDEO_FORMAT"
	fi
	echo "Random Seed:             $SEED"
	echo "======================================================================"
	echo ""
}

# Function to run generation
run_generation() {
	print_config
	
	# Build command with all parameters
	CMD="python custom_maze_data/generate_custom_mazes.py"
	CMD="$CMD --output-dir $OUTPUT_DIR"
	CMD="$CMD --total-mazes $TOTAL_MAZES"
	CMD="$CMD --grid-sizes $GRID_SIZES"
	CMD="$CMD --generators $GENERATORS"
	CMD="$CMD --resolution $RESOLUTION"
	CMD="$CMD --frames $FRAMES"
	CMD="$CMD --fps $FPS"
	CMD="$CMD --seed $SEED"
	
	if [ "$WRITE_VIDEO" = true ]; then
		CMD="$CMD --write-video"
		CMD="$CMD --video-format $VIDEO_FORMAT"
	fi
	
	echo "Running command:"
	echo "$CMD"
	echo ""
	
	eval "$CMD"
}

# ==============================================================================
# EXAMPLE CONFIGURATIONS
# ==============================================================================
#
# Example 1: Quick test with small dataset
#   OUTPUT_DIR="data/test_output"
#   TOTAL_MAZES=4
#   GRID_SIZES="3 5"
#   FRAMES=12
#   WRITE_VIDEO=false
#
# Example 2: Large high-quality dataset
#   TOTAL_MAZES=1000
#   GRID_SIZES="3 5 7 9 11"
#   GENERATORS="gen_dfs gen_wilson gen_prim"
#   RESOLUTION=512
#   FRAMES=48
#   FPS=24
#   WRITE_VIDEO=true
#
# Example 3: GIF output only (no MP4 encoding)
#   WRITE_VIDEO=true
#   VIDEO_FORMAT="gif"
#   # No need for imageio-ffmpeg, just imageio
#
# ==============================================================================

# Run the generation
run_generation
