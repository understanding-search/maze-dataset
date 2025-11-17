# Maze Data Generation Plan

## Goals
- Generate a corpus of ~100k solved mazes across grid sizes 3×3, 5×5, 7×7, and 9×9 plus a “uniform/full” category for qualitative variety.
- For every maze, persist:
  - `reference_image`: initial maze layout without the solution (RGB PNG).
  - `goal_image`: same maze with the ground-truth solution path highlighted.
  - `instruction`: templated natural-language guidance that pairs the reference and goal frames.
  - `reference_video` (optional but desired): a short clip that visualizes the solver progressively tracing the path.
- Support future extensions (other visual styles, metadata filters, additional maze algorithms) without regenerating everything from scratch.

## Existing Capabilities to Leverage
- `MazeDataset` + `MazeDatasetConfig` already manage batched generation of `SolvedMaze` objects via DFS, Wilson, percolation, etc., and expose seeding/parallelism knobs (`maze_dataset/dataset/maze_dataset.py:1-188`).
- Each `SolvedMaze` contains both the maze structure (`connection_list`) and the full ordered `solution` path (`maze_dataset/maze/lattice_maze.py:1275-1398`).
- `SolvedMaze.as_pixels(show_solution=...)` renders RGB grids with or without the solution and endpoints, so we can reuse the same rasterization logic for `reference_image` and `goal_image` (`maze_dataset/maze/lattice_maze.py:858-923`).
- `MazePlot` can overlay styled paths on matplotlib figures, enabling richer visualizations or higher-resolution renders for videos (`maze_dataset/plotting/plot_maze.py:1-200, 400-520`).

## Maze Generation Strategy
1. **Size buckets & counts**  
   - Target equal counts per size to reach 100k (e.g., 20k per grid for 3×3, 5×5, 7×7, 9×9, and “uniform 0.6 density percolation”).  
   - Keep a config manifest so we can regenerate subsets deterministically.

2. **Algorithms & styles**  
   - For each size bucket, rotate between DFS (`LatticeMazeGenerators.gen_dfs`), Wilson (`gen_wilson`), and percolation (`gen_percolation`).  
   - Track `maze_ctor` + kwargs inside metadata for reproducibility.

3. **Config driving script**  
   - Define a table of `MazeDatasetConfig` objects (grid_n, n_mazes chunk size, ctor, kwargs, endpoint constraints).  
   - Use `MazeDataset.from_config(..., do_generate=True, load_local=False)` to force fresh generation per chunk and write outputs straight to disk pipelines.
   - Generate in batches of ≤2k mazes per worker to keep RAM bounded, accumulate until global target is hit.

4. **Seeding & reproducibility**  
   - Accept a top-level seed; pass deterministic `cfg.seed` values for each chunk (e.g., `base_seed + chunk_idx`).  
   - Persist chunk manifests (`config_hash`, num successes, time) so runs can resume if interrupted.

## Rendering & Snapshot Plan
1. **Reference/goal images**  
   - `reference_image`: call `solved_maze.as_pixels(show_solution=False)` and write with `PIL.Image.fromarray`.  
   - `goal_image`: same call with `show_solution=True` to overlay the true path.  
   - Optionally run through `MazePlot` to generate vector-looking PNGs for stylistic diversity (switch via config flag).

2. **Intermediate snapshots for videos**  
   - The ordered coordinates live in `SolvedMaze.solution`, so we can iterate prefixes `[0:k]` and recolor them onto a copy of the reference pixel grid using the same loop as in `as_pixels` (`maze_dataset/maze/lattice_maze.py:897-920`).  
   - Pseudocode:  
     ```python
     base = solved_maze.as_pixels(show_solution=False)
     coords = solved_maze.solution
     frames = []
     for k in range(1, len(coords) + 1):
         frame = base.copy()
         paint_solution(frame, coords[:k])  # reuse the inner loop from as_pixels
         frames.append(frame)
     ```
   - For higher-fidelity renders, use `MazePlot.add_true_path(path[:k])` and `plot(plain=True)` to capture matplotlib frames, which also allows varying colors, line styles, and node annotations.
   - Encode frames into MP4/GIF using `imageio.v3.imwrite` or `imageio.mimsave`, keeping per-video length ~1–3 seconds. Maintain a flag to skip video generation if disk pressure is high.

3. **Multiple visualization presets**  
   - Define a handful of palettes (classic black/white, heatmap, textured).  
   - Each sample stores metadata about which preset was used so downstream models can filter.

## Instruction Generation
- Use deterministic templates that reference grid size and maze properties, e.g.,  
  - “Solve the `{{grid_n}}×{{grid_n}}` maze and reach the exit marked in red.”  
  - “Trace the blue path from the green start to the red goal without crossing walls.”  
- Include variability (mention algorithm, mention approximate path length, or highlight interesting facts like “starts in top-left dead-end”).  
- Consider augmenting with templated reasoning (“At the first fork head east, then follow the corridor south.”) using `SolvedMaze.get_solution_forking_points(...)` for richer prompts if desired.

## Data Layout & Metadata
```
data/
  maze_videos/
    size_3/
      chunk_0001/
        maze_000001/
          reference.png
          goal.png
          video.mp4
          metadata.json  # algorithm, seed, path_length, instruction text, etc.
```
- Maintain a global `index.jsonl` (or Parquet) where each row contains sample ID, relative paths, instruction text, vectorized metadata (size, algorithm, path length, success flag, visualization preset).  
- Include hashes of the image/video files to detect corruption.

## Pipeline Outline
1. Parse a YAML/JSON config describing desired size buckets, counts, visualization presets, and chunk sizes.  
2. For each chunk spec:  
   1. Instantiate `MazeDatasetConfig` with `n_mazes=chunk_size` and generate `SolvedMaze` samples.  
   2. For every sample:  
      - Save `reference_image`, `goal_image`, computed `instruction`, metadata JSON.  
      - If enabled, build intermediate frames and encode `reference_video`.  
      - Record success/failure stats.  
   3. Flush chunk-level manifest (counts, elapsed time, RNG seed) for reproducibility.  
3. After all chunks, build the consolidated `index.jsonl`, shuffle entries if needed, and run validation scripts.  
4. Optionally compress data directory per chunk for archival.

## Quality Assurance
- **Automated checks**: verify image sizes, ensure `goal_image` contains a connected path matching `solution`, confirm video frame counts align with path length, and re-run `SolvedMaze.from_pixels` spot checks to ensure reversibility.  
- **Stat collection**: log distribution of path lengths, branching factors, algorithm usage, and failure rates so we can rebalance if needed.  
- **Visual spot checks**: randomly sample outputs per chunk and compile contact sheets / GIFs for manual review.  
- **Performance monitoring**: track generation throughput (mazes/sec) and video encoding time to adjust chunk sizing or parallelism.

## Next Steps
1. Turn the outline above into a Python CLI (e.g., `scripts/generate_mazes.py`) that accepts the config manifest and manages chunking, rendering, and metadata writing.  
2. Prototype the video snapshot routine on a small sample to confirm runtime and visual fidelity.  
3. Decide on concrete instruction templates and finalize metadata schema before scaling to 100k samples.  
4. Once satisfied with the pilot run, launch the full generation job with monitoring + resumable manifests.

