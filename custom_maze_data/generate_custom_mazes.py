#!/usr/bin/env python3
"""Utility for generating custom maze renders/videos tailored for video diffusion experiments.

Features
--------
* Generates equal numbers of mazes across the requested grid sizes (default: 3/5/7/9).
* Uses maze-dataset generation primitives so you still get authentic SolvedMaze objects.
* Renders reference/goal images at any requested resolution (default 256x256) with circular
  or star-shaped endpoint markers (circle 70%, star 30%).
* Produces intermediate frames (default 24) following the solution path, and can optionally
  assemble those frames into an mp4/gif clip.
* Stores per-sample metadata (grid size, generator, seed, marker styles, etc.) for auditing.
* Displays a tqdm progress bar so you can track dataset generation at a glance.

The heavy lifting (maze generation/path solving) is CPU-bound and already optimized in the
underlying maze-dataset package; there is no GPU code path here, but you can run multiple copies
of the script in parallel or invoke it via a job scheduler if you need more throughput.

Example
-------
python data/generate_custom_mazes.py \\
    --total-mazes 40 \\
    --output-dir data/example_output \\
    --resolution 256 \\
    --frames 24 \\
    --fps 12 \\
    --write-video
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

try:
	# imageio is only required when writing videos
	import imageio.v3 as iio
except ModuleNotFoundError:  # pragma: no cover - optional dependency
	iio = None  # type: ignore[assignment]

try:
	from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
	tqdm = None  # type: ignore[assignment]

try:
	# ffmpeg plugin for imageio video writing
	import imageio_ffmpeg  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
	imageio_ffmpeg = None

from maze_dataset.generation import GENERATORS_MAP
from maze_dataset.maze import LatticeMaze, SolvedMaze
from maze_dataset.maze.lattice_maze import PixelColors

MARKER_PROBABILITY: float = 0.7  # Circle probability


@dataclass(frozen=True)
class RenderConfig:
	"""Rendering-related configuration."""

	resolution: int = 256
	n_frames: int = 24
	fps: int = 8
	write_video: bool = False
	video_format: str = "mp4"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate maze renders with custom endpoint markers and intermediate frames.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/maze_output"),
		help="Where to write rendered samples and metadata.",
	)
	parser.add_argument(
		"--total-mazes",
		type=int,
		default=32,
		help="Total number of mazes to generate across all grid sizes.",
	)
	parser.add_argument(
		"--grid-sizes",
		type=int,
		nargs="+",
		default=[3, 5, 7, 9],
		help="Grid sizes (n means n x n) to cover uniformly.",
	)
	parser.add_argument(
		"--generators",
		type=str,
		nargs="+",
		default=["gen_dfs"],
		help="Maze generator functions to cycle through (names from maze_dataset.generation.GENERATORS_MAP).",
	)
	parser.add_argument(
		"--resolution",
		type=int,
		default=256,
		help="Final square image resolution (pixels).",
	)
	parser.add_argument(
		"--frames",
		type=int,
		default=24,
		help="Number of intermediate frames to render per maze (set to 0 to skip).",
	)
	parser.add_argument(
		"--fps",
		type=int,
		default=12,
		help="Frames per second when encoding videos.",
	)
	parser.add_argument(
		"--write-video",
		action="store_true",
		help="If set, encode intermediate frames into a single video per maze.",
	)
	parser.add_argument(
		"--video-format",
		type=str,
		choices=["mp4", "gif"],
		default="mp4",
		help="Video container to use when --write-video is enabled.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed for reproducible generation and marker sampling.",
	)
	return parser.parse_args()


def _split_counts(total: int, n_parts: int) -> list[int]:
	if n_parts <= 0:
		return []
	base = total // n_parts
	remainder = total % n_parts
	output: list[int] = [base] * n_parts
	for idx in range(remainder):
		output[idx] += 1
	return output


def _generate_solved_maze(
	grid_n: int,
	maze_ctor_name: str,
	endpoint_kwargs: dict | None,
	max_attempts: int = 32,
) -> SolvedMaze:
	if maze_ctor_name not in GENERATORS_MAP:
		err_msg: str = f"Unknown generator '{maze_ctor_name}'. Available: {sorted(GENERATORS_MAP)}"
		raise ValueError(err_msg)

	maze_ctor = GENERATORS_MAP[maze_ctor_name]
	endpoint_kwargs = endpoint_kwargs or dict(endpoints_not_equal=True)

	for attempt in range(max_attempts):
		maze: LatticeMaze = maze_ctor(grid_shape=(grid_n, grid_n))
		solution = maze.generate_random_path(**endpoint_kwargs, except_on_no_valid_endpoint=False)
		if solution is None or solution.shape[0] == 0:
			continue
		return SolvedMaze.from_lattice_maze(maze, solution=solution)
	err_msg = f"Failed to generate a valid maze after {max_attempts} attempts (grid={grid_n}, generator={maze_ctor_name})"
	raise RuntimeError(err_msg)


def _draw_path_on_pixels(
	pixel_grid: np.ndarray,
	path_coords: np.ndarray,
	color: tuple[int, int, int] = PixelColors.PATH,
) -> np.ndarray:
	if path_coords.size == 0:
		return pixel_grid
	grid = pixel_grid.copy()
	for coord in path_coords:
		grid[coord[0] * 2 + 1, coord[1] * 2 + 1] = color
	for current, nxt in zip(path_coords[:-1], path_coords[1:], strict=False):
		grid[
			current[0] * 2 + 1 + (nxt[0] - current[0]),
			current[1] * 2 + 1 + (nxt[1] - current[1]),
		] = color
	return grid


def _star_points(
	center: tuple[float, float],
	outer_radius: float,
	inner_radius: float,
	n_branches: int = 5,
) -> list[tuple[float, float]]:
	points: list[tuple[float, float]] = []
	angle = -math.pi / 2.0
	step = math.pi / n_branches
	for idx in range(n_branches * 2):
		radius = outer_radius if idx % 2 == 0 else inner_radius
		points.append(
			(
				center[0] + radius * math.cos(angle),
				center[1] + radius * math.sin(angle),
			)
		)
		angle += step
	return points


def _draw_marker(
	draw_ctx: ImageDraw.ImageDraw,
	coord: np.ndarray,
	base_pixels: int,
	resolution: int,
	shape: str,
	color: tuple[int, int, int],
) -> None:
	pixel_row = coord[0] * 2 + 1
	pixel_col = coord[1] * 2 + 1
	scale = resolution / base_pixels
	center = (
		(pixel_col + 0.5) * scale,
		(pixel_row + 0.5) * scale,
	)
	radius = 0.45 * scale
	if shape == "circle":
		bbox = [
			center[0] - radius,
			center[1] - radius,
			center[0] + radius,
			center[1] + radius,
		]
		draw_ctx.ellipse(bbox, fill=color)
	else:
		points = _star_points(center=center, outer_radius=radius, inner_radius=radius * 0.5)
		draw_ctx.polygon(points, fill=color)


def _choose_shape(rng: random.Random) -> str:
	return "circle" if rng.random() < MARKER_PROBABILITY else "star"


def _render_frame(
	solved_maze: SolvedMaze,
	step_count: int,
	resolution: int,
	start_shape: str,
	end_shape: str,
) -> Image.Image:
	base = solved_maze.as_pixels(show_solution=False, show_endpoints=False)
	if step_count > 0:
		path = solved_maze.solution[:step_count]
		base = _draw_path_on_pixels(base, path)
	img = Image.fromarray(base)
	if resolution != base.shape[0]:
		img = img.resize((resolution, resolution), resample=Image.NEAREST)
	draw_ctx = ImageDraw.Draw(img)
	_draw_marker(
		draw_ctx,
		solved_maze.start_pos,
		base_pixels=base.shape[0],
		resolution=resolution,
		shape=start_shape,
		color=PixelColors.START,
	)
	_draw_marker(
		draw_ctx,
		solved_maze.end_pos,
		base_pixels=base.shape[0],
		resolution=resolution,
		shape=end_shape,
		color=PixelColors.END,
	)
	return img


def _generate_frames(
	solved_maze: SolvedMaze,
	render_cfg: RenderConfig,
	start_shape: str,
	end_shape: str,
) -> list[Image.Image]:
	if render_cfg.n_frames <= 0:
		return []
	total_steps = solved_maze.solution.shape[0]
	frame_counts: list[int] = []
	for idx in range(1, render_cfg.n_frames + 1):
		target = math.ceil(idx / render_cfg.n_frames * total_steps)
		frame_counts.append(min(target, total_steps))
	frames: list[Image.Image] = []
	for count in frame_counts:
		frames.append(
			_render_frame(
				solved_maze,
				step_count=count,
				resolution=render_cfg.resolution,
				start_shape=start_shape,
				end_shape=end_shape,
			)
		)
	return frames


def _save_video(
	frames: Sequence[Image.Image],
	path: Path,
	fps: int,
	video_format: str,
) -> None:
	if not frames:
		return
	if iio is None:
		err_msg = "imageio is required for video export; install with: pip install imageio imageio-ffmpeg"
		raise RuntimeError(err_msg)
	np_frames = [np.array(frame) for frame in frames]
	if video_format == "gif":
		iio.imwrite(path, np_frames, duration=1 / max(fps, 1))
	else:
		# Use FFMPEG backend for MP4 encoding with imageio-ffmpeg
		iio.imwrite(path, np_frames, fps=fps)


def main() -> None:
	args = parse_args()
	random.seed(args.seed)
	np.random.seed(args.seed)
	user_rng = random.Random(args.seed)

	render_cfg = RenderConfig(
		resolution=args.resolution,
		n_frames=args.frames,
		fps=args.fps,
		write_video=args.write_video,
		video_format=args.video_format,
	)

	output_dir: Path = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	grid_sizes = sorted(set(args.grid_sizes))
	per_grid_counts = _split_counts(args.total_mazes, len(grid_sizes))
	total_target = sum(per_grid_counts)
	progress = (
		tqdm(total=total_target, desc="Generating mazes", unit="maze")
		if (tqdm is not None and total_target > 0)
		else None
	)

	sample_index: int = 0
	for grid_n, size_target in zip(grid_sizes, per_grid_counts, strict=False):
		if size_target == 0:
			continue
		generator_names = list(dict.fromkeys(args.generators))  # Preserve order, remove dupes
		counts_per_generator = _split_counts(size_target, len(generator_names))
		for gen_name, gen_target in zip(generator_names, counts_per_generator, strict=False):
			for _ in range(gen_target):
				sample_index += 1
				solved_maze = _generate_solved_maze(
					grid_n=grid_n,
					maze_ctor_name=gen_name,
					endpoint_kwargs=dict(endpoints_not_equal=True),
				)
				start_shape = _choose_shape(user_rng)
				end_shape = _choose_shape(user_rng)
				frames = _generate_frames(
					solved_maze,
					render_cfg=render_cfg,
					start_shape=start_shape,
					end_shape=end_shape,
				)
				reference_image = _render_frame(
					solved_maze,
					step_count=0,
					resolution=render_cfg.resolution,
					start_shape=start_shape,
					end_shape=end_shape,
				)
				goal_image = _render_frame(
					solved_maze,
					step_count=solved_maze.solution.shape[0],
					resolution=render_cfg.resolution,
					start_shape=start_shape,
					end_shape=end_shape,
				)

				sample_dir = output_dir / f"size_{grid_n}" / f"{gen_name}" / f"maze_{sample_index:06d}"
				sample_dir.mkdir(parents=True, exist_ok=True)
				(reference_image).save(sample_dir / "reference.png")
				(goal_image).save(sample_dir / "goal.png")

				frame_dir = sample_dir / "frames"
				if frames:
					frame_dir.mkdir(exist_ok=True)
					for idx, frame in enumerate(frames):
						frame.save(frame_dir / f"frame_{idx:03d}.png")

				if render_cfg.write_video and frames:
					video_path = sample_dir / f"solution.{render_cfg.video_format}"
					_save_video(frames, path=video_path, fps=render_cfg.fps, video_format=render_cfg.video_format)

				metadata = {
					"grid_n": grid_n,
					"generator": gen_name,
					"path_length": int(solved_maze.solution.shape[0]),
					"start_shape": start_shape,
					"end_shape": end_shape,
					"resolution": render_cfg.resolution,
					"n_frames": render_cfg.n_frames,
					"fps": render_cfg.fps,
					"video": render_cfg.write_video,
					"seed": args.seed,
					"maze_index": sample_index,
				}
					with (sample_dir / "metadata.json").open("w", encoding="utf-8") as fp:
						json.dump(metadata, fp, indent=2)

					if progress is not None:
						progress.update(1)

	if sample_index == 0:
		print("No mazes were generated. Check --total-mazes or --grid-sizes.")
	else:
		print(f"Generated {sample_index} mazes in {output_dir}")
	if progress is not None:
		progress.close()


if __name__ == "__main__":
	main()
