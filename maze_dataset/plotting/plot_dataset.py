import matplotlib.pyplot as plt

from maze_dataset.dataset.maze_dataset import MazeDataset


def plot_dataset_mazes(ds: MazeDataset, count: int | None = None) -> tuple:
    count = count or len(ds)
    if count == 0:
        print(f"No mazes to plot for dataset")
        return
    fig, axes = plt.subplots(1, count, figsize=(15, 5))
    if count == 1:
        axes = [axes]
    for i in range(count):
        axes[i].imshow(ds[i].as_pixels())
        # remove ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    return fig, axes


def print_dataset_mazes(ds: MazeDataset, count: int | None = None):
    count = count or len(ds)
    if count == 0:
        print(f"No mazes to print for dataset")
        return
    for i in range(count):
        print(ds[i].as_ascii(), "\n\n-----\n")
