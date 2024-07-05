from zanj import ZANJ

from maze_dataset import MazeDataset, MazeDatasetConfig


def test_remove_duplicates():

    cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="test_collect",
        grid_n=5,
        n_mazes=10,
    )

    dataset: MazeDataset = MazeDataset.from_config(
        cfg,
        load_local=False,
        save_local=True,
        local_base_path="tests/_temp/test_collect/",
        verbose=True,
        zanj=ZANJ(external_list_threshold=1000),
    )
    print(f"Generated {len(dataset)} mazes")

    dataset = dataset.filter_by.remove_duplicates(
        minimum_difference_connection_list=0, minimum_difference_solution=1
    )
    print(f"After removing duplicates, we have {len(dataset)} mazes")
