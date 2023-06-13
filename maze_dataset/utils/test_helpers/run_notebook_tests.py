import os
import subprocess
import sys
from pathlib import Path


class NotebookTestError(Exception):
    pass


def test_notebooks(
    notebooks_dir: Path,
    converted_notebooks_temp_dir: Path,
    CI_output_suffix: str = ".CI-output.txt",
):
    # get paths
    notebooks_dir = Path(notebooks_dir)
    converted_notebooks_temp_dir = Path(converted_notebooks_temp_dir)
    root_relative_to_notebooks: Path = Path(os.path.relpath(".", notebooks_dir))

    print(f"testing notebooks in '{notebooks_dir}'")
    print(f"reading converted notebooks from '{converted_notebooks_temp_dir}'")

    try:
        # check things exist
        if not notebooks_dir.exists():
            raise NotebookTestError(f"Notebooks dir '{notebooks_dir}' does not exist")
        if not notebooks_dir.is_dir():
            raise NotebookTestError(
                f"Notebooks dir '{notebooks_dir}' is not a directory"
            )
        if not converted_notebooks_temp_dir.exists():
            raise NotebookTestError(
                f"Converted notebooks dir '{converted_notebooks_temp_dir}' does not exist"
            )
        if not converted_notebooks_temp_dir.is_dir():
            raise NotebookTestError(
                f"Converted notebooks dir '{converted_notebooks_temp_dir}' is not a directory"
            )

        notebooks: list[Path] = list(notebooks_dir.glob("*.ipynb"))
        if not notebooks:
            raise NotebookTestError(f"No notebooks found in '{notebooks_dir}'")

        converted_notebooks: list[Path] = list()
        for nb in notebooks:
            converted_file: Path = (
                converted_notebooks_temp_dir / nb.with_suffix(".py").name
            )
            if not converted_file.exists():
                raise NotebookTestError(
                    f"Did not find converted notebook '{converted_file}' for '{nb}'"
                )
            converted_notebooks.append(converted_file)

        converted_file = None

        # the location of this line is important
        os.chdir(notebooks_dir)

        for file in converted_notebooks:
            # run the file
            print(f"  Running {file}")
            output_file: Path = file.with_suffix(CI_output_suffix)
            print(f"  Output in {output_file}")

            command: str = f"poetry run python {root_relative_to_notebooks / file} > {root_relative_to_notebooks / output_file} 2>&1"
            process: subprocess.CompletedProcess = subprocess.run(
                command, shell=True, text=True
            )

            print(f"  Run completed with return code {process.returncode}")

            # print the output of the file to the console if it failed
            if process.returncode != 0:
                with open(root_relative_to_notebooks / output_file, "r") as f:
                    file_output: str = f.read()
                raise NotebookTestError(f"Error in {file}:\n\n{file_output}")

            process = None

    except NotebookTestError as e:
        print("!" * 50, file=sys.stderr)
        print(e, file=sys.stderr)
        print("!" * 50, file=sys.stderr)
        raise e


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument(
        "--notebooks-dir",
        type=str,
        help="The directory from which to run the notebooks",
    )
    parser.add_argument(
        "--converted-notebooks-temp-dir",
        type=str,
        help="The directory containing the converted notebooks to test",
    )

    args: argparse.Namespace = parser.parse_args()

    test_notebooks(
        Path(args.notebooks_dir),
        Path(args.converted_notebooks_temp_dir),
    )
