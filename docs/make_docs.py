import argparse
import inspect
import re
import tomllib
import warnings
from pathlib import Path

# notebooks
import jinja2
import nbconvert
import nbformat

# pdoc
import pdoc
import pdoc.doc
import pdoc.extract
import pdoc.render
import pdoc.render_helpers
from markupsafe import Markup

OUTPUT_DIR: Path = Path("docs")

pdoc.render_helpers.markdown_extensions["alerts"] = True
pdoc.render_helpers.markdown_extensions["admonitions"] = True


def increment_markdown_headings(markdown_text: str, increment: int = 2) -> str:
    """
    Increment all Markdown headings in the given text by the specified amount.
    Args:
        markdown_text (str): The input Markdown text.
        increment (int): The number of levels to increment the headings by. Default is 2.
    Returns:
        str: The Markdown text with incremented heading levels.
    """

    def replace_heading(match):
        current_level = len(match.group(1))
        new_level = min(current_level + increment, 6)  # Cap at h6
        return "#" * new_level + match.group(2)

    # Regular expression to match Markdown headings
    heading_pattern = re.compile(r"^(#{1,6})(.+)$", re.MULTILINE)

    # Replace all headings with incremented versions
    return heading_pattern.sub(replace_heading, markdown_text)


def format_signature(sig: inspect.Signature, colon: bool) -> str:
    """Format a function signature for Markdown. Returns a single-line Markdown string."""
    # First get a list with all params as strings.
    result = pdoc.doc._PrettySignature._params(sig)  # type: ignore
    return_annot = pdoc.doc._PrettySignature._return_annotation_str(sig)  # type: ignore

    def _format_param(param: str) -> str:
        """Format a parameter for Markdown, including potential links."""
        # This is a simplified version. You might need to adjust this
        # to properly handle links in your specific use case.
        return f"`{param}`"

    # Format each parameter
    pretty_result = [_format_param(param) for param in result]

    # Join parameters
    params_str = ", ".join(pretty_result)

    # Add return annotation
    anno = ")"
    if return_annot:
        anno += f" -> `{return_annot}`"
    if colon:
        anno += ":"

    # Construct the full signature
    rendered = f"`(`{params_str}`{anno}`"

    return rendered


def markup_safe(sig: inspect.Signature) -> str:
    output: str = str(sig)
    return Markup(output)


def use_markdown_format():
    pdoc.render_helpers.format_signature = format_signature
    pdoc.render.env.filters["markup_safe"] = markup_safe
    pdoc.render.env.filters["increment_markdown_headings"] = increment_markdown_headings


def ignore_warnings(config_path: str | Path = Path("pyproject.toml")):
    # Read the pyproject.toml file
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    # Extract the warning messages from the tool.pdoc.ignore section
    warning_messages: list[str] = (
        pyproject_data.get("tool", {}).get("pdoc", {}).get("warnings_ignore", [])
    )

    # Process and apply the warning filters
    for message in warning_messages:
        warnings.filterwarnings("ignore", message=message)


NOTEBOOKS_INDEX_TEMPLATE: str = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Notebooks</title>
    <link rel="stylesheet" href="../resources/bootstrap-reboot.min.css">
    <link rel="stylesheet" href="../resources/theme.css">
    <link rel="stylesheet" href="../resources/content.css">
</head>
<body>    
    <h1>Notebooks</h1>
    <p>
        You can find the source code for the notebooks at
        <a href="https://github.com/understanding-search/maze-dataset/tree/main/notebooks">
            github.com/understanding-search/maze-dataset/tree/main/notebooks
        </a>.
    <ul>
        {% for notebook in notebooks %}
        <li><a href="{{ notebook.html }}">{{ notebook.ipynb }}</a> {{ notebook.desc }}</li>
        {% endfor %}
    </ul>
    <a href="../">Back to index</a>
"""


NOTEBOOK_DESCRIPTIONS: dict[str, str] = dict(
    demo_dataset="how to easily create a dataset of mazes, utilities for filtering the generates mazes via properties, and basic visualization. View this one first.",
    demo_tokenization="converting mazes to and from textual representations, as well as utilities for working with them.",
    demo_latticemaze="internals of the `LatticeMaze` and `SolvedMaze` objects, and advanced visualization.",
)


def convert_notebooks(
    source_path: Path | str = Path("notebooks"),
    output_path: Path | str = Path("docs/notebooks"),
    index_template: str = NOTEBOOKS_INDEX_TEMPLATE,
):
    source_path = Path(source_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    notebook_names: list[Path] = list(source_path.glob("*.ipynb"))
    notebooks: list[dict[str, str]] = [
        dict(
            ipynb=notebook.name,
            html=notebook.with_suffix(".html").name,
            desc=NOTEBOOK_DESCRIPTIONS.get(notebook.stem, ""),
        )
        for notebook in notebook_names
    ]

    # Render the index template
    template: jinja2.Template = jinja2.Template(index_template)
    rendered_index: str = template.render(notebooks=notebooks)

    # Write the rendered index to a file
    index_path = output_path / "index.html"
    with open(index_path, "w") as f:
        f.write(rendered_index)

    # convert with nbconvert
    for notebook in notebook_names:
        output_notebook = output_path / notebook.with_suffix(".html").name
        with open(notebook, "r") as f:
            nb: nbformat.NotebookNode = nbformat.read(f, as_version=4)
            html_exporter: nbconvert.HTMLExporter = nbconvert.HTMLExporter()
            body: str
            body, _ = html_exporter.from_notebook_node(nb)
            with open(output_notebook, "w") as f:
                f.write(body)


def pdoc_combined(*modules: Path | str, output_file: Path) -> None:
    """Render the documentation for a list of modules into a single HTML file.
    Args:
        *modules: Paths or names of the modules to document.
        output_file: Path to the output HTML file.
    This function will:
    1. Extract all modules and submodules.
    2. Generate documentation for each module.
    3. Combine all module documentation into a single HTML file.
    4. Write the combined documentation to the specified output file.
    Rendering options can be configured by calling `pdoc.render.configure` in advance.
    """
    # Extract all modules and submodules
    all_modules: dict[str, pdoc.doc.Module] = {}
    for module_name in pdoc.extract.walk_specs(modules):
        all_modules[module_name] = pdoc.doc.Module.from_name(module_name)

    # Generate HTML content for each module
    module_contents: list[str] = []
    for module in all_modules.values():
        module_html = pdoc.render.html_module(module, all_modules)
        module_contents.append(module_html)

    # Combine all module contents
    combined_content = "\n".join(module_contents)

    # Write the combined content to the output file
    with output_file.open("w", encoding="utf-8") as f:
        f.write(combined_content)


if __name__ == "__main__":
    argparser: argparse.ArgumentParser = argparse.ArgumentParser()
    # whether to start an HTTP server to serve the documentation
    argparser.add_argument(
        "--serve",
        "-s",
        action="store_true",
        help="Whether to start an HTTP server to serve the documentation",
    )
    argparser.add_argument(
        "--warn-all",
        "-w",
        action="store_true",
        help="Whether to show all warnings, instead of ignoring the ones specified in pyproject.toml:tool.pdoc.ignore",
    )
    argparser.add_argument(
        "--combined",
        "-c",
        action="store_true",
        help="Whether to combine the documentation for multiple modules into a single markdown file",
    )
    argparser.add_argument(
        "--notebooks",
        "-n",
        action="store_true",
        help="convert notebooks to HTML",
    )
    parsed_args = argparser.parse_args()

    if parsed_args.notebooks:
        convert_notebooks()
        exit()

    if not parsed_args.warn_all:
        ignore_warnings()

    pdoc.render.configure(
        edit_url_map={
            "maze_dataset": "https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/",
        },
        template_directory=(
            Path("docs/templates/html/")
            if not parsed_args.combined
            else Path("docs/templates/markdown/")
        ),
        show_source=True,
        math=True,
        mermaid=True,
        search=True,
    )

    if not parsed_args.combined:
        pdoc.pdoc(
            "maze_dataset",
            output_directory=OUTPUT_DIR,
        )
    else:
        use_markdown_format()
        pdoc_combined(
            "maze_dataset", output_file=OUTPUT_DIR / "combined" / "maze_dataset.md"
        )

    if parsed_args.serve:
        import http.server
        import os
        import socketserver

        port: int = 8000
        os.chdir(OUTPUT_DIR)
        with socketserver.TCPServer(
            ("", port), http.server.SimpleHTTPRequestHandler
        ) as httpd:
            print(f"Serving at http://localhost:{port}")
            httpd.serve_forever()
