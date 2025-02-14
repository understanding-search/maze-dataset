#|==================================================================|
#| python project makefile template                                 |
#| originally by Michael Ivanitskiy (mivanits@umich.edu)            |
#| https://github.com/mivanit/python-project-makefile-template      |
#| version: v0.2.0                                                  |
#| license: https://creativecommons.org/licenses/by-sa/4.0/         |
#| modifications from the original should be denoted with `~~~~~`   |
#| as this makes it easier to find edits when updating makefile     |
#|==================================================================|


 ######  ########  ######
##    ## ##       ##    ##
##       ##       ##
##       ######   ##   ####
##       ##       ##    ##
##    ## ##       ##    ##
 ######  ##        ######

# ==================================================
# configuration & variables
# ==================================================

# it assumes that the source is in a directory named the same as the package name
# this also gets passed to some other places
PACKAGE_NAME := maze_dataset

# for checking you are on the right branch when publishing
PUBLISH_BRANCH := main

# where to put docs
# if you change this, you must also change pyproject.toml:tool.makefile.docs.output_dir to match
DOCS_DIR := docs

# where the tests are, for pytest
TESTS_DIR := tests

# tests temp directory to clean up. will remove this in `make clean`
TESTS_TEMP_DIR := $(TESTS_DIR)/_temp/

# probably don't change these:
# --------------------------------------------------

# where the pyproject.toml file is. no idea why you would change this but just in case
PYPROJECT := pyproject.toml

# dir to store various configuration files
# use of `.meta/` inspired by https://news.ycombinator.com/item?id=36472613
META_DIR := .meta

# requirements.txt files for base package, all extras, dev, and all
REQUIREMENTS_DIR := $(META_DIR)/requirements

# local files (don't push this to git!)
LOCAL_DIR := $(META_DIR)/local

# will print this token when publishing. make sure not to commit this file!!!
PYPI_TOKEN_FILE := $(LOCAL_DIR)/.pypi-token

# version files
VERSIONS_DIR := $(META_DIR)/versions

# the last version that was auto-uploaded. will use this to create a commit log for version tag
# see `gen-commit-log` target
LAST_VERSION_FILE := $(VERSIONS_DIR)/.lastversion

# current version (writing to file needed due to shell escaping issues)
VERSION_FILE := $(VERSIONS_DIR)/.version

# base python to use. Will add `uv run` in front of this if `RUN_GLOBAL` is not set to 1
PYTHON_BASE := python

# where the commit log will be stored
COMMIT_LOG_FILE := $(LOCAL_DIR)/.commit_log

# pandoc commands (for docs)
PANDOC ?= pandoc

# where to put the coverage reports
# note that this will be published with the docs!
# modify the `docs` targets and `.gitignore` if you don't want that
COVERAGE_REPORTS_DIR := $(DOCS_DIR)/coverage

# this stuff in the docs will be kept
# in addition to anything specified in `pyproject.toml:tool.makefile.docs.no_clean`
DOCS_RESOURCES_DIR := $(DOCS_DIR)/resources

# location of the make docs script
MAKE_DOCS_SCRIPT_PATH := $(DOCS_RESOURCES_DIR)/make_docs.py

# version vars - extracted automatically from `pyproject.toml`, `$(LAST_VERSION_FILE)`, and $(PYTHON)
# --------------------------------------------------

# assuming your `pyproject.toml` has a line that looks like `version = "0.0.1"`, `gen-version-info` will extract this
PROJ_VERSION := NULL
# `gen-version-info` will read the last version from `$(LAST_VERSION_FILE)`, or `NULL` if it doesn't exist
LAST_VERSION := NULL
# get the python version, now that we have picked the python command
PYTHON_VERSION := NULL


# ==================================================
# reading command line options
# ==================================================

# for formatting or something, we might want to run python without uv
# RUN_GLOBAL=1 to use global `PYTHON_BASE` instead of `uv run $(PYTHON_BASE)`
RUN_GLOBAL ?= 0

ifeq ($(RUN_GLOBAL),0)
	PYTHON = uv run $(PYTHON_BASE)
else
	PYTHON = $(PYTHON_BASE)
endif

# if you want different behavior for different python versions
# --------------------------------------------------
# COMPATIBILITY_MODE := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info < (3, 10) else 0)")

# options we might want to pass to pytest
# --------------------------------------------------

# base options for pytest, will be appended to if `COV` or `VERBOSE` are 1.
# user can also set this when running make to add more options
PYTEST_OPTIONS ?=

# set to `1` to run pytest with `--cov=.` to get coverage reports in a `.coverage` file
COV ?= 1
# set to `1` to run pytest with `--verbose`
VERBOSE ?= 0

ifeq ($(VERBOSE),1)
	PYTEST_OPTIONS += --verbose
endif

ifeq ($(COV),1)
	PYTEST_OPTIONS += --cov=.
endif

# ==================================================
# default target (help)
# ==================================================

# first/default target is help
.PHONY: default
default: help



 ######   ######  ########  #### ########  ########  ######
##    ## ##    ## ##     ##  ##  ##     ##    ##    ##    ##
##       ##       ##     ##  ##  ##     ##    ##    ##
 ######  ##       ########   ##  ########     ##     ######
      ## ##       ##   ##    ##  ##           ##          ##
##    ## ##    ## ##    ##   ##  ##           ##    ##    ##
 ######   ######  ##     ## #### ##           ##     ######

# ==================================================
# python scripts we want to use inside the makefile
# when developing, these are populated by `scripts/assemble_make.py`
# ==================================================

# create commands for exporting requirements as specified in `pyproject.toml:tool.uv-exports.exports`
define SCRIPT_EXPORT_REQUIREMENTS
import sys
import warnings

try:
	import tomllib  # Python 3.11+
except ImportError:
	import tomli as tomllib
from pathlib import Path
from typing import Any, Union, List
from functools import reduce

TOOL_PATH: str = "tool.makefile.uv-exports"


def deep_get(d: dict, path: str, default: Any = None, sep: str = ".") -> Any:
	return reduce(
		lambda x, y: x.get(y, default) if isinstance(x, dict) else default,  # function
		path.split(sep) if isinstance(path, str) else path,  # sequence
		d,  # initial
	)


def export_configuration(
	export: dict,
	all_groups: List[str],
	all_extras: List[str],
	export_opts: dict,
	output_dir: Path,
):
	# get name and validate
	name = export.get("name")
	if not name or not name.isalnum():
		warnings.warn(
			f"Export configuration missing valid 'name' field {export}",
			file=sys.stderr,
		)
		return

	# get other options with default fallbacks
	filename: str = export.get("filename") or f"requirements-{name}.txt"
	groups: Union[List[str], bool, None] = export.get("groups", None)
	extras: Union[List[str], bool] = export.get("extras", [])
	options: List[str] = export.get("options", [])

	# init command
	cmd: List[str] = ["uv", "export"] + export_opts.get("args", [])

	# handle groups
	if groups is not None:
		groups_list: List[str] = []
		if isinstance(groups, bool):
			if groups:
				groups_list = all_groups.copy()
		else:
			groups_list = groups

		for group in all_groups:
			if group in groups_list:
				cmd.extend(["--group", group])
			else:
				cmd.extend(["--no-group", group])

	# handle extras
	extras_list: List[str] = []
	if isinstance(extras, bool):
		if extras:
			extras_list = all_extras.copy()
	else:
		extras_list = extras

	for extra in extras_list:
		cmd.extend(["--extra", extra])

	# add extra options
	cmd.extend(options)

	# assemble the command and print to console -- makefile will run it
	output_path = output_dir / filename
	print(f"{' '.join(cmd)} > {output_path.as_posix()}")


def main(
	pyproject_path: Path,
	output_dir: Path,
):
	# read pyproject.toml
	with open(pyproject_path, "rb") as f:
		pyproject_data: dict = tomllib.load(f)

	# all available groups
	all_groups: List[str] = list(pyproject_data.get("dependency-groups", {}).keys())
	all_extras: List[str] = list(
		deep_get(pyproject_data, "project.optional-dependencies", {}).keys()
	)

	# options for exporting
	export_opts: dict = deep_get(pyproject_data, TOOL_PATH, {})

	# what are we exporting?
	exports: List[str] = export_opts.get("exports", [])
	if not exports:
		exports = [{"name": "all", "groups": [], "extras": [], "options": []}]

	# export each configuration
	for export in exports:
		export_configuration(
			export=export,
			all_groups=all_groups,
			all_extras=all_extras,
			export_opts=export_opts,
			output_dir=output_dir,
		)


if __name__ == "__main__":
	main(
		pyproject_path=Path(sys.argv[1]),
		output_dir=Path(sys.argv[2]),
	)

endef

export SCRIPT_EXPORT_REQUIREMENTS


# get the version from `pyproject.toml:project.version`
define SCRIPT_GET_VERSION
import sys

try:
	try:
		import tomllib  # Python 3.11+
	except ImportError:
		import tomli as tomllib

	pyproject_path: str = sys.argv[1].strip()

	with open(pyproject_path, "rb") as f:
		pyproject_data: dict = tomllib.load(f)

	print("v" + pyproject_data["project"]["version"], end="")
except Exception:
	print("NULL", end="")
	sys.exit(1)

endef

export SCRIPT_GET_VERSION


# get the commit log since the last version from `$(LAST_VERSION_FILE)`
define SCRIPT_GET_COMMIT_LOG
import subprocess
import sys
from typing import List


def main(
	last_version: str,
	commit_log_file: str,
):
	if last_version == "NULL":
		print("!!! ERROR !!!", file=sys.stderr)
		print("LAST_VERSION is NULL, can't get commit log!", file=sys.stderr)
		sys.exit(1)

	try:
		log_cmd: List[str] = [
			"git",
			"log",
			f"{last_version}..HEAD",
			"--pretty=format:- %s (%h)",
		]
		commits: List[str] = (
			subprocess.check_output(log_cmd).decode("utf-8").strip().split("\n")
		)
		with open(commit_log_file, "w") as f:
			f.write("\n".join(reversed(commits)))
	except subprocess.CalledProcessError as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main(
		last_version=sys.argv[1].strip(),
		commit_log_file=sys.argv[2].strip(),
	)

endef

export SCRIPT_GET_COMMIT_LOG


# get cuda information and whether torch sees it
define SCRIPT_CHECK_TORCH
import os
import sys
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def print_info_dict(
	info: Dict[str, Union[Any, Dict[str, Any]]],
	indent: str = "  ",
	level: int = 1,
) -> None:
	indent_str: str = indent * level
	longest_key_len: int = max(map(len, info.keys()))
	for key, value in info.items():
		if isinstance(value, dict):
			print(f"{indent_str}{key:<{longest_key_len}}:")
			print_info_dict(value, indent, level + 1)
		else:
			print(f"{indent_str}{key:<{longest_key_len}} = {value}")


def get_nvcc_info() -> Dict[str, str]:
	# Run the nvcc command.
	try:
		result: subprocess.CompletedProcess[str] = subprocess.run(
			["nvcc", "--version"],
			check=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
	except Exception as e:
		return {"Failed to run 'nvcc --version'": str(e)}

	output: str = result.stdout
	lines: List[str] = [line.strip() for line in output.splitlines() if line.strip()]

	# Ensure there are exactly 5 lines in the output.
	assert len(lines) == 5, (
		f"Expected exactly 5 lines from nvcc --version, got {len(lines)} lines:\n{output}"
	)

	# Compile shared regex for release info.
	release_regex: re.Pattern = re.compile(
		r"Cuda compilation tools,\s*release\s*([^,]+),\s*(V.+)"
	)

	# Define a mapping for each desired field:
	# key -> (line index, regex pattern, group index, transformation function)
	patterns: Dict[str, Tuple[int, re.Pattern, int, Callable[[str], str]]] = {
		"build_time": (
			2,
			re.compile(r"Built on (.+)"),
			1,
			lambda s: s.replace("_", " "),
		),
		"release": (3, release_regex, 1, str.strip),
		"release_V": (3, release_regex, 2, str.strip),
		"build": (4, re.compile(r"Build (.+)"), 1, str.strip),
	}

	info: Dict[str, str] = {}
	for key, (line_index, pattern, group_index, transform) in patterns.items():
		match: Optional[re.Match] = pattern.search(lines[line_index])
		if not match:
			raise ValueError(
				f"Unable to parse {key} from nvcc output: {lines[line_index]}"
			)
		info[key] = transform(match.group(group_index))

	info["release_short"] = info["release"].replace(".", "").strip()

	return info


def get_torch_info() -> Tuple[List[Exception], Dict[str, str]]:
	exceptions: List[Exception] = []
	info: Dict[str, str] = {}

	try:
		import torch

		info["torch.__version__"] = torch.__version__
		info["torch.cuda.is_available()"] = torch.cuda.is_available()

		if torch.cuda.is_available():
			info["torch.version.cuda"] = torch.version.cuda
			info["torch.cuda.device_count()"] = torch.cuda.device_count()

			if torch.cuda.device_count() > 0:
				info["torch.cuda.current_device()"] = torch.cuda.current_device()
				n_devices: int = torch.cuda.device_count()
				info["n_devices"] = n_devices
				for current_device in range(n_devices):
					try:
						current_device_info: Dict[str, str] = {}

						dev_prop = torch.cuda.get_device_properties(
							torch.device(f"cuda:{current_device}")
						)

						current_device_info["name"] = dev_prop.name
						current_device_info["version"] = (
							f"{dev_prop.major}.{dev_prop.minor}"
						)
						current_device_info["total_memory"] = (
							f"{dev_prop.total_memory} ({dev_prop.total_memory:.1e})"
						)
						current_device_info["multi_processor_count"] = (
							dev_prop.multi_processor_count
						)
						current_device_info["is_integrated"] = dev_prop.is_integrated
						current_device_info["is_multi_gpu_board"] = (
							dev_prop.is_multi_gpu_board
						)

						info[f"device cuda:{current_device}"] = current_device_info

					except Exception as e:
						exceptions.append(e)
			else:
				raise Exception(
					f"{torch.cuda.device_count() = } devices detected, invalid"
				)

		else:
			raise Exception(
				f"CUDA is NOT available in torch: {torch.cuda.is_available() =}"
			)

	except Exception as e:
		exceptions.append(e)

	return exceptions, info


if __name__ == "__main__":
	print(f"python: {sys.version}")
	print_info_dict(
		{
			"python executable path: sys.executable": str(sys.executable),
			"sys.platform": sys.platform,
			"current working directory: os.getcwd()": os.getcwd(),
			"Host name: os.name": os.name,
			"CPU count: os.cpu_count()": str(os.cpu_count()),
		}
	)

	nvcc_info: Dict[str, str] = get_nvcc_info()
	print("nvcc:")
	print_info_dict(nvcc_info)

	torch_exceptions, torch_info = get_torch_info()
	print("torch:")
	print_info_dict(torch_info)

	if torch_exceptions:
		print("torch_exceptions:")
		for e in torch_exceptions:
			print(f"  {e}")

endef

export SCRIPT_CHECK_TORCH


# get todo's from the code
define SCRIPT_GET_TODOS
from __future__ import annotations

import urllib.parse
import argparse
import fnmatch
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Union
from functools import reduce
import warnings
from jinja2 import Template

try:
	import tomllib  # Python 3.11+
except ImportError:
	import tomli as tomllib

TOOL_PATH: str = "tool.makefile.inline-todo"


def deep_get(d: dict, path: str, default: Any = None, sep: str = ".") -> Any:
	return reduce(
		lambda x, y: x.get(y, default) if isinstance(x, dict) else default,  # function
		path.split(sep) if isinstance(path, str) else path,  # sequence
		d,  # initial
	)


TEMPLATE_MD: str = """\
# Inline TODOs

{% for tag, file_map in grouped|dictsort %}
# {{ tag }}
{% for filepath, item_list in file_map|dictsort %}
## [`{{ filepath }}`](/{{ filepath }})
{% for itm in item_list %}
- {{ itm.stripped_title }}  
  local link: [`/{{ filepath }}#{{ itm.line_num }}`](/{{ filepath }}#{{ itm.line_num }}) 
  | view on GitHub: [{{ itm.file }}#L{{ itm.line_num }}]({{ itm.code_url | safe }})
  | [Make Issue]({{ itm.issue_url | safe }})
{% if itm.context %}
  ```{{ itm.file_lang }}
{{ itm.context.strip() }}
  ```
{% endif %}
{% endfor %}

{% endfor %}
{% endfor %}
"""

TEMPLATE_ISSUE: str = """\
# source

[`{file}#L{line_num}`]({code_url})

# context
```{file_lang}
{context}
```
"""


@dataclass
class Config:
	"""Configuration for the inline-todo scraper"""

	search_dir: Path = Path(".")
	out_file: Path = Path("docs/todo-inline.md")
	tags: List[str] = field(
		default_factory=lambda: ["CRIT", "TODO", "FIXME", "HACK", "BUG"]
	)
	extensions: List[str] = field(default_factory=lambda: ["py", "md"])
	exclude: List[str] = field(default_factory=lambda: ["docs/**", ".venv/**"])
	context_lines: int = 2
	tag_label_map: Dict[str, str] = field(
		default_factory=lambda: {
			"CRIT": "bug",
			"TODO": "enhancement",
			"FIXME": "bug",
			"BUG": "bug",
			"HACK": "enhancement",
		}
	)
	extension_lang_map: Dict[str, str] = field(
		default_factory=lambda: {
			"py": "python",
			"md": "markdown",
			"html": "html",
			"css": "css",
			"js": "javascript",
		}
	)

	template_md: str = TEMPLATE_MD
	# template for the output markdown file

	template_issue: str = TEMPLATE_ISSUE
	# template for the issue creation

	template_html_source: Path = Path("docs/resources/templates/todo-template.html")
	# template source for the output html file (interactive table)

	@property
	def template_html(self) -> str:
		return self.template_html_source.read_text(encoding="utf-8")

	template_code_url_: str = "{repo_url}/blob/{branch}/{file}#L{line_num}"
	# template for the code url

	@property
	def template_code_url(self) -> str:
		return self.template_code_url_.replace("{repo_url}", self.repo_url).replace(
			"{branch}", self.branch
		)

	repo_url: str = "UNKNOWN"
	# for the issue creation url

	branch: str = "main"
	# branch for links to files on github

	@classmethod
	def read(cls, config_file: Path) -> Config:
		output: Config
		if config_file.is_file():
			# read file and load if present
			with config_file.open("rb") as f:
				data: Dict[str, Any] = tomllib.load(f)

			# try to get the repo url
			repo_url: str = "UNKNOWN"
			try:
				urls: Dict[str, str] = {
					k.lower(): v for k, v in data["project"]["urls"].items()
				}
				if "repository" in urls:
					repo_url = urls["repository"]
				if "github" in urls:
					repo_url = urls["github"]
			except Exception as e:
				warnings.warn(
					f"No repository URL found in pyproject.toml, 'make issue' links will not work.\n{e}"
				)

			# load the inline-todo config if present
			data_inline_todo: Dict[str, Any] = deep_get(
				d=data, path=TOOL_PATH, default={}
			)

			if "repo_url" not in data_inline_todo:
				data_inline_todo["repo_url"] = repo_url

			output = cls.load(data_inline_todo)
		else:
			# return default otherwise
			output = cls()

		return output

	@classmethod
	def load(cls, data: dict) -> Config:
		data = {
			k: Path(v) if k in {"search_dir", "out_file", "template_html_source"} else v
			for k, v in data.items()
		}

		return cls(**data)


CFG: Config = Config()
# this is messy, but we use a global config so we can get `TodoItem().issue_url` to work


@dataclass
class TodoItem:
	"""Holds one todo occurrence"""

	tag: str
	file: str
	line_num: int
	content: str
	context: str = ""

	def serialize(self) -> Dict[str, Union[str, int]]:
		return {
			**asdict(self),
			"issue_url": self.issue_url,
			"file_lang": self.file_lang,
			"stripped_title": self.stripped_title,
			"code_url": self.code_url,
		}

	@property
	def code_url(self) -> str:
		"""Returns a URL to the code on GitHub"""
		return CFG.template_code_url.format(
			file=self.file,
			line_num=self.line_num,
		)

	@property
	def stripped_title(self) -> str:
		"""Returns the title of the issue, stripped of the tag"""
		return self.content.split(self.tag, 1)[-1].lstrip(":").strip()

	@property
	def issue_url(self) -> str:
		"""Constructs a GitHub issue creation URL for a given TodoItem."""
		# title
		title: str = self.stripped_title
		if not title:
			title = "Issue from inline todo"
		# body
		body: str = CFG.template_issue.format(
			file=self.file,
			line_num=self.line_num,
			context=self.context,
			code_url=self.code_url,
			file_lang=self.file_lang,
		).strip()
		# labels
		label: str = CFG.tag_label_map.get(self.tag, self.tag)
		# assemble url
		query: Dict[str, str] = dict(title=title, body=body, labels=label)
		query_string: str = urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
		return f"{CFG.repo_url}/issues/new?{query_string}"

	@property
	def file_lang(self) -> str:
		"""Returns the language for the file extension"""
		ext: str = Path(self.file).suffix.lstrip(".")
		return CFG.extension_lang_map.get(ext, ext)


def scrape_file(
	file_path: Path,
	tags: List[str],
	context_lines: int,
) -> List[TodoItem]:
	"""Scrapes a file for lines containing any of the specified tags"""
	items: List[TodoItem] = []
	if not file_path.is_file():
		return items
	lines: List[str] = file_path.read_text(encoding="utf-8").splitlines(True)

	for i, line in enumerate(lines):
		for tag in tags:
			if tag in line[:200]:
				start: int = max(0, i - context_lines)
				end: int = min(len(lines), i + context_lines + 1)
				snippet: str = "".join(lines[start:end])
				items.append(
					TodoItem(
						tag=tag,
						file=file_path.as_posix(),
						line_num=i + 1,
						content=line.strip("\n"),
						context=snippet.strip("\n"),
					)
				)
				break
	return items


def collect_files(
	search_dir: Path,
	extensions: List[str],
	exclude: List[str],
) -> List[Path]:
	"""Recursively collects all files with specified extensions, excluding matches via globs"""
	results: List[Path] = []
	for ext in extensions:
		results.extend(search_dir.rglob(f"*.{ext}"))

	filtered: List[Path] = []
	for f in results:
		# Skip if it matches any exclude glob
		if not any(fnmatch.fnmatch(f.as_posix(), pattern) for pattern in exclude):
			filtered.append(f)
	return filtered


def group_items_by_tag_and_file(
	items: List[TodoItem],
) -> Dict[str, Dict[str, List[TodoItem]]]:
	"""Groups items by tag, then by file"""
	grouped: Dict[str, Dict[str, List[TodoItem]]] = {}
	for itm in items:
		grouped.setdefault(itm.tag, {}).setdefault(itm.file, []).append(itm)
	for tag_dict in grouped.values():
		for file_list in tag_dict.values():
			file_list.sort(key=lambda x: x.line_num)
	return grouped


def main(config_file: Path) -> None:
	global CFG
	# read configuration
	cfg: Config = Config.read(config_file)
	CFG = cfg

	# get data
	files: List[Path] = collect_files(cfg.search_dir, cfg.extensions, cfg.exclude)
	all_items: List[TodoItem] = []
	n_files: int = len(files)
	for i, fpath in enumerate(files):
		print(f"Scraping {i + 1:>2}/{n_files:>2}: {fpath.as_posix():<60}", end="\r")
		all_items.extend(scrape_file(fpath, cfg.tags, cfg.context_lines))

	# create dir
	cfg.out_file.parent.mkdir(parents=True, exist_ok=True)

	# write raw to jsonl
	with open(cfg.out_file.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
		for itm in all_items:
			f.write(json.dumps(itm.serialize()) + "\n")

	# group, render
	grouped: Dict[str, Dict[str, List[TodoItem]]] = group_items_by_tag_and_file(
		all_items
	)

	rendered: str = Template(cfg.template_md).render(grouped=grouped)

	# write md output
	cfg.out_file.with_suffix(".md").write_text(rendered, encoding="utf-8")

	# write html output
	try:
		html_rendered: str = cfg.template_html.replace(
			"//{{DATA}}//", json.dumps([itm.serialize() for itm in all_items])
		)
		cfg.out_file.with_suffix(".html").write_text(html_rendered, encoding="utf-8")
	except Exception as e:
		warnings.warn(f"Failed to write html output: {e}")

	print("wrote to:")
	print(cfg.out_file.with_suffix(".md").as_posix())


if __name__ == "__main__":
	# parse args
	parser: argparse.ArgumentParser = argparse.ArgumentParser("inline_todo")
	parser.add_argument(
		"--config-file",
		default="pyproject.toml",
		help="Path to the TOML config, will look under [tool.inline-todo].",
	)
	args: argparse.Namespace = parser.parse_args()
	# call main
	main(Path(args.config_file))

endef

export SCRIPT_GET_TODOS


# markdown to html using pdoc
define SCRIPT_PDOC_MARKDOWN2_CLI
import argparse
from pathlib import Path
from typing import Optional

from pdoc.markdown2 import Markdown, _safe_mode


def convert_file(
	input_path: Path,
	output_path: Path,
	safe_mode: Optional[_safe_mode] = None,
	encoding: str = "utf-8",
) -> None:
	"""Convert a markdown file to HTML"""
	# Read markdown input
	text: str = input_path.read_text(encoding=encoding)

	# Convert to HTML using markdown2
	markdown: Markdown = Markdown(
		extras=["fenced-code-blocks", "header-ids", "markdown-in-html", "tables"],
		safe_mode=safe_mode,
	)
	html: str = markdown.convert(text)

	# Write HTML output
	output_path.write_text(str(html), encoding=encoding)


def main() -> None:
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Convert markdown files to HTML using pdoc's markdown2"
	)
	parser.add_argument("input", type=Path, help="Input markdown file path")
	parser.add_argument("output", type=Path, help="Output HTML file path")
	parser.add_argument(
		"--safe-mode",
		choices=["escape", "replace"],
		help="Sanitize literal HTML: 'escape' escapes HTML meta chars, 'replace' replaces with [HTML_REMOVED]",
	)
	parser.add_argument(
		"--encoding",
		default="utf-8",
		help="Character encoding for reading/writing files (default: utf-8)",
	)

	args: argparse.Namespace = parser.parse_args()

	convert_file(
		args.input, args.output, safe_mode=args.safe_mode, encoding=args.encoding
	)


if __name__ == "__main__":
	main()

endef

export SCRIPT_PDOC_MARKDOWN2_CLI

# clean up the docs (configurable in pyproject.toml)
define SCRIPT_DOCS_CLEAN
import sys
import shutil
from functools import reduce
from pathlib import Path
from typing import Any, List, Set

try:
	import tomllib  # Python 3.11+
except ImportError:
	import tomli as tomllib

TOOL_PATH: str = "tool.makefile.docs"


def deep_get(d: dict, path: str, default: Any = None, sep: str = ".") -> Any:
	"""Get nested dictionary value via separated path with default."""
	return reduce(
		lambda x, y: x.get(y, default) if isinstance(x, dict) else default,  # function
		path.split(sep) if isinstance(path, str) else path,  # sequence
		d,  # initial
	)


def read_config(pyproject_path: Path) -> tuple[Path, Set[Path]]:
	if not pyproject_path.is_file():
		return set()

	with pyproject_path.open("rb") as f:
		config = tomllib.load(f)

	preserved: List[str] = deep_get(config, f"{TOOL_PATH}.no_clean", [])
	docs_dir: Path = Path(deep_get(config, f"{TOOL_PATH}.output_dir", "docs"))

	# Convert to absolute paths and validate
	preserve_set: Set[Path] = set()
	for p in preserved:
		full_path = (docs_dir / p).resolve()
		if not full_path.as_posix().startswith(docs_dir.resolve().as_posix()):
			raise ValueError(f"Preserved path '{p}' must be within docs directory")
		preserve_set.add(docs_dir / p)

	return docs_dir, preserve_set


def clean_docs(docs_dir: Path, preserved: Set[Path]) -> None:
	for path in docs_dir.iterdir():
		if path.is_file() and path not in preserved:
			path.unlink()
		elif path.is_dir() and path not in preserved:
			shutil.rmtree(path)


def main(
	pyproject_path: str,
	docs_dir_cli: str,
	extra_preserve: list[str],
) -> None:
	docs_dir: Path
	preserved: Set[Path]
	docs_dir, preserved = read_config(Path(pyproject_path))

	assert docs_dir.is_dir(), f"Docs directory '{docs_dir}' not found"
	assert docs_dir == Path(docs_dir_cli), (
		f"Docs directory mismatch: {docs_dir = } != {docs_dir_cli = }. this is probably because you changed one of `pyproject.toml:{TOOL_PATH}.output_dir` (the former) or `makefile:DOCS_DIR` (the latter) without updating the other."
	)

	for x in extra_preserve:
		preserved.add(Path(x))
	clean_docs(docs_dir, preserved)


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3:])

endef

export SCRIPT_DOCS_CLEAN


##     ## ######## ########   ######  ####  #######  ##    ##
##     ## ##       ##     ## ##    ##  ##  ##     ## ###   ##
##     ## ##       ##     ## ##        ##  ##     ## ####  ##
##     ## ######   ########   ######   ##  ##     ## ## ## ##
 ##   ##  ##       ##   ##         ##  ##  ##     ## ##  ####
  ## ##   ##       ##    ##  ##    ##  ##  ##     ## ##   ###
   ###    ######## ##     ##  ######  ####  #######  ##    ##

# ==================================================
# getting version info
# we do this in a separate target because it takes a bit of time
# ==================================================

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$SCRIPT_GET_VERSION within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	@mkdir -p $(VERSIONS_DIR)
	@$(PYTHON) -c "$$SCRIPT_GET_VERSION" "$(PYPROJECT)" > $(VERSION_FILE)

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	@mkdir -p $(LOCAL_DIR)
	$(eval PROJ_VERSION := $(shell cat $(VERSION_FILE)) )
	$(eval LAST_VERSION := $(shell [ -f $(LAST_VERSION_FILE) ] && cat $(LAST_VERSION_FILE) || echo NULL) )
	$(eval PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") )

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	@if [ "$(LAST_VERSION)" = "NULL" ]; then \
		echo "!!! ERROR !!!"; \
		echo "LAST_VERSION is NULL, cant get commit log!"; \
		exit 1; \
	fi
	@mkdir -p $(LOCAL_DIR)
	@$(PYTHON) -c "$$SCRIPT_GET_COMMIT_LOG" "$(LAST_VERSION)" "$(COMMIT_LOG_FILE)"


# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(PROJ_VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version from '$(COMMIT_LOG_FILE)':"
	@cat $(COMMIT_LOG_FILE)
	@echo ""
	@if [ "$(PROJ_VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Python package $(PROJ_VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi



########  ######## ########   ######
##     ## ##       ##     ## ##    ##
##     ## ##       ##     ## ##
##     ## ######   ########   ######
##     ## ##       ##              ##
##     ## ##       ##        ##    ##
########  ######## ##         ######

# ==================================================
# dependencies and setup
# ==================================================

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	@echo "To activate the virtual environment, run one of:"
	@echo "  source .venv/bin/activate"
	@echo "  source .venv/Scripts/activate"

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	$(PYTHON) -c "$$SCRIPT_CHECK_TORCH"

.PHONY: dep
dep:
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	uv sync --all-extras --all-groups --compile-bytecode
	mkdir -p $(REQUIREMENTS_DIR)
	$(PYTHON) -c "$$SCRIPT_EXPORT_REQUIREMENTS" $(PYPROJECT) $(REQUIREMENTS_DIR) | sh -x
	

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	uv sync --all-extras --all-groups
	mkdir -p $(REQUIREMENTS_DIR)-TEMP
	$(PYTHON) -c "$$SCRIPT_EXPORT_REQUIREMENTS" $(PYPROJECT) $(REQUIREMENTS_DIR)-TEMP | sh -x
	diff -r $(REQUIREMENTS_DIR)-TEMP $(REQUIREMENTS_DIR)
	rm -rf $(REQUIREMENTS_DIR)-TEMP


.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	rm -rf .venv
	rm -rf uv.lock
	rm -rf $(REQUIREMENTS_DIR)/*.txt


 ######  ##     ## ########  ######  ##    ##  ######
##    ## ##     ## ##       ##    ## ##   ##  ##    ##
##       ##     ## ##       ##       ##  ##   ##
##       ######### ######   ##       #####     ######
##       ##     ## ##       ##       ##  ##         ##
##    ## ##     ## ##       ##    ## ##   ##  ##    ##
 ######  ##     ## ########  ######  ##    ##  ######

# ==================================================
# checks (formatting/linting, typing, tests)
# ==================================================

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	$(PYTHON) -m ruff format --config $(PYPROJECT) .
	$(PYTHON) -m ruff check --fix --config $(PYPROJECT) .
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	$(PYTHON) -m ruff check --config $(PYPROJECT) .
	$(PYTHON) -m pycln --check --config $(PYPROJECT) .

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(PACKAGE_NAME)/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(TESTS_DIR)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# separate unit and notebook tests
.PHONY: unit
unit:
	@echo "run unit tests"
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) tests/unit

.PHONY: save_tok_hashes
save_tok_hashes:
	@echo "generate and save tokenizer hashes"
	$(PYTHON) -m maze_dataset.tokenization.save_hashes -p

.PHONY: test_tok_hashes
test_tok_hashes:
	@echo "re-run tokenizer hashes and compare"
	$(PYTHON) -m maze_dataset.tokenization.save_hashes -p --check


.PHONY: test_all_tok
test_all_tok:
	@echo "run tests on all tokenizers. can pass NUM_TOKENIZERS_TO_TEST arg or SKIP_HASH_TEST"
	@echo "NUM_TOKENIZERS_TO_TEST=$(NUM_TOKENIZERS_TO_TEST)"
	@if [ "$(SKIP_HASH_TEST)" != "1" ]; then \
		echo "Running tokenizer hash tests"; \
		$(MAKE) test_tok_hashes; \
	else \
		echo "Skipping tokenizer hash tests"; \
	fi
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) --verbosity=-1 --durations=50 tests/all_tokenizers



.PHONY: convert_notebooks
convert_notebooks:
	@echo "convert notebooks in $(NOTEBOOKS_DIR) using muutils.nbutils.convert_ipynb_to_script.py"
	$(PYTHON) -m muutils.nbutils.convert_ipynb_to_script $(NOTEBOOKS_DIR) --output_dir $(CONVERTED_NOTEBOOKS_TEMP_DIR) --disable_plots


.PHONY: test_notebooks
test_notebooks: convert_notebooks
	@echo "run tests on converted notebooks in $(CONVERTED_NOTEBOOKS_TEMP_DIR) using muutils.nbutils.run_notebook_tests.py"
	$(PYTHON) -m muutils.nbutils.run_notebook_tests --notebooks-dir=$(NOTEBOOKS_DIR) --converted-notebooks-temp-dir=$(CONVERTED_NOTEBOOKS_TEMP_DIR)

.PHONY: test
test: clean unit test_notebooks
	@echo "ran all tests: unit, integration, and notebooks"

.PHONY: test-all
test-all: clean test test_tok_hashes test_all_tok
	@echo "run all tests, including tokenizers"

.PHONY: check
check: clean check-format typing clean test
	@echo "run format check and test"

.PHONY: check-all
check-all: clean check-format typing clean test-all
	@echo "run format check and test-all (includes tokenizers)"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


########   #######   ######   ######
##     ## ##     ## ##    ## ##    ##
##     ## ##     ## ##       ##
##     ## ##     ## ##        ######
##     ## ##     ## ##             ##
##     ## ##     ## ##    ## ##    ##
########   #######   ######   ######

# ==================================================
# coverage & docs
# ==================================================

# generates a whole tree of documentation in html format.
# see `$(MAKE_DOCS_SCRIPT_PATH)` and the templates in `$(DOCS_RESOURCES_DIR)/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) $(MAKE_DOCS_SCRIPT_PATH)

# instead of a whole website, generates a single markdown file with all docs using the templates in `$(DOCS_RESOURCES_DIR)/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) $(MAKE_DOCS_SCRIPT_PATH) --combined

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	@if [ ! -f .coverage ]; then \
		echo ".coverage not found, running tests first..."; \
		$(MAKE) test; \
	fi
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

# runs the coverage report, then the docs, then the combined docs
.PHONY: docs
docs: cov docs-html docs-combined todo lmcat
	@echo "generate all documentation and coverage reports"

# removed all generated documentation files, but leaves everything in `$DOCS_RESOURCES_DIR`
# and leaves things defined in `pyproject.toml:tool.makefile.docs.no_clean`
# (templates, svg, css, make_docs.py script)
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs except resources"
	$(PYTHON) -c "$$SCRIPT_DOCS_CLEAN" $(PYPROJECT) $(DOCS_DIR) $(DOCS_RESOURCES_DIR)

.PHONY: todo
todo:
	@echo "get all TODO's from the code"
	$(PYTHON) -c "$$SCRIPT_GET_TODOS"


.PHONY: lmcat-tree
lmcat-tree:
	@echo "show in console the lmcat tree view"
	-$(PYTHON) -m lmcat -t --output STDOUT

.PHONY: lmcat
lmcat:
	@echo "write the lmcat full output to pyproject.toml:[tool.lmcat.output]"
	-$(PYTHON) -m lmcat


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# benchmark generation

.PHONY: benchmark
benchmark:
	@echo "run benchmarks"
	$(PYTHON) docs/benchmarks/benchmark_generation.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

########  ##     ## #### ##       ########
##     ## ##     ##  ##  ##       ##     ##
##     ## ##     ##  ##  ##       ##     ##
########  ##     ##  ##  ##       ##     ##
##     ## ##     ##  ##  ##       ##     ##
##     ## ##     ##  ##  ##       ##     ##
########   #######  #### ######## ########

# ==================================================
# build and publish
# ==================================================

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		git branch; \
		git status; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not clean, exiting!"; \
		git status; \
		exit 1; \
	fi; \


.PHONY: build
build: 
	@echo "build the package"
	uv build

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@echo "Now would also be the time to edit $(COMMIT_LOG_FILE), as that will be used as the tag description"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" = $(PROJ_VERSION) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Version confirmed. Proceeding with publish."; \
	else \
		echo "Version mismatch, exiting: you gave $$NEW_VERSION but expected $(PROJ_VERSION)"; \
		exit 1; \
	fi;

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(PROJ_VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(PROJ_VERSION)"; \
	git tag -a $(PROJ_VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(PROJ_VERSION); \
	twine upload dist/* --verbose

# ==================================================
# cleanup of temp files
# ==================================================

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf $(PACKAGE_NAME).egg-info
	rm -rf $(TESTS_TEMP_DIR)
	$(PYTHON_BASE) -Bc "import pathlib; [p.unlink() for path in ['$(PACKAGE_NAME)', '$(TESTS_DIR)', '$(DOCS_DIR)'] for pattern in ['*.py[co]', '__pycache__/*'] for p in pathlib.Path(path).rglob(pattern)]"

.PHONY: clean-all
clean-all: clean docs-clean dep-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"


##     ## ######## ##       ########
##     ## ##       ##       ##     ##
##     ## ##       ##       ##     ##
######### ######   ##       ########
##     ## ##       ##       ##
##     ## ##       ##       ##
##     ## ######## ######## ##

# ==================================================
# smart help command
# ==================================================

# listing targets is from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
# no .PHONY because this will only be run before `make help`
# it's a separate command because getting the `info` takes a bit of time
# and we want to show the make targets right away without making the user wait for `info` to finish running
help-targets:
	@echo -n "# make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30


.PHONY: info
info: gen-version-info
	@echo "# makefile variables"
	@echo "    PYTHON = $(PYTHON)"
	@echo "    PYTHON_VERSION = $(PYTHON_VERSION)"
	@echo "    PACKAGE_NAME = $(PACKAGE_NAME)"
	@echo "    PROJ_VERSION = $(PROJ_VERSION)"
	@echo "    LAST_VERSION = $(LAST_VERSION)"
	@echo "    PYTEST_OPTIONS = $(PYTEST_OPTIONS)"

.PHONY: info-long
info-long: info
	@echo "# other variables"
	@echo "    PUBLISH_BRANCH = $(PUBLISH_BRANCH)"
	@echo "    DOCS_DIR = $(DOCS_DIR)"
	@echo "    COVERAGE_REPORTS_DIR = $(COVERAGE_REPORTS_DIR)"
	@echo "    TESTS_DIR = $(TESTS_DIR)"
	@echo "    TESTS_TEMP_DIR = $(TESTS_TEMP_DIR)"
	@echo "    PYPROJECT = $(PYPROJECT)"
	@echo "    REQUIREMENTS_DIR = $(REQUIREMENTS_DIR)"
	@echo "    LOCAL_DIR = $(LOCAL_DIR)"
	@echo "    PYPI_TOKEN_FILE = $(PYPI_TOKEN_FILE)"
	@echo "    LAST_VERSION_FILE = $(LAST_VERSION_FILE)"
	@echo "    PYTHON_BASE = $(PYTHON_BASE)"
	@echo "    COMMIT_LOG_FILE = $(COMMIT_LOG_FILE)"
	@echo "    PANDOC = $(PANDOC)"
	@echo "    COV = $(COV)"
	@echo "    VERBOSE = $(VERBOSE)"
	@echo "    RUN_GLOBAL = $(RUN_GLOBAL)"
	@echo "    TYPECHECK_ARGS = $(TYPECHECK_ARGS)"

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""


 ######  ##     ##  ######  ########  #######  ##     ##
##    ## ##     ## ##    ##    ##    ##     ## ###   ###
##       ##     ## ##          ##    ##     ## #### ####
##       ##     ##  ######     ##    ##     ## ## ### ##
##       ##     ##       ##    ##    ##     ## ##     ##
##    ## ##     ## ##    ##    ##    ##     ## ##     ##
 ######   #######   ######     ##     #######  ##     ##

# ==================================================
# custom targets
# ==================================================
# (put them down here, or delimit with ~~~~~)