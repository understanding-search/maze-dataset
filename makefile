NOTEBOOKS_DIR := notebooks
CONVERTED_NOTEBOOKS_TEMP_DIR := tests/_temp/notebooks
POETRY_RUN_PYTHON := poetry run python
COVERAGE_REPORTS_DIR := docs/coverage


.PHONY: default
default: help

# format and lint
# --------------------------------------------------

.PHONY: lint
lint: clean
	@echo "run linting: mypy"
	$(POETRY_RUN_PYTHON) -m mypy --config-file pyproject.toml maze_dataset/
	$(POETRY_RUN_PYTHON) -m mypy --config-file pyproject.toml tests/


.PHONY: format
format: clean
	@echo "run formatting: pycln, isort, and black"
	$(POETRY_RUN_PYTHON) -m pycln --config pyproject.toml --all .
	$(POETRY_RUN_PYTHON) -m isort format .
	$(POETRY_RUN_PYTHON) -m black .


.PHONY: check-format
check-format: clean
	@echo "check formatting"
	$(POETRY_RUN_PYTHON) -m pycln --config pyproject.toml --check --all .
	$(POETRY_RUN_PYTHON) -m isort --check-only .
	$(POETRY_RUN_PYTHON) -m black --check .

# testing
# --------------------------------------------------

.PHONY: unit
unit:
	@echo "run unit tests"
	$(POETRY_RUN_PYTHON) -m pytest --cov=. tests/unit


.PHONY: convert_notebooks
convert_notebooks:
	@echo "convert notebooks in $(NOTEBOOKS_DIR) using muutils.nbutils.convert_ipynb_to_script.py"
	$(POETRY_RUN_PYTHON) -m muutils.nbutils.convert_ipynb_to_script $(NOTEBOOKS_DIR) --output_dir $(CONVERTED_NOTEBOOKS_TEMP_DIR) --disable_plots


.PHONY: test_notebooks
test_notebooks: convert_notebooks
	@echo "run tests on converted notebooks in $(CONVERTED_NOTEBOOKS_TEMP_DIR) using muutils.nbutils.run_notebook_tests.py"
	$(POETRY_RUN_PYTHON) -m muutils.nbutils.run_notebook_tests --notebooks-dir=$(NOTEBOOKS_DIR) --converted-notebooks-temp-dir=$(CONVERTED_NOTEBOOKS_TEMP_DIR)


.PHONY: test
test: clean unit test_notebooks
	@echo "ran all tests: unit, integration, and notebooks"

# coverage reports
# --------------------------------------------------
.PHONY: cov
cov:
	@echo "generate text coverage report"
	$(POETRY_RUN_PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(POETRY_RUN_PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg

.PHONY: cov-html
cov-html:
	@echo "generate html coverage report"
	$(POETRY_RUN_PYTHON) -m coverage html	

# general util
# --------------------------------------------------

.PHONY: clean
clean:
	@echo "cleaning up caches and temp files"
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
	rm -rf tests/_temp
	python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"


# listing targets, from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
.PHONY: help
help:
	@echo -n "# list make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30