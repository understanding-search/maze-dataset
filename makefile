PUBLISH_BRANCH := main
PYPI_TOKEN_FILE := .pypi-token
LAST_VERSION_FILE := .lastversion
COVERAGE_REPORTS_DIR := docs/coverage
PYPROJECT := pyproject.toml
POETRY_RUN_PYTHON := poetry run python

NOTEBOOKS_DIR := notebooks
CONVERTED_NOTEBOOKS_TEMP_DIR := tests/_temp/notebooks

VERSION := $(shell python -c "import tomllib; print(tomllib.load(open('$(PYPROJECT)', 'rb'))['tool']['poetry']['version'])")
LAST_VERSION := $(shell cat $(LAST_VERSION_FILE))

# note that the commands at the end:
# 1) format the git log
# 2) replace backticks with single quotes, to avoid funny business
# 3) add a final newline, to make tac happy
# 4) reverse the order of the lines, so that the oldest commit is first
# 5) replace newlines with tabs, to prevent the newlines from being lost
COMMIT_LOG_FILE := .commit_log
COMMIT_LOG_SINCE_LAST_VERSION := $(shell (git log $(LAST_VERSION)..HEAD --pretty=format:"- %s (%h)" | tr '`' "'" ; echo) | tac | tr '\n' '\t')
#                                                                                    1                2            3       4     5


.PHONY: default
default: help

.PHONY: version
version:
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version:"
	@echo "$(COMMIT_LOG_SINCE_LAST_VERSION)" | tr '\t' '\n' > $(COMMIT_LOG_FILE)
	@cat $(COMMIT_LOG_FILE)
	@if [ "$(VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "Python package $(VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi

# format and lint
# --------------------------------------------------

.PHONY: lint
lint: clean
	@echo "run linting: mypy"
	$(POETRY_RUN_PYTHON) -m mypy --config-file $(PYPROJECT) maze_dataset/
	$(POETRY_RUN_PYTHON) -m mypy --config-file $(PYPROJECT) tests/


.PHONY: format
format: clean
	@echo "run formatting: pycln, isort, and black"
	$(POETRY_RUN_PYTHON) -m pycln --config $(PYPROJECT) --all .
	$(POETRY_RUN_PYTHON) -m isort format .
	$(POETRY_RUN_PYTHON) -m black .


.PHONY: check-format
check-format: clean
	@echo "check formatting"
	$(POETRY_RUN_PYTHON) -m pycln --config $(PYPROJECT) --check --all .
	$(POETRY_RUN_PYTHON) -m isort --check-only .
	$(POETRY_RUN_PYTHON) -m black --check .


# coverage reports & benchmarks
# --------------------------------------------------
# whether to run pytest with coverage report generation
COV ?= 1

ifeq ($(COV),1)
    PYTEST_OPTIONS=--cov=.
else
    PYTEST_OPTIONS=
endif

.PHONY: cov
cov:
	@echo "generate coverage reports (run tests manually)"
	$(POETRY_RUN_PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(POETRY_RUN_PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(POETRY_RUN_PYTHON) -m coverage html	

.PHONY: benchmark
benchmark:
	@echo "run benchmarks"
	$(POETRY_RUN_PYTHON) docs/benchmarks/benchmark_generation.py


# testing
# --------------------------------------------------

.PHONY: unit
unit:
	@echo "run unit tests"
	$(POETRY_RUN_PYTHON) -m pytest $(PYTEST_OPTIONS) tests/unit

.PHONY: save_tok_hashes
save_tok_hashes:
	@echo "generate and save tokenizer hashes"
	$(POETRY_RUN_PYTHON) -m maze_dataset.tokenization.save_hashes

.PHONY: test_all_tok
test_all_tok: save_tok_hashes
	@echo "run tests on all tokenizers"
	$(POETRY_RUN_PYTHON) -m pytest $(PYTEST_OPTIONS) tests/all_tokenizers


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

.PHONY: check
check: clean check-format clean test
	@echo "run format check and test"


# build and publish
# --------------------------------------------------

.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "Git is not clean, exiting!"; \
		exit 1; \
	fi; \

.PHONY: build
build: 
	@echo "build via poetry, assumes checks have been run"
	poetry build

.PHONY: publish
publish: check build verify-git version
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" != "$(VERSION)" ]; then \
		echo "Confirmation failed, exiting!"; \
		exit 1; \
	fi; \

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(VERSION)"; \
	git tag -a $(VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(VERSION); \
	twine upload dist/* --verbose

# general util
# --------------------------------------------------

.PHONY: clean
clean:
	@echo "cleaning up caches and temp files"
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .coverage
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