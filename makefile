PUBLISH_BRANCH := main
PYPI_TOKEN_FILE := .pypi-token
LAST_VERSION_FILE := .lastversion
PYPROJECT := pyproject.toml
PYTHON := poetry run python
# where to put docs
DOCS_DIR := docs
COVERAGE_REPORTS_DIR := $(DOCS_DIR)/coverage

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

# pandoc commands (for docs)
PANDOC ?= pandoc

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
	$(PYTHON) -m mypy --config-file $(PYPROJECT) maze_dataset/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) tests/


.PHONY: format
format: clean
	@echo "run formatting: pycln, isort, and black"
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .
	$(PYTHON) -m isort format .
	$(PYTHON) -m black .


.PHONY: check-format
check-format: clean
	@echo "check formatting"
	$(PYTHON) -m pycln --config $(PYPROJECT) --check --all .
	$(PYTHON) -m isort --check-only .
	$(PYTHON) -m black --check .

PYTEST_OPTIONS ?=
PYTEST_PARALLEL ?= 0

ifeq ($(PYTEST_PARALLEL),1)
	PYTEST_OPTIONS+=-n auto
endif

# docs, coverage reports, and benchmarks
# --------------------------------------------------
# whether to run pytest with coverage report generation
COV ?= 1

ifeq ($(COV),1)
    PYTEST_OPTIONS+=--cov=.
endif

.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) docs/make_docs.py

.PHONY: docs-md
docs-md:
	@echo "generate combined docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) docs/make_docs.py --combined


.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined docs in markdown and other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# $(PANDOC) -f markdown -t pdf $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).pdf


.PHONY: cov
cov:
	@echo "generate coverage reports (run tests manually)"
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

.PHONY: benchmark
benchmark:
	@echo "run benchmarks"
	$(PYTHON) docs/benchmarks/benchmark_generation.py

.PHONY: docs
docs: cov docs-html docs-combined
	@echo "generate all documentation"

.PHONY: clean-docs
clean-docs:
	@echo "clean up docs"
	rm -rf $(DOCS_DIR)/combined/
	rm -rf $(DOCS_DIR)/maze-dataset/
	rm -rf $(COVERAGE_REPORTS_DIR)/
	rm $(DOCS_DIR)/maze-dataset.html
	rm $(DOCS_DIR)/index.html
	rm $(DOCS_DIR)/search.js


# testing
# --------------------------------------------------

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