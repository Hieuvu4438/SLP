PY ?= python

.PHONY: install test lint smoke list new aggregate

install:            ## editable install of the shared library + dev tools
	$(PY) -m pip install -e ".[dev]"

test:               ## run all tests (core + every method conformance check)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src methods scripts tests

smoke:              ## end-to-end sanity run on toy data
	$(PY) scripts/run.py --method baseline_majority --data toy --name smoke --seeds 0,1

list:               ## list all methods and their status
	$(PY) scripts/run.py --list

new:                ## make new NAME=my_method
	$(PY) scripts/new_method.py $(NAME)

aggregate:          ## make aggregate NAME=<experiment name>
	$(PY) scripts/aggregate.py --name $(NAME)
