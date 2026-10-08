# Interpreter selection.
#   make test                         -> system python3
#   make test PYTHON=/path/to/python  -> that interpreter
#   make test VENV=.vidur-npm-venv -> <VENV>/bin/python
VENV ?=
PYTHON ?= python3
ifeq ($(strip $(VENV)),)
INTERP := $(PYTHON)
else
INTERP := $(VENV)/bin/python
endif

.PHONY: run test clean-caches

run:
	PYTHONDONTWRITEBYTECODE=1 $(INTERP) -m vidur

test:
	PYTHONDONTWRITEBYTECODE=1 $(INTERP) -m pytest -q

clean-caches:
	find vidur tests .venv .vidur-npm-venv -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
