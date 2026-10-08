PYTHON ?= python3.12

.PHONY: run test clean-caches

run:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m geoscope

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q

clean-caches:
	find geoscope tests .venv -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
