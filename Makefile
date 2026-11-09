PYTHON ?= .venv/bin/python
TECTONIC ?= tectonic
export PYTHONPATH := $(CURDIR)/src
export OMP_NUM_THREADS := 2
export OPENBLAS_NUM_THREADS := 2

.PHONY: setup setup-full reproduce-results paper test data check-frozen
setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) -r requirements-replay.lock
setup-full: setup
	uv pip install --python $(PYTHON) -r requirements.lock
reproduce-results:
	$(PYTHON) scripts/reproduce_paper.py
paper:
	cd paper && $(TECTONIC) -k main.tex
test:
	$(PYTHON) -m pytest -q
check-frozen:
	$(PYTHON) scripts/check_frozen_inputs.py
data:
	$(PYTHON) scripts/download_data.py
