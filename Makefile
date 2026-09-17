PY ?= .venv/bin/python
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -shared

.PHONY: all native test examples flowchart invent-flowchart figures clean

all: native

native: crazyai/_kernels.so

crazyai/_kernels.so: cpp/kernels.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: native
	$(PY) -m pytest -q

examples: native
	for f in examples/0[1-5]_*.py examples/07_*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

flowchart:
	node assets/flowchart/flowchart.js

invent-flowchart:
	node assets/flowchart/invent_flowchart.js

figures: native
	$(PY) assets/figures/make_figures.py

clean:
	rm -f crazyai/_kernels.so
	rm -rf archive/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
