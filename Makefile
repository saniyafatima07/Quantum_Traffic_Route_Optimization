SHELL := /bin/bash
NET := data/networks/net/bengaluru_central.net.xml
PORT ?= 8000

.PHONY: help sync net net-extract preset methods solve demo demo-build demo-ui check validate test clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

sync:  ## create .venv and install dependencies
	uv sync

$(NET):  ## compile the real Bengaluru network from OpenStreetMap
	uv run qitransit bengaluru build central --out $(NET)

net: $(NET)  ## alias for the above

net-extract:  ## clip the Geofabrik extract to the city (needs the south-india PBF)
	uv run qitransit bengaluru extract city

preset:  ## build a small synthetic city instead, for testing
	uv run qitransit net build manhattan

methods:  ## list the registered optimisers
	uv run qitransit methods

solve: $(NET)  ## solve the demo instance with one method
	uv run qitransit solve qisep-ls --budget 3000 --routes-out data/demo/best.rou.xml

demo: $(NET)  ## serve the SUMO comparison page (the thing to actually run)
	uv run qitransit demo --port $(PORT)

demo-build: $(NET)  ## build the comparison without serving it
	uv run qitransit demo --build-only

demo-ui:  ## check the page's JavaScript against a saved comparison
	uv run python scripts/check_demo_ui.py

check:  ## objectives must describe their own routes, for every method
	uv run python scripts/check_consistency.py

validate: $(NET)  ## optimise then simulate, on the command line
	uv run python scripts/validate_loop.py

test: $(NET)  ## run every check
	uv run python scripts/check_consistency.py
	uv run python scripts/check_demo_ui.py
	uv run pytest

clean:  ## remove generated artefacts (keeps the compiled network)
	rm -rf data/demo data/simulation .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
