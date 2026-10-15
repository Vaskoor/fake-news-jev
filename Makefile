# MIT License
#
# Copyright (c) 2024 Bhaskar Kaphle <kaphlebhaskar012345@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# =============================================================================
# Makefile — fake-news-jev
#
# All targets work offline against the synthetic Jev backend and the
# rule-based meta-classifier fallback (no real datasets, no Jev API key,
# no trained LightGBM model required for `make train` / `make evaluate`).
# =============================================================================

PYTHON      ?= python
PIP         ?= $(PYTHON) -m pip
PYTEST      ?= $(PYTHON) -m pytest
RUFF        ?= $(PYTHON) -m ruff
BLACK       ?= $(PYTHON) -m black
MYPY        ?= $(PYTHON) -m mypy
DOCKER      ?= docker
DOCKER_COMPOSE ?= docker compose

SEED        ?= 42
DATASET     ?= synthetic
PORT_API    ?= 8000
PORT_UI     ?= 8501

SRC_DIR     := src
TEST_DIR    := tests
SCRIPTS_DIR := scripts

.PHONY: help install install-dev lint test train evaluate api ui \
        docker-build docker-up docker-down docker-logs clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub(/^/, "  "); printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: install ## Install runtime + dev dependencies (pytest, ruff, black, mypy)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

lint: ## Run ruff + black --check
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

format: ## Apply ruff fixes + black formatting
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	$(BLACK) $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

typecheck: ## Run mypy on the src tree
	$(MYPY) $(SRC_DIR)

test: ## Run the offline pytest suite
	PYTHONPATH=$(SRC_DIR) $(PYTEST) -q $(TEST_DIR)

test-cov: ## Run pytest with coverage report
	PYTHONPATH=$(SRC_DIR) $(PYTEST) --cov=fakenews --cov-report=term-missing $(TEST_DIR)

train: ## Train the meta-classifier on the synthetic dataset (offline, seed=42)
	PYTHONPATH=$(SRC_DIR) $(PYTHON) $(SCRIPTS_DIR)/train.py --dataset $(DATASET) --seed $(SEED)

evaluate: ## Evaluate the trained meta-classifier on the synthetic test split
	PYTHONPATH=$(SRC_DIR) $(PYTHON) $(SCRIPTS_DIR)/evaluate.py --dataset $(DATASET) --seed $(SEED)

api: ## Run the FastAPI server on :8000 (uvicorn, reload)
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m uvicorn api.main:app \
	    --host 0.0.0.0 --port $(PORT_API) --reload

ui: ## Run the Streamlit UI on :8501
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m streamlit run ui/app.py \
	    --server.port $(PORT_UI) --server.address 0.0.0.0 \
	    --server.headless true

docker-build: ## Build the multi-stage Docker image
	$(DOCKER) build -t fake-news-jev:latest .

docker-up: ## docker compose up — api + ui (depends_on healthcheck)
	$(DOCKER_COMPOSE) up -d --build

docker-down: ## docker compose down
	$(DOCKER_COMPOSE) down

docker-logs: ## Tail compose logs
	$(DOCKER_COMPOSE) logs -f

clean: ## Remove caches, build artefacts, results (keeps data + models)
	$(PYTHON) -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'build', 'dist'] + glob.glob('*.egg-info')]"
	rm -rf results/* 2>/dev/null || true
	@touch results/.gitkeep