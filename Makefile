# ─────────────────────────────────────────────────────────────────────────────
# DOC-ANNs Makefile
# Shortcuts for Docker and local development
# ─────────────────────────────────────────────────────────────────────────────

IMAGE   = doc-anns:latest
COMPOSE = docker compose

.PHONY: help build notebook cli test clean predict-demo owc-test local-install local-test

# ── Default: show help ────────────────────────────────────────────────────
help:
	@echo ""
	@echo "DOC-ANNs — available commands"
	@echo "────────────────────────────────────────────────"
	@echo ""
	@echo "  Docker (recommended):"
	@echo "    make build          Build the Docker image"
	@echo "    make notebook       Start Jupyter at http://localhost:8888"
	@echo "    make test           Run all tests inside Docker"
	@echo "    make predict-demo   Run predict_from_csv.py demo inside Docker"
	@echo "    make owc-test       Run OWC classifier self-test inside Docker"
	@echo "    make clean          Remove containers and image"
	@echo ""
	@echo "  Local (without Docker):"
	@echo "    make local-install  pip install -r requirements.txt"
	@echo "    make local-test     Run tests locally with pytest"
	@echo ""

# ── Docker ────────────────────────────────────────────────────────────────
build:
	$(COMPOSE) build

notebook:
	@echo "Starting Jupyter — open http://localhost:8888 in your browser"
	@echo "Press Ctrl+C to stop."
	$(COMPOSE) up notebook

test:
	$(COMPOSE) run --rm test

predict-demo:
	$(COMPOSE) run --rm cli examples/predict_from_csv.py --model ANNb

owc-test:
	$(COMPOSE) run --rm cli water_classification/owc_classifier.py

clean:
	$(COMPOSE) down --rmi local --volumes --remove-orphans
	docker image rm $(IMAGE) 2>/dev/null || true

# ── Local (no Docker) ─────────────────────────────────────────────────────
local-install:
	pip install -r requirements.txt

local-test:
	PYTHONPATH=. pytest tests/ -v --tb=short
