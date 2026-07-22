# HF Kernels on Neuron — dev tasks
# Run `make help` for available targets

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install verify demo test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv ## Install dependencies into venv
	$(PIP) install -r requirements.txt

verify: ## Run neuron device path verification (Week 1 goal)
	$(PYTHON) scripts/verify_neuron_path.py

demo: ## Run identity kernel swap demo
	$(PYTHON) scripts/demo_identity_swap.py

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

lint: ## Check code style
	$(PYTHON) -m py_compile kernels/neuron_rmsnorm/layers.py
	$(PYTHON) -m py_compile kernels/neuron_identity/layers.py
	$(PYTHON) -m py_compile scripts/verify_neuron_path.py
	$(PYTHON) -m py_compile scripts/demo_identity_swap.py

clean: ## Remove caches and compiled artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .neuron_cache neuron_cache
	rm -rf .pytest_cache

versions: ## Print installed package versions (for PoC record)
	@echo "=== Package Versions ==="
	@$(PYTHON) -c "import kernels; print(f'kernels: {getattr(kernels, \"__version__\", \"dev\")}')" 2>/dev/null || echo "kernels: NOT INSTALLED"
	@$(PYTHON) -c "import transformers; print(f'transformers: {transformers.__version__}')" 2>/dev/null || echo "transformers: NOT INSTALLED"
	@$(PYTHON) -c "import torch; print(f'torch: {torch.__version__}')" 2>/dev/null || echo "torch: NOT INSTALLED"
	@$(PYTHON) -c "import torch_neuronx; print('torch_neuronx: available')" 2>/dev/null || echo "torch_neuronx: NOT INSTALLED"
	@$(PYTHON) -c "import neuronxcc; print(f'neuronx-cc: {getattr(neuronxcc, \"__version__\", \"unknown\")}')" 2>/dev/null || echo "neuronx-cc: NOT INSTALLED"
