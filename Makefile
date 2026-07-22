# HF Kernels on Neuron — dev tasks
# Run `make help` for available targets

.PHONY: help install verify demo test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies (run on trn2)
	pip install -r requirements.txt

verify: ## Run neuron device path verification (Week 1 goal)
	python scripts/verify_neuron_path.py

demo: ## Run identity kernel swap demo
	python scripts/demo_identity_swap.py

test: ## Run all tests
	python -m pytest tests/ -v

lint: ## Check code style
	python -m py_compile kernels/neuron_rmsnorm/layers.py
	python -m py_compile kernels/neuron_identity/layers.py
	python -m py_compile scripts/verify_neuron_path.py
	python -m py_compile scripts/demo_identity_swap.py

clean: ## Remove caches and compiled artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .neuron_cache neuron_cache
	rm -rf .pytest_cache

versions: ## Print installed package versions (for PoC record)
	@echo "=== Package Versions ==="
	@python -c "import kernels; print(f'kernels: {getattr(kernels, \"__version__\", \"dev\")}')" 2>/dev/null || echo "kernels: NOT INSTALLED"
	@python -c "import transformers; print(f'transformers: {transformers.__version__}')" 2>/dev/null || echo "transformers: NOT INSTALLED"
	@python -c "import torch; print(f'torch: {torch.__version__}')" 2>/dev/null || echo "torch: NOT INSTALLED"
	@python -c "import torch_neuronx; print('torch_neuronx: available')" 2>/dev/null || echo "torch_neuronx: NOT INSTALLED"
	@python -c "import neuronxcc; print(f'neuronx-cc: {getattr(neuronxcc, \"__version__\", \"unknown\")}')" 2>/dev/null || echo "neuronx-cc: NOT INSTALLED"
