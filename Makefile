# HF Kernels on Neuron — dev tasks
# Run `make help` for available targets
#
# On Neuron DLAMI: uses the pre-installed PyTorch venv at /opt/aws_neuronx_venv_pytorch_2_9
# Override with: make install NEURON_VENV=/path/to/other/venv

NEURON_VENV := /opt/aws_neuronx_venv_pytorch_2_9
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install verify demo test test-nki test-e2e probe registration sync lint clean versions

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

venv: ## Create venv linked to Neuron PyTorch environment
	@if [ -d "$(NEURON_VENV)" ]; then \
		echo "Using Neuron venv at $(NEURON_VENV)"; \
		python3 -m venv --system-site-packages $(VENV); \
		echo "$(NEURON_VENV)/lib/python3.12/site-packages" > $(VENV)/lib/python3.12/site-packages/neuron.pth; \
	else \
		echo "Neuron venv not found at $(NEURON_VENV), creating standalone venv"; \
		python3 -m venv $(VENV); \
		$(PIP) install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com; \
	fi
	$(PIP) install --upgrade pip

install: venv ## Install project dependencies
	$(PIP) install -r requirements.txt

verify: ## Run neuron device path verification (Week 1 goal)
	$(PYTHON) scripts/verify_neuron_path.py

demo: ## Run identity kernel swap demo
	$(PYTHON) scripts/demo_identity_swap.py

# Kernel accuracy suites. These are standalone scripts, not pytest modules: each
# owns its device setup and prints a result table. They MUST run on Trainium —
# require_neuron() refuses to report results otherwise, because a CPU run silently
# exercises the PyTorch fallback instead of NKI (docs/poc-findings.md Finding #8).
NKI_TESTS := tests/test_rmsnorm_nki.py tests/test_rope_nki.py tests/test_silu_nki.py
E2E_TESTS := tests/test_qwen3_neuron_e2e.py

test: test-nki test-e2e ## Run all kernel + e2e tests (must be on trn2)

test-nki: ## Run per-kernel accuracy suites (RMSNorm, RoPE, SiLU)
	@for t in $(NKI_TESTS); do \
		echo "===== $$t ====="; \
		$(PYTHON) $$t || exit 1; \
	done

test-e2e: ## Run the Qwen3 end-to-end kernel swap test
	@for t in $(E2E_TESTS); do \
		echo "===== $$t ====="; \
		$(PYTHON) $$t || exit 1; \
	done

probe: ## Run all investigation probes (device path, NKI execution, API, packaging)
	$(PYTHON) scripts/probe_neuron_device_path.py
	$(PYTHON) scripts/probe_nki_execution.py
	$(PYTHON) scripts/probe_nki_api.py
	$(PYTHON) scripts/probe_hub_packaging.py

registration: ## Print the neuron kernel mapping + proposed upstream diff
	$(PYTHON) scripts/neuron_kernel_registration.py

sync: ## rsync the working tree to trn2 for testing
	./scripts/sync_to_trn2.sh

lint: ## Byte-compile all kernels, scripts, and tests
	$(PYTHON) -m compileall -q kernels scripts tests

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
