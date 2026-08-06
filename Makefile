# HF Kernels on Neuron — dev tasks
# Run `make help` for available targets
#
# On Neuron DLAMI: uses the pre-installed PyTorch venv at /opt/aws_neuronx_venv_pytorch_2_9
# Override with: make install NEURON_VENV=/path/to/other/venv

NEURON_VENV := /opt/aws_neuronx_venv_pytorch_2_9
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install verify demo test test-nki test-e2e test-all probe mfu mfu-unfixed \
        mfu-amortisation rootcause fusion insitu profile experiments registration sync lint clean \
        versions results results-render check-docs check-provenance flagcheck

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

results: ## Re-run EVERY measurement, writing raw artifacts into results/raw/ (30-60 min, on trn2)
	$(PYTHON) scripts/regenerate_results.py

results-render: ## Regenerate results/README.md from results/measurements.json (runs anywhere)
	python3 scripts/render_results.py

check-docs: ## Verify doc links, number qualifiers, results sync, and artifact provenance
	python3 scripts/check_docs_consistency.py
	python3 scripts/check_measurement_provenance.py

check-provenance: ## Verify every measurement's named artifact exists AND would be committed
	python3 scripts/check_measurement_provenance.py

insitu: ## In-situ split: how much of the regression is device vs dispatch
	$(PYTHON) scripts/profile_model_device_time.py --mode baseline \
		--outdir results/raw/prof_model_baseline
	$(PYTHON) scripts/profile_model_device_time.py --mode kernelized \
		--outdir results/raw/prof_model_kernelized
	# No --wall-* : each profile dir carries its own wall_times.json, so the walls and the device
	# times come from the same run. Passing them here is how a previous version ended up pairing
	# walls from an expired host with device times from a live one.
	$(PYTHON) scripts/sum_model_device_time.py \
		results/raw/prof_model_baseline results/raw/prof_model_kernelized --nki-calls 169 \
		--json-out results/raw/insitu-summary/insitu_decomposition.json

flagcheck: ## Controls: is the NKI/torch gap a compiler-flag artifact? (wall clock, then device)
	$(PYTHON) scripts/probe_compiler_flags.py
	$(PYTHON) scripts/probe_device_time_under_flags.py --op silu \
		--outdir-base results/raw/flagcheck

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
E2E_TESTS := tests/test_qwen3_neuron_e2e.py tests/test_qwen3_moe_e2e.py

test: test-nki test-e2e ## Run all kernel + e2e tests (must be on trn2)

test-all: ## Same coverage as `test`, in one launchable process (use with run_detached.sh)
	$(PYTHON) scripts/run_all_tests.py

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

probe: ## Run all investigation probes (device path, NKI execution, API, packaging, versions)
	$(PYTHON) scripts/smoke_device.py
	$(PYTHON) scripts/probe_neuron_device_path.py
	$(PYTHON) scripts/probe_nki_execution.py
	$(PYTHON) scripts/probe_nki_api.py
	$(PYTHON) scripts/probe_nki05_api.py
	$(PYTHON) scripts/probe_nki_versions.py
	$(PYTHON) scripts/probe_hub_packaging.py
	$(PYTHON) scripts/probe_nkilib_bundled.py

mfu: ## Measure MFU with the Finding #24 fix applied (long-running)
	$(PYTHON) scripts/measure_mfu.py --preset 0.6b --seq 512 --fix-target-detection

mfu-unfixed: ## Same without the fix — reproduces the original 208x regression
	$(PYTHON) scripts/measure_mfu.py --preset 0.6b --seq 512

# --json-out under results/raw/, not /tmp: on a rented instance /tmp is the least durable path
# available, and this project already lost every raw artifact to it once (sticking point #17).
mfu-amortisation: ## Two sequence lengths + comparison, showing the residual is near-fixed per call
	$(PYTHON) scripts/measure_mfu.py --preset 0.6b --seq 512 --fix-target-detection \
		--json-out results/raw/mfu-amortisation/mfu_512.json
	$(PYTHON) scripts/measure_mfu.py --preset 0.6b --seq 2048 --fix-target-detection \
		--json-out results/raw/mfu-amortisation/mfu_2048.json
	$(PYTHON) scripts/compare_mfu_runs.py \
		results/raw/mfu-amortisation/mfu_512.json results/raw/mfu-amortisation/mfu_2048.json

fusion: ## Are the kernels faster than the ops they replace, on device?
	$(PYTHON) scripts/run_device_profile_sweep.py --calls 1 28 --outdir-base results/raw
	$(PYTHON) scripts/analyse_fusion_barrier.py --profile-base results/raw

rootcause: ## Reproduce Finding #24: graph batching -> device profile -> cProfile -> verified fix
	$(PYTHON) scripts/probe_neff_count.py
	$(PYTHON) scripts/probe_where_is_the_time.py
	$(PYTHON) scripts/probe_inside_one_call.py
	$(PYTHON) scripts/probe_inside_one_call.py --fix-target-detection
	$(PYTHON) scripts/probe_target_override_fix.py

profile: ## Generate a NEFF+NTFF for the 28-call graph (then read with neuron-explorer)
	$(PYTHON) scripts/profile_nki_call_cost.py --calls 28 --outdir results/raw/prof_n28

experiments: ## Run the perf-attribution experiments behind Findings #20, #21 and #23
	$(PYTHON) scripts/experiment_nkilib_thin_wrapper.py
	$(PYTHON) scripts/spike_nkilib_mlp.py
	$(PYTHON) scripts/experiment_nki_graph_break.py
	$(PYTHON) scripts/diagnose_torch_compile.py

registration: ## Print the neuron kernel mapping + proposed upstream diff
	$(PYTHON) scripts/neuron_kernel_registration.py

sync: ## rsync the working tree to trn2 for testing
	./scripts/sync_to_trn2.sh

# Set QUIP_THREAD to the token in the doc URL, e.g. quip-amazon.com/AbCdEfGhIjKl/... -> AbCdEfGhIjKl
# Dry run by default. QUIP_API_TOKEN must be in the environment; never commit it.
quip-status: ## Preview syncing the status doc to Quip (add APPLY=1 to actually write)
	python3 scripts/sync_quip.py --thread $(QUIP_THREAD) \
		--file deliverables/status-and-questions.md $(if $(APPLY),--apply,)

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
