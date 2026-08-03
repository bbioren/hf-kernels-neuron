"""The two NKI dispatch fixes this project found, in one place so they cannot drift.

Both are runtime patches, not shipped changes, and both are verified accuracy-neutral by their own
probe. They exist here as a single implementation because two scripts apply them —
`measure_mfu.py` to measure the model-level effect and the probes to verify each fix in isolation —
and a copied patch is a patch that will eventually differ between the number and the verification of
the number.

  fix_target_detection()    Finding #24. `nki/framework/compiled.py::_compile_opts()` resolves the
                            compile target on every invocation, which forks `neuron-ls` and costs
                            ~52 ms. It sits OUTSIDE `_nki_compile_cache` because its result is part
                            of that cache's key, so a cache hit still pays the subprocess in full.
                            One `lru_cache` removes it: ~52 ms -> ~0.5 ms per call.
                            Verified by scripts/probe_target_override_fix.py.

  fix_op_registry_cache()   Open item B12. `torch_xla`'s `xla_op_registry.Op` already memoises the
                            built XLA computation in `self._computations`, and its docstring asks
                            callers to register ops globally "in order to amortize the lowering
                            cost". NKI applies `@xla_hlo_call` INSIDE
                            `TorchXlaKernel.__call__`, so a fresh `Op` with a fresh empty cache is
                            constructed per call and the memo is never used. Registering once per
                            compile-cache key: ~0.53 ms -> ~0.18 ms per call.
                            Verified by scripts/probe_op_registry_cache.py.

The two are the same bug twice: a cache exists and the surrounding code path defeats it. That is
worth stating plainly, because it is the difference between "per-layer kernel dispatch on Neuron is
expensive" and "two specific caches are being bypassed" — and only the second one is true.

WHY fix_op_registry_cache's KEY IS SOUND
The lowering closure captures `config = nir.build_config()`. `nir` comes from
`self._cached_compile_to_bir(frontend, converted_inputs, compile_opts)`, which is already memoised on
`self._generate_cache_key(converted_inputs, compile_opts)`. Same key => same `nir` => same `config`
=> same closure. The key is not a judgement about what is safe to share; it is the key NKI already
uses for the object the closure is built from. Two guards make that concrete: the Op is cached only
when NKI's compile cache is enabled (with it disabled, `nir` is rebuilt per call), and a null key
falls through to the original uncached path.
"""

import functools
import hashlib
import inspect

import torch

# Structural landmarks in TorchXlaKernel.__call__ that the Op-cache patch reimplements. If NKI
# restructures its dispatch path, the patch is stale and could be silently wrong, so it refuses
# rather than guesses.
LANDMARKS = (
    "@xla_hlo_call",
    "def nki_custom_call(*tensors):",
    "self._cached_compile_to_bir(",
    "nir.build_config()",
    "xla_result = nki_custom_call(*input_tensors)",
)


def fix_target_detection(verbose=True):
    """Finding #24. Returns the detected target string."""
    import nki.compiler.target as nki_target

    if not hasattr(nki_target._detect_target, "cache_info"):
        nki_target._detect_target = functools.lru_cache(maxsize=1)(nki_target._detect_target)
    target = nki_target._detect_target()
    if verbose:
        print(f"  FIX APPLIED (#24): nki.compiler.target._detect_target is lru_cached "
              f"(detected target: {target!r})")
    return target


def _build_lowering(tx, config):
    """The lowering closure, extracted from TorchXlaKernel.__call__ so it can be registered once.

    A faithful copy of the body of `nki_custom_call`. Symbols are reached through the module object
    rather than re-imported, so this uses exactly what the original uses.
    """

    def nki_custom_call(*tensors):
        scribe = tensors[0].scribe
        output_xla_types = [
            tx._scribe_type(scribe, spec.dtype, spec.shape) for spec in config.output_specs
        ]
        ret_type = output_xla_types[0] if len(output_xla_types) == 1 \
            else scribe.tuple(*output_xla_types)
        ret_inst = ret_type.CustomCall(
            *tensors,
            backend_config=config.backend_config_b64,
            custom_call_target=tx.AwsNeuronNkiKernel,
        )
        num_outputs = len(config.output_specs)
        for input_idx, output_idx in config.operand_output_aliases.items():
            alias_proto = tx.xla_data_pb2.OutputOperandAliasing()
            alias_proto.output_shape_index.extend([output_idx] if num_outputs > 1 else [])
            alias_proto.operand_index = input_idx
            alias_proto.operand_shape_index.extend([])
            ret_inst.instruction.output_operand_aliasing.append(alias_proto)
        if config.has_collectives:
            ret_inst.instruction.frontend_attributes.map.update(dict(has_collectives=str(1)))
        return ret_inst

    # torch_neuronx derives the registered op name from __qualname__. Match the original exactly, so
    # the resulting user computation is named identically and the only difference between patched and
    # unpatched is how often it is built.
    nki_custom_call.__qualname__ = "TorchXlaKernel.__call__.<locals>.nki_custom_call"
    return nki_custom_call


def check_patch_applies(verbose=True):
    """Whether the installed NKI dispatch path matches what the Op-cache patch was written against.

    Returns (ok, source_hash, missing_landmarks).
    """
    import nki.framework._torch_xla as tx

    src = inspect.getsource(tx.TorchXlaKernel.__call__)
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    missing = [m for m in LANDMARKS if m not in src]
    if verbose:
        print(f"  TorchXlaKernel.__call__ source sha256[:16] = {h}")
        if missing:
            print("  PATCH DOES NOT APPLY — missing landmark(s):")
            for m in missing:
                print(f"    {m!r}")
        else:
            print(f"  all {len(LANDMARKS)} structural landmarks present")
    return (not missing), h, missing


def fix_op_registry_cache(verbose=True):
    """Open item B12. Returns (stats_dict, restore_callable), or (None, None) if it cannot apply.

    `stats` accumulates hit / miss / uncacheable counts, so a caller can prove the cache was
    actually used rather than inferring it from a timing change.
    """
    import nki.framework._torch_xla as tx

    ok, _, _ = check_patch_applies(verbose=verbose)
    if not ok:
        if verbose:
            print("  REFUSING to apply the Op-registry cache on an unrecognised NKI version.")
        return None, None

    original = tx.TorchXlaKernel.__call__
    op_cache = {}
    stats = {"hit": 0, "miss": 0, "uncacheable": 0}

    def patched(self, *args, **kwargs):
        frontend = self._frontend_cls(enable_backend_opt=self._enable_backend_opt)
        compile_opts = self._compile_opts()
        inputs = self._bind_args(args, kwargs)
        converted_inputs = {n: tx._convert_input(v, n) for n, v in inputs.items()}
        nir = self._cached_compile_to_bir(
            frontend=frontend, inputs=converted_inputs, compile_opts=compile_opts)
        config = nir.build_config()
        input_tensors = [v for v in inputs.values() if isinstance(v, torch.Tensor)]

        key = None
        if self._get_compile_cache() is not None:
            key = self._generate_cache_key(converted_inputs, compile_opts)

        op = op_cache.get(key) if key is not None else None
        if op is None:
            op = tx.xla_hlo_call(_build_lowering(tx, config))
            if key is not None:
                op_cache[key] = op
                stats["miss"] += 1
            else:
                stats["uncacheable"] += 1
        else:
            stats["hit"] += 1

        return op(*input_tensors)

    tx.TorchXlaKernel.__call__ = patched

    def restore():
        tx.TorchXlaKernel.__call__ = original

    stats["distinct_keys"] = 0  # replaced on read below; kept so the dict shape is stable

    def _snapshot():
        stats["distinct_keys"] = len(op_cache)
        return stats

    stats_view = _StatsView(_snapshot)
    if verbose:
        print("  FIX APPLIED (B12): the XLA computation is registered once per compile-cache key "
              "instead of once per call")
    return stats_view, restore


class _StatsView:
    """A dict-like view that refreshes distinct_keys when read.

    The caller wants to print cache statistics after the run, and `distinct_keys` is a property of
    the cache rather than a counter, so it cannot be incremented as it goes.
    """

    def __init__(self, snapshot):
        self._snapshot = snapshot

    def __getitem__(self, k):
        return self._snapshot()[k]

    def get(self, k, default=None):
        return self._snapshot().get(k, default)

    def as_dict(self):
        return dict(self._snapshot())

    def __repr__(self):
        d = self.as_dict()
        return (f"{d.get('hit', 0)} hit(s), {d.get('miss', 0)} miss(es), "
                f"{d.get('uncacheable', 0)} uncacheable, {d.get('distinct_keys', 0)} key(s)")
