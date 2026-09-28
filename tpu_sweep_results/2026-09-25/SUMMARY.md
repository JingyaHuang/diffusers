# TorchTPU sweep — 2026-09-25

- Branch `torchtpu-sweep` @ `5d6ad22495`, v6e-8 (8 chips)
- torch 2.13.0+cpu, torch_tpu 0.1.1, libtpu 0.0.48, diffusers 0.41.0.dev0, transformers 5.17.0, accelerate 1.15.0

| suite/mode | pass | fail | error | skipped |
|---|---|---|---|---|
| smoke/compile | 51 | 15 | 172 | 0 |
| smoke/eager | 185 | 14 | 39 | 0 |
| tp/compile | 2 | 0 | 1 | 0 |
| tp/eager | 3 | 0 | 0 | 0 |

## Top error clusters

| suite/mode | count | signature |
|---|---|---|
| smoke/compile | 122 | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; consider creating a new tensor using reshape() instead of taking a view` |
| smoke/eager | 38 | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; consider creating a new tensor using reshape() instead of taking a view` |
| smoke/compile | 15 | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor: shape=[…], strides=[…], numel=N, dtype=N, device=cpu, is_contiguous=N, is_cpu=N, is_cuda=N, is_meta=N, storag` |
| smoke/compile | 11 | `TPU output contains non-finite values` |
| smoke/eager | 8 | `output differs from CPU beyond atol=rtol=N` |
| smoke/compile | 7 | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; consider creating a new tensor using reshape() instead of taking a view` |
| smoke/compile | 6 | `FailOnRecompileLimitHit: Hard failure due to fullgraph=True` |
| smoke/compile | 6 | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not provided: <tensor> [type.googleapis.com/torch_tpu.root_op_name='` |
| smoke/eager | 6 | `TPU output contains non-finite values` |
| smoke/compile | 4 | `output differs from CPU beyond atol=rtol=N` |
| smoke/compile | 2 | `Unsupported: Data-dependent branching` |
| smoke/compile | 1 | `BackendCompilerFailed > RuntimeError: Expected all tensors to be on the same device, but found at least two devices, tpu:N and cpu!` |
| smoke/compile | 1 | `BackendCompilerFailed > RuntimeError: to_copy(): cannot record EventSnapshot during an FX trace` |
| smoke/compile | 1 | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor: shape=[], strides=[], numel=N, dtype=N, device=cpu, is_contiguous=N, is_cpu=N, is_cuda=N, is_meta=N, storage_` |
| smoke/compile | 1 | `SIGABRT in (unknown)` |
| smoke/compile | 1 | `SIGSEGV in torch_tpu::(anonymous namespace)::RemoveDataPtrAlias()` |
| smoke/compile | 1 | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not provided: float32 [type.googleapis.com/torch_tpu.root_op_name='t` |
| smoke/compile | 1 | `Unsupported: Tensor.tolist() with non-integer tensor` |
| smoke/compile | 1 | `Unsupported: Unsupported context manager` |
| smoke/compile | 1 | `UserError: Consider annotating your code using torch._check*(). Could not extract specialized integer from data-dependent expression u0 (unhinted: u0).  (Size-like symbols: u0)` |
| smoke/compile | 1 | `UserError: Consider annotating your code using torch._check*(). Could not extract specialized integer from data-dependent expression u0 + N (unhinted: u0 + N).  (Size-like symbols: none)` |
| smoke/compile | 1 | `UserError: Could not extract specialized integer from data-dependent expression u0 + N (unhinted: u0 + N).  (Size-like symbols: u0)` |
| smoke/compile | 1 | `UserError: Could not extract specialized integer from data-dependent expression u0 + u1 + u2 + u3 + N (unhinted: u0 + u1 + u2 + u3 + N).  (Size-like symbols: u2, u1, u0, u3)` |
| smoke/compile | 1 | `UserError: Could not extract specialized integer from data-dependent expression u1 + N (unhinted: u1 + N).  (Size-like symbols: none)` |
| smoke/compile | 1 | `UserError: Could not extract specialized integer from data-dependent expression u2*(u0 + N) (unhinted: u2*(u0 + N)).  (Size-like symbols: none)` |
| smoke/eager | 1 | `SIGSEGV in torch_tpu::(anonymous namespace)::RemoveDataPtrAlias()` |
| tp/compile | 1 | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor: shape=[…], strides=[…], numel=N, dtype=N, device=cpu, is_contiguous=N, is_cpu=N, is_cuda=N, is_meta=N, storag` |
