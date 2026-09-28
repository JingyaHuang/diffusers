# TorchTPU sweep report — 2026-09-25

Branch `torchtpu-sweep` @ `5d6ad22495` on v6e-8 (8 chips).
torch 2.13.0+cpu, torch_tpu 0.1.1, libtpu 0.0.48, diffusers 0.41.0.dev0, transformers 5.17.0, accelerate 1.15.0

## Pass rates

238 pipeline configs in 85 families, each run in eager and compile mode on one chip; 3 tensor-parallel models on all chips. Pass rate = pass / (pass + fail + error).

| suite/mode | pass rate | pass | fail | error | skipped |
|---|---|---|---|---|---|
| smoke/compile | 21% (51/238) | 51 | 15 | 172 | 0 |
| smoke/eager | 78% (185/238) | 185 | 14 | 39 | 0 |
| tp/compile | 67% (2/3) | 2 | 0 | 1 | 0 |
| tp/eager | 100% (3/3) | 3 | 0 | 0 | 0 |

## Eager × compile

Configs per (eager, compile) outcome.

| eager \ compile | pass | fail | error | skipped |
|---|---|---|---|---|
| **pass** | 50 | 11 | 124 | 0 |
| **fail** | 1 | 4 | 9 | 0 |
| **error** | 0 | 0 | 39 | 0 |
| **skipped** | 0 | 0 | 0 | 0 |

## What breaks compile

When a compile job fails, each component is retried compiled alone. Jobs where a component fails on its own (out of 187 failing compile jobs):

| component | jobs |
|---|---|
| `vae` | 115 |
| `unet` | 38 |
| `transformer` | 31 |
| `movq` | 16 |
| `motion_adapter` | 7 |
| `controlnet` | 6 |
| `watermarker` | 6 |
| `prior_prior` | 6 |
| `condition_encoder` | 1 |
| `vqvae` | 1 |
| `model` | 1 |

## Top root causes

| suite/mode | jobs | families | root cause |
|---|---|---|---|
| smoke/compile | 122 | animatediff, audioldm2, aura_flow, chroma, cogview3, consistency_models +36 | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; consider creating a new tens` |
| smoke/eager | 38 | animatediff, consistency_models, controlnet_hunyuandit, controlnet_sd3, ddim, ddpm +9 | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; consider creating a new tensor using reshape() inste` |
| smoke/compile | 15 | consisid, helios, kandinsky5, lumina, nucleusmoe_image, qwenimage +1 | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor: shape=[…], strides=[…], numel=N, dtype=N, device=` |
| smoke/compile | 11 | dreamlite, ltx2, pag, sana | `TPU output contains non-finite values` |
| smoke/eager | 8 | diffusion_gemma, glm_image, kandinsky5, kolors, ltx2, pag | `output differs from CPU beyond atol=rtol=N` |
| smoke/compile | 7 | kandinsky, kandinsky2_2, kandinsky3 | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; consider creating a new tensor using reshape() inste` |
| smoke/compile | 6 | kandinsky, kandinsky2_2 | `FailOnRecompileLimitHit: Hard failure due to fullgraph=True` |
| smoke/compile | 6 | anyflow, cogview4, easyanimate, joyimage, longcat_audio_dit | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov` |
| smoke/eager | 6 | helios, joyimage, lumina2, sana, sana_video | `TPU output contains non-finite values` |
| smoke/compile | 4 | hunyuan_video, sana, sana_video | `output differs from CPU beyond atol=rtol=N` |
| smoke/compile | 2 | hunyuan_video1_5, omnigen | `Unsupported: Data-dependent branching` |
| smoke/compile | 1 | chroma | `BackendCompilerFailed > RuntimeError: Expected all tensors to be on the same device, but found at least two devices, tpu:N and cpu!` |
| smoke/compile | 1 | diffusion_gemma | `BackendCompilerFailed > RuntimeError: to_copy(): cannot record EventSnapshot during an FX trace` |
| smoke/compile | 1 | latent_diffusion | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor: shape=[], strides=[], numel=N, dtype=N, device=cp` |
| smoke/compile | 1 | glm_image | `SIGABRT in (unknown)` |
| smoke/compile | 1 | ltx2 | `SIGSEGV in torch_tpu::(anonymous namespace)::RemoveDataPtrAlias()` |
| smoke/compile | 1 | anyflow | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov` |
| smoke/compile | 1 | qwenimage21 | `Unsupported: Tensor.tolist() with non-integer tensor` |
| smoke/compile | 1 | z_image | `Unsupported: Unsupported context manager` |
| smoke/compile | 1 | hidream_image | `UserError: Consider annotating your code using torch._check*(). Could not extract specialized integer from data-dependent expression u0 (unhinted: u0).  (Size-l` |

## Precision-sensitive

Failed at the default reduced fp32 matmul precision, passed with `torch.set_float32_matmul_precision("highest")`; counted as passes.

- eager: `DreamLite`
- eager: `DreamLiteMobile`
- eager: `HunyuanSkyreelsImageToVideo`
- eager: `HunyuanVideoFramepack`
- eager: `HunyuanVideoImageToVideo`
- eager: `LEditsPPStableDiffusionXL`
- eager: `LongCatAudioDiT`
- eager: `MarigoldNormals`
- eager: `MotifVideo`
- eager: `MotifVideoImage2Video`
- eager: `SanaControlNet`

## Crashes

Workers that died (signal or TorchTPU fatal check) without writing a result.

- compile: `AnyFlow` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- compile: `AnyFlowFAR` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- compile: `CogView4` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- compile: `EasyAnimate` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- compile: `GlmImage` — `SIGABRT in (unknown)`
- compile: `JoyImageEdit` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- compile: `JoyImageEditPlus` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- compile: `LTX2DFRTemporalRefine` — `SIGSEGV in torch_tpu::(anonymous namespace)::RemoveDataPtrAlias()`
- compile: `LongCatAudioDiT` — `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_ARGUMENT: identified an argument that was not prov`
- eager: `LTX2DFRTemporalRefine` — `SIGSEGV in torch_tpu::(anonymous namespace)::RemoveDataPtrAlias()`

## Tensor parallelism

| model | mode | status | root cause |
|---|---|---|---|
| flux | compile | pass | — |
| flux | eager | pass | — |
| flux2 | compile | pass | — |
| flux2 | eager | pass | — |
| qwenimage | compile | error | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor: shape=[…]` |
| qwenimage | eager | pass | — |

## Per family

| family | configs | eager | compile | main root cause |
|---|---|---|---|---|
| ace_step | 1 | 100% (1/1) | 0% (0/1) | `UserError: Could not extract specialized integer from data-dependent expression u2*(u0 + N) (unhinted: u2*(u0 ` |
| allegro | 1 | 100% (1/1) | 100% (1/1) | — |
| animatediff | 6 | 0% (0/6) | 0% (0/6) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| anyflow | 2 | 100% (2/2) | 0% (0/2) | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_` |
| audioldm2 | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| aura_flow | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| bria | 1 | 100% (1/1) | 100% (1/1) | — |
| bria_fibo | 1 | 100% (1/1) | 100% (1/1) | — |
| bria_fibo_edit | 1 | 100% (1/1) | 100% (1/1) | — |
| chroma | 2 | 100% (2/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: Expected all tensors to be on the same device, but found at least two de` |
| chronoedit | 1 | 100% (1/1) | 100% (1/1) | — |
| cogvideo | 4 | 100% (4/4) | 100% (4/4) | — |
| cogview3 | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| cogview4 | 1 | 100% (1/1) | 0% (0/1) | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_` |
| consisid | 1 | 100% (1/1) | 0% (0/1) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| consistency_models | 1 | 0% (0/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| controlnet | 14 | 100% (14/14) | 0% (0/14) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| controlnet_flux | 3 | 100% (3/3) | 0% (0/3) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| controlnet_hunyuandit | 1 | 0% (0/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| controlnet_sd3 | 2 | 50% (1/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| cosmos | 7 | 100% (7/7) | 100% (7/7) | — |
| ddim | 1 | 0% (0/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| ddpm | 1 | 0% (0/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| deepfloyd_if | 6 | 0% (0/6) | 0% (0/6) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| diffusion_gemma | 1 | 0% (0/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: to_copy(): cannot record EventSnapshot during an FX trace` |
| dit | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| dreamlite | 2 | 100% (2/2) | 0% (0/2) | `TPU output contains non-finite values` |
| easyanimate | 1 | 100% (1/1) | 0% (0/1) | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_` |
| flux | 9 | 100% (9/9) | 0% (0/9) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| flux2 | 4 | 100% (4/4) | 0% (0/4) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| glm_image | 1 | 0% (0/1) | 0% (0/1) | `SIGABRT in (unknown)` |
| helios | 1 | 0% (0/1) | 0% (0/1) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| hidream_image | 1 | 100% (1/1) | 0% (0/1) | `UserError: Consider annotating your code using torch._check*(). Could not extract specialized integer from dat` |
| hunyuan_image_21 | 1 | 100% (1/1) | 0% (0/1) | `UserError: Could not extract specialized integer from data-dependent expression u0 + u1 + u2 + u3 + N (unhinte` |
| hunyuan_video | 4 | 100% (4/4) | 25% (1/4) | `output differs from CPU beyond atol=rtol=N` |
| hunyuan_video1_5 | 1 | 100% (1/1) | 0% (0/1) | `Unsupported: Data-dependent branching` |
| hunyuandit | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| ideogram4 | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| joyimage | 2 | 50% (1/2) | 0% (0/2) | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_` |
| kandinsky | 7 | 14% (1/7) | 14% (1/7) | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; co` |
| kandinsky2_2 | 10 | 20% (2/10) | 20% (2/10) | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; co` |
| kandinsky3 | 2 | 0% (0/2) | 0% (0/2) | `RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape […] and strides […]; co` |
| kandinsky5 | 4 | 75% (3/4) | 0% (0/4) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| kolors | 2 | 0% (0/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| krea2 | 1 | 100% (1/1) | 100% (1/1) | — |
| latent_consistency_models | 2 | 100% (2/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| latent_diffusion | 2 | 50% (1/2) | 0% (0/2) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| latte | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| ledits_pp | 2 | 50% (1/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| llada2 | 1 | 100% (1/1) | 100% (1/1) | — |
| longcat_audio_dit | 1 | 100% (1/1) | 0% (0/1) | `TorchTPU fatal: Check failed: traversal->ValidateAndReorderArguments(std::move(argument_refs)) is OK (INVALID_` |
| ltx | 4 | 100% (4/4) | 100% (4/4) | — |
| ltx2 | 8 | 62% (5/8) | 0% (0/8) | `TPU output contains non-finite values` |
| lumina | 1 | 100% (1/1) | 0% (0/1) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| lumina2 | 1 | 0% (0/1) | 0% (0/1) | `UserError: Consider annotating your code using torch._check*(). Could not extract specialized integer from dat` |
| marigold | 3 | 100% (3/3) | 0% (0/3) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| mochi | 1 | 100% (1/1) | 0% (0/1) | `UserError: Could not extract specialized integer from data-dependent expression u0 + N (unhinted: u0 + N).  (S` |
| motif_video | 2 | 100% (2/2) | 100% (2/2) | — |
| nucleusmoe_image | 1 | 100% (1/1) | 0% (0/1) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| omnigen | 1 | 100% (1/1) | 0% (0/1) | `Unsupported: Data-dependent branching` |
| ovis_image | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| pag | 17 | 88% (15/17) | 0% (0/17) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| pixart_alpha | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| pixart_sigma | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| pndm | 1 | 0% (0/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| prx | 2 | 100% (2/2) | 50% (1/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| qwenimage | 6 | 100% (6/6) | 0% (0/6) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| qwenimage21 | 1 | 100% (1/1) | 0% (0/1) | `Unsupported: Tensor.tolist() with non-integer tensor` |
| sana | 4 | 75% (3/4) | 50% (2/4) | `TPU output contains non-finite values` |
| sana_video | 2 | 0% (0/2) | 50% (1/2) | `TPU output contains non-finite values` |
| shap_e | 2 | 100% (2/2) | 100% (2/2) | — |
| skyreels_v2 | 6 | 100% (6/6) | 100% (6/6) | — |
| stable_audio | 1 | 100% (1/1) | 0% (0/1) | `RuntimeError: execute(): failed to prepare compiled mode arguments: failed to get buffer from argument tensor:` |
| stable_audio_3 | 1 | 100% (1/1) | 100% (1/1) | — |
| stable_diffusion | 5 | 100% (5/5) | 0% (0/5) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_diffusion_2 | 6 | 83% (5/6) | 0% (0/6) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_diffusion_3 | 3 | 100% (3/3) | 0% (0/3) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_diffusion_adapter | 3 | 100% (3/3) | 0% (0/3) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_diffusion_image_variation | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_diffusion_xl | 7 | 100% (7/7) | 0% (0/7) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_unclip | 2 | 100% (2/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| stable_video_diffusion | 1 | 100% (1/1) | 0% (0/1) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| visualcloze | 2 | 100% (2/2) | 0% (0/2) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
| wan | 10 | 100% (10/10) | 100% (10/10) | — |
| z_image | 3 | 100% (3/3) | 0% (0/3) | `BackendCompilerFailed > RuntimeError: view(): cannot create a view of shape […] from the input tensor of shape` |
