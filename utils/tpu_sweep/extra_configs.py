"""Sweep configs for the pipelines whose fast tests do not follow the `BasePipelineTesterConfig` contract.

`discover.py` finds `BasePipelineTesterConfig` subclasses and adds these, which it would otherwise miss:

- LEdits++ (SD, SDXL): a call only works after `invert()` has populated the pipeline's state.
- T2I-Adapter (full, light, multi): legacy `unittest` tests whose `get_dummy_inputs` takes a device.
- DiffusionGemma, LLaDA2: discrete text diffusion pipelines that return token sequences and whose model is a plain
  `nn.Module` / transformers model rather than a diffusers `ModelMixin`.

Each class here exposes what `smoke_worker.py` expects: `pipeline_class`, `get_dummy_components()` and a no-argument
`get_dummy_inputs()`. `extra_compile_components` names non-`ModelMixin` components to compile in `compile` mode.
"""

import torch

from diffusers import (
    BlockRefinementScheduler,
    DiffusionGemmaPipeline,
    LEditsPPPipelineStableDiffusion,
    LEditsPPPipelineStableDiffusionXL,
    LLaDA2Pipeline,
    StableDiffusionAdapterPipeline,
)
from tests.pipelines.ledits_pp.test_ledits_pp_stable_diffusion import TestLEditsPPPipelineStableDiffusion
from tests.pipelines.ledits_pp.test_ledits_pp_stable_diffusion_xl import TestLEditsPPPipelineStableDiffusionXL
from tests.pipelines.llada2.test_llada2 import _DummyCausalLM
from tests.pipelines.stable_diffusion_adapter.test_stable_diffusion_adapter import (
    StableDiffusionFullAdapterPipelineFastTests,
    StableDiffusionLightAdapterPipelineFastTests,
    StableDiffusionMultiAdapterPipelineFastTests,
)


# --- LEdits++: invert, then edit, in one call ---


def _invert_then_edit(base, test_cls, single_image=False):
    class Pipeline(base):
        def __call__(self, *args, **kwargs):
            inputs = test_cls().get_dummy_inversion_inputs()
            if single_image:
                inputs["image"] = inputs["image"][0]
            self.invert(**inputs)
            return super().__call__(*args, **kwargs)

    Pipeline.__name__ = Pipeline.__qualname__ = base.__name__
    return Pipeline


class LEditsPPStableDiffusionConfig(TestLEditsPPPipelineStableDiffusion):
    pipeline_class = _invert_then_edit(LEditsPPPipelineStableDiffusion, TestLEditsPPPipelineStableDiffusion)


class LEditsPPStableDiffusionXLConfig(TestLEditsPPPipelineStableDiffusionXL):
    # Like its tests: the SDXL edit step only runs after a single-image inversion (batched inversion is checked alone).
    pipeline_class = _invert_then_edit(
        LEditsPPPipelineStableDiffusionXL, TestLEditsPPPipelineStableDiffusionXL, single_image=True
    )


# --- T2I-Adapter: bind the legacy tests' `get_dummy_inputs(device)` to CPU ---


class _AdapterConfig:
    pipeline_class = StableDiffusionAdapterPipeline
    test_cls = None

    def __init__(self):
        self._test = self.test_cls()

    def get_dummy_components(self):
        return self._test.get_dummy_components()

    def get_dummy_inputs(self):
        return self._test.get_dummy_inputs("cpu")


class StableDiffusionFullAdapterConfig(_AdapterConfig):
    test_cls = StableDiffusionFullAdapterPipelineFastTests


class StableDiffusionLightAdapterConfig(_AdapterConfig):
    test_cls = StableDiffusionLightAdapterPipelineFastTests


class StableDiffusionMultiAdapterConfig(_AdapterConfig):
    test_cls = StableDiffusionMultiAdapterPipelineFastTests


# --- Text diffusion: token sequences out, compared exactly (greedy decoding) ---


class DiffusionGemmaConfig:
    """The tiny hub checkpoint `tests/pipelines/diffusion_gemma` uses for its end-to-end tests."""

    pipeline_class = DiffusionGemmaPipeline
    extra_compile_components = ("model",)
    model_id = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"

    def get_dummy_components(self):
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        model = DiffusionGemmaForBlockDiffusion.from_pretrained(self.model_id, dtype=torch.float32)
        self.canvas_length = model.config.canvas_length
        return {
            "model": model,
            "scheduler": BlockRefinementScheduler(),
            "processor": AutoProcessor.from_pretrained(self.model_id),
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "Name a color.",
            "gen_length": self.canvas_length * 2,
            "num_inference_steps": 4,
            "temperature": 0.0,
            "eos_early_stop": False,
            "output_type": "seq",
        }


class LLaDA2Config:
    """The deterministic stand-in causal LM from `tests/pipelines/llada2`."""

    pipeline_class = LLaDA2Pipeline
    extra_compile_components = ("model",)

    def get_dummy_components(self):
        return {"model": _DummyCausalLM(vocab_size=32), "scheduler": BlockRefinementScheduler(), "tokenizer": None}

    def get_dummy_inputs(self):
        return {
            "input_ids": torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long),
            "use_chat_template": False,
            "gen_length": 24,
            "block_length": 8,
            "num_inference_steps": 8,
            "temperature": 0.0,
            "threshold": 2.0,
            "minimal_topk": 1,
            "eos_early_stop": False,
            "mask_token_id": 31,
            "eos_token_id": None,
            "output_type": "seq",
        }


EXTRA_TARGETS = {
    "LEditsPPStableDiffusionConfig": "ledits_pp",
    "LEditsPPStableDiffusionXLConfig": "ledits_pp",
    "StableDiffusionFullAdapterConfig": "stable_diffusion_adapter",
    "StableDiffusionLightAdapterConfig": "stable_diffusion_adapter",
    "StableDiffusionMultiAdapterConfig": "stable_diffusion_adapter",
    "DiffusionGemmaConfig": "diffusion_gemma",
    "LLaDA2Config": "llada2",
}
