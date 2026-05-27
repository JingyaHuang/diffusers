# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for Flux2KleinPipeline on Neuron in the vllm-omni serving calling pattern.

vllm-omni's DiffusersAdapterPipeline calls the pipeline as follows:
  1. DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, **load_kwargs)
  2. pipeline.to(device)                        # auto-synchronizes for Neuron
  3. torch.inference_mode():
       output = pipeline(prompt=..., generator=torch.Generator("cpu").manual_seed(seed), ...)

These tests verify that this exact calling sequence works correctly on Neuron hardware,
and that the fixes in diffusers (auto-synchronize in .to(), CPU generator support) are
sufficient for vllm-omni to serve diffusers models on Neuron without further patches.
"""
import gc
import os
import unittest

import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.utils.import_utils import is_torch_neuronx_available

from ...testing_utils import (
    backend_empty_cache,
    require_torch_neuron,
    torch_device,
)


def get_dummy_components(num_layers: int = 1, num_single_layers: int = 1):
    torch.manual_seed(0)
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=256,
        axes_dims_rope=[4, 4, 4, 4],
        guidance_embeds=False,
    )

    config = Qwen3Config(
        intermediate_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=151936,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    text_encoder = Qwen3ForCausalLM(config)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
    )

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=1,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )

    scheduler = FlowMatchEulerDiscreteScheduler()

    return {
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "transformer": transformer,
        "vae": vae,
    }


@require_torch_neuron
class Flux2KleinNeuronServingTests(unittest.TestCase):
    """
    Tests simulating the vllm-omni DiffusersAdapterPipeline calling pattern on Neuron.

    Uses small dummy-weight models to avoid requiring real checkpoints.
    """

    def setUp(self):
        super().setUp()
        self._saved_env = {}
        if is_torch_neuronx_available():
            neff_cache_dir = "/tmp/neff_cache"
            os.makedirs(neff_cache_dir, exist_ok=True)
            for key in ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_ENABLE_NKI_SDPA"):
                self._saved_env[key] = os.environ.get(key)
            os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = neff_cache_dir
            os.environ.setdefault("TORCH_NEURONX_ENABLE_NKI_SDPA", "0")
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        for key, original in self._saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        gc.collect()
        backend_empty_cache(torch_device)

    def _make_pipe(self):
        """Build and return a dummy pipeline moved to the Neuron device.

        The .to(device) call triggers auto-synchronize for Neuron (via the fix
        in DiffusionPipeline.to()), so no manual torch.neuron.synchronize() is
        needed here — this mirrors what vllm-omni does.
        """
        pipe = Flux2KleinPipeline(**get_dummy_components())
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _base_inputs(self):
        return {
            "prompt": "a dog is dancing",
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 64,
            "output_type": "np",
            "text_encoder_out_layers": (1,),
        }

    def test_serving_with_cpu_generator(self):
        """vllm-omni passes a CPU-based torch.Generator to the pipeline.

        Neuron doesn't support device generators, so generators must be on CPU.
        randn_tensor() already handles this, but we verify the end-to-end path works.
        """
        pipe = self._make_pipe()
        generator = torch.Generator("cpu").manual_seed(42)

        with torch.inference_mode():
            output = pipe(**self._base_inputs(), generator=generator)

        self.assertIsNotNone(output.images)
        self.assertEqual(len(output.images), 1)
        image = output.images[0]
        self.assertEqual(image.shape, (8, 8, 3))
        self.assertTrue(np.all((image >= 0.0) & (image <= 1.0)), "Pixel values must be in [0, 1]")

    def test_serving_reproducibility(self):
        """Same CPU seed must produce identical output across two calls.

        vllm-omni seeds each request deterministically; Neuron must honor that.
        """
        pipe = self._make_pipe()
        inputs = self._base_inputs()

        with torch.inference_mode():
            out1 = pipe(**inputs, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        with torch.inference_mode():
            out2 = pipe(**inputs, generator=torch.Generator("cpu").manual_seed(0)).images[0]

        np.testing.assert_array_equal(out1, out2, err_msg="Same seed must produce identical output on Neuron.")

    def test_serving_without_explicit_generator(self):
        """vllm-omni may omit the generator when no seed is specified.

        The pipeline must run correctly with generator=None.
        """
        pipe = self._make_pipe()

        with torch.inference_mode():
            output = pipe(**self._base_inputs(), generator=None)

        self.assertIsNotNone(output.images)
        self.assertEqual(output.images[0].shape, (8, 8, 3))

    def test_pipeline_to_auto_synchronizes(self):
        """DiffusionPipeline.to() must synchronize Neuron before returning.

        After this fix, callers (including vllm-omni) no longer need to call
        torch.neuron.synchronize() manually after pipeline.to(device).
        """
        # Build without calling _make_pipe so we can inspect the state before inference.
        pipe = Flux2KleinPipeline(**get_dummy_components())
        # .to() should internally synchronize; no explicit synchronize call here.
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        # If auto-synchronize is missing, this first inference call would raise NCC_IDRV017.
        with torch.inference_mode():
            output = pipe(**self._base_inputs(), generator=torch.Generator("cpu").manual_seed(1))

        self.assertIsNotNone(output.images)


@require_torch_neuron
class Flux2KleinNeuronServingIntegrationTests(unittest.TestCase):
    """Integration tests against real Flux2Klein-4B weights.

    Skipped unless `RUN_SLOW=1` is set.
    """

    ckpt_id = "black-forest-labs/FLUX.2-klein-4B"
    prompt = "A small cactus with a happy face in the Sahara desert."

    def setUp(self):
        super().setUp()
        slow = os.environ.get("RUN_SLOW", "0")
        if slow != "1":
            self.skipTest("Set RUN_SLOW=1 to run integration tests against real weights.")
        self._saved_env = {}
        if is_torch_neuronx_available():
            neff_cache_dir = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache")
            os.makedirs(neff_cache_dir, exist_ok=True)
            for key in ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_ENABLE_NKI_SDPA"):
                self._saved_env[key] = os.environ.get(key)
            os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = neff_cache_dir
            os.environ.setdefault("TORCH_NEURONX_ENABLE_NKI_SDPA", "0")
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        for key, original in self._saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        gc.collect()
        backend_empty_cache(torch_device)

    def test_serving_pattern_512(self):
        """Full vllm-omni serving pattern: from_pretrained → .to() → inference_mode → CPU generator."""
        # Step 1: Load (mirrors DiffusersAdapterPipeline.load_weights)
        pipe = Flux2KleinPipeline.from_pretrained(self.ckpt_id, torch_dtype=torch.bfloat16)
        # Step 2: Move to device — auto-synchronizes for Neuron
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Step 3: Inference under torch.inference_mode with CPU generator (mirrors forward())
        with torch.inference_mode():
            output = pipe(
                prompt=self.prompt,
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=torch.Generator("cpu").manual_seed(0),
                output_type="np",
            )

        image = output.images
        self.assertEqual(image.shape, (1, 512, 512, 3))
        self.assertTrue(np.all((image >= 0.0) & (image <= 1.0)), "Pixel values must be in [0, 1]")

    def test_serving_reproducibility_real_weights(self):
        """Seeded inference must be reproducible end-to-end on Neuron."""
        pipe = Flux2KleinPipeline.from_pretrained(self.ckpt_id, torch_dtype=torch.bfloat16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=True)

        kwargs = dict(
            prompt=self.prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=1.0,
            output_type="np",
        )

        with torch.inference_mode():
            img1 = pipe(**kwargs, generator=torch.Generator("cpu").manual_seed(42)).images[0]
        with torch.inference_mode():
            img2 = pipe(**kwargs, generator=torch.Generator("cpu").manual_seed(42)).images[0]

        np.testing.assert_array_equal(img1, img2, err_msg="Seeded inference must be reproducible on Neuron.")
