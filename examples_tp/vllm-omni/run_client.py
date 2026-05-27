"""
vllm-omni Neuron client — send text-to-image requests to a running server.

Start the server first:
  bash run_server.sh

Then run this client:
  ~/.venv/bin/python run_client.py [--url http://localhost:8091] [--prompt "..."]

The client uses the OpenAI-compatible /v1/images/generations endpoint.
Output images are saved to /tmp/vllm_omni_output_<n>.png.
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path


def generate_image(
    url: str,
    prompt: str,
    model: str | None = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 4,
    seed: int | None = 42,
) -> bytes:
    payload: dict = {
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
        "num_inference_steps": num_inference_steps,
    }
    if model is not None:
        payload["model"] = model
    if seed is not None:
        payload["seed"] = seed

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/v1/images/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())

    b64 = result["data"][0]["b64_json"]
    return base64.b64decode(b64)


def main():
    parser = argparse.ArgumentParser(description="vllm-omni Neuron image generation client")
    parser.add_argument("--url", default="http://localhost:8091", help="Server base URL")
    parser.add_argument("--model", default=None, help="Model ID (omit to use server default)")
    parser.add_argument("--prompt", default="A serene mountain lake at sunset, photorealistic")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-requests", type=int, default=3, help="Number of requests to send")
    parser.add_argument("--output-dir", default="/tmp", help="Directory to save output images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Server : {args.url}")
    print(f"Model  : {args.model or '(server default)'}")
    print(f"Prompt : {args.prompt}")
    print(f"Size   : {args.width}x{args.height}")
    print(f"Steps  : {args.steps}")
    print(f"Seed   : {args.seed}")
    print()

    latencies = []
    for i in range(args.num_requests):
        t0 = time.perf_counter()
        png_bytes = generate_image(
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
        latency = time.perf_counter() - t0
        latencies.append(latency)

        out_path = output_dir / f"vllm_omni_output_{i}.png"
        out_path.write_bytes(png_bytes)
        print(f"  request {i+1}/{args.num_requests}: {latency:.2f}s → {out_path} ({len(png_bytes)//1024} KB)")

    print()
    print(f"Avg latency : {sum(latencies)/len(latencies):.2f}s")
    print(f"Min latency : {min(latencies):.2f}s  (warm)")


if __name__ == "__main__":
    main()
