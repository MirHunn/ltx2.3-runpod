"""
RunPod Serverless worker for LTX-2.3 (distilled, two-stage) image-to-video.

Design (same as the Qwen worker):
  - Models load at MODULE IMPORT (worker startup) -> counts as delay time, not execution time.
  - Reads all weights from the network volume; fully offline.

Request input:
  { "prompt": "...",                        # required
    "image_url": "..." | "image_base64": "...",   # optional (omit = text-to-video)
    "image_strength"?: 1.0,                 # 0..1, how strongly the image conditions frame 0
    "duration_s"?: 5.0, "fps"?: 25,         # OR pass "num_frames" directly
    "num_frames"?: int,                     # takes precedence over duration_s
    "width"?: 1280, "height"?: 704,         # snapped to /64 (two-stage requirement)
    "audio"?: true,                         # mux the generated audio track
    "enhance_prompt"?: false,               # Gemma rewrites the prompt (uses the image if given)
    "seed"?: 0 }

Response: { "video_base64": "<mp4>", "meta": {...} }   # meta.duration_s = ACTUAL clip length
"""
import os
import io
import time
import base64
import tempfile

import requests
import torch
import runpod

from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

MODEL_DIR   = "/runpod-volume/models/ltx-2.3"
CKPT        = os.path.join(MODEL_DIR, "ltx-2.3-22b-distilled-1.1.safetensors")
UPSAMPLER   = os.path.join(MODEL_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_ROOT  = "/runpod-volume/models/gemma-3-12b"
OFFLOAD     = getattr(OffloadMode, os.environ.get("LTX_OFFLOAD_MODE", "NONE"), OffloadMode.NONE)


def build_pipe():
    print(f"[init] loading LTX-2.3 distilled | offload={OFFLOAD}", flush=True)
    print(f"[init] ckpt={CKPT}", flush=True)
    p = DistilledPipeline(
        distilled_checkpoint_path=CKPT,
        spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA_ROOT,
        loras=[],
        offload_mode=OFFLOAD,
    )
    print("[init] READY", flush=True)
    return p


PIPE = build_pipe()


def _snap(v: int, div: int) -> int:
    return max(div, round(int(v) / div) * div)


def _snap_frames(n: int) -> int:
    # frame count must be 8k + 1
    return max(9, round((int(n) - 1) / 8) * 8 + 1)


def _fetch_image_to_tmp(inp) -> str | None:
    url, b64 = inp.get("image_url"), inp.get("image_base64")
    if isinstance(url, list):
        url = url[0] if url else None
    if isinstance(b64, list):
        b64 = b64[0] if b64 else None
    if not url and not b64:
        return None
    data = requests.get(url, timeout=60).content if url else base64.b64decode(b64)
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(data)
    f.close()
    return f.name


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "missing 'prompt'"}

    fps = int(inp.get("fps", 25))
    if inp.get("num_frames"):
        num_frames = _snap_frames(inp["num_frames"])
    else:
        num_frames = _snap_frames(float(inp.get("duration_s", 5.0)) * fps + 1)

    width  = _snap(inp.get("width", 1280), 64)
    height = _snap(inp.get("height", 704), 64)
    seed   = int(inp.get("seed", 0))
    want_audio     = bool(inp.get("audio", True))
    enhance_prompt = bool(inp.get("enhance_prompt", False))

    img_path = _fetch_image_to_tmp(inp)
    images = []
    if img_path:
        images.append(ImageConditioningInput(
            path=img_path,
            frame_idx=0,
            strength=float(inp.get("image_strength", 1.0)),
        ))

    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(num_frames, tiling)

    t0 = time.time()
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with torch.inference_mode():
        video_iter, audio = PIPE(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=images,
            tiling_config=tiling,
            enhance_prompt=enhance_prompt,
        )
        encode_video(
            video=video_iter,
            fps=fps,
            audio=audio if want_audio else None,
            output_path=out_path,
            video_chunks_number=chunks,
        )
    gen_s = round(time.time() - t0, 1)

    with open(out_path, "rb") as f:
        mp4 = f.read()
    for pth in (img_path, out_path):
        if pth:
            try:
                os.remove(pth)
            except OSError:
                pass

    return {
        "video_base64": base64.b64encode(mp4).decode(),
        "meta": {
            "duration_s": round((num_frames - 1) / fps, 3),
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "seed": seed,
            "audio": want_audio,
            "enhance_prompt": enhance_prompt,
            "image_conditioned": bool(images),
            "gen_time_s": gen_s,
            "file_size_mb": round(len(mp4) / 1e6, 2),
        },
    }


runpod.serverless.start({"handler": handler})
