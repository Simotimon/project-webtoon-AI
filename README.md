# project-webtoon-AI

A local AI pipeline that turns a story script into a vertical-scroll webtoon episode — panels, consistent characters, speech bubbles, and a final stitched strip — running on a single consumer GPU (RTX 5080, 16 GB VRAM).

> Status: **planning / scaffolding**. See [PATCH_NOTES.md](PATCH_NOTES.md).

## Pipeline

```
script.txt
   │
   ▼
[1] Story → Panels      LLM splits the script into panels
   │                    (shot type, characters, pose, dialogue)
   ▼
[2] Character Sheets    one reference sheet per character
   │
   ▼
[3] Panel Generation    image-edit model + character references
   │                    → consistent characters across panels
   ▼
[4] Lettering           speech bubbles + text (Pillow, code-only)
   │
   ▼
[5] Vertical Stitch     panels + gutters → one long webtoon image
   │
   ▼
outputs/episode_01.png
```

## Models

| Stage | Model | Why |
|---|---|---|
| Story → Panels | **Qwen3** (local, 8B–14B, quantized) — or an API LLM | Outputs structured JSON per panel; handles Korean scripts well |
| Character sheets | **FLUX.1-dev** (+ optional character LoRA) | Strong base illustration quality; already familiar setup |
| Panel generation (main) | **Qwen-Image-Edit** (latest multi-image version, FP8/GGUF) | Takes multiple reference images → keeps the same character across panels. Fits 16 GB with quantization |
| Panel generation (alt) | **FLUX.1 Kontext [dev]** | Single-reference editing; fallback / A/B comparison |
| Upgrade option | **FLUX.2 [dev]** | Native multi-reference, higher quality, but heavy — needs FP8 + CPU offload on 16 GB |
| Lettering | none (Pillow) | Diffusion models are unreliable at Korean text, so bubbles and text are drawn in code |

Optional: train a per-character **LoRA** (e.g. with ai-toolkit) once a main character is locked in, for the strongest consistency over many episodes.

> Licenses: FLUX.1-dev / FLUX.2-dev are non-commercial by default. Qwen-Image-Edit is Apache 2.0. Check before publishing for money.

## Planned layout

```
project-webtoon-AI/
├── src/webtoon_ai/
│   ├── story.py        # [1] script → panel JSON
│   ├── characters.py   # [2] character reference sheets
│   ├── panels.py       # [3] panel image generation
│   ├── lettering.py    # [4] speech bubbles / text
│   └── stitch.py       # [5] vertical strip export
├── characters/         # reference images per character
├── outputs/            # generated episodes
├── docs/
├── requirements.txt
└── PATCH_NOTES.md
```

## Requirements (target)

- Windows 11, NVIDIA RTX 5080 (16 GB), CUDA 12.8+
- Python 3.13
- PyTorch (CUDA build), diffusers, transformers, accelerate, Pillow
