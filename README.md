# project-webtoon-AI

A local AI pipeline that turns **novel text** into a vertical-scroll webtoon episode — panels, consistent characters, speech bubbles, and a final stitched strip — running on a single consumer GPU (RTX 5080, 16 GB VRAM).

> Status: **early development** — stage [1] done. See [PATCH_NOTES.md](PATCH_NOTES.md).

## How it works

You paste in a novel chapter. You do **not** write image prompts.

1. An LLM reads the novel, finds the characters, and adapts the text into panels (shot type, action, emotion, narration, dialogue).
2. Image prompts are built automatically from each panel + a fixed character appearance sheet, so a character is described the same way in every panel.
3. You can review/edit the panel JSON before rendering.
4. **Manual prompt mode:** set `prompt_override` on any panel to use your own prompt for just that panel.

## Pipeline

```
novel.txt
   │
   ▼
[1] Novel → Panels      LLM adapts the text into panels       ✅ story.py
   │                    (shot, characters, action, narration, dialogue)
   │                    + characters/characters.json (appearance sheet)
   │                    → review / edit panels.json here
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

## Usage

### [1] Novel → panels

Run a local LLM with an OpenAI-compatible API (e.g. Ollama: `ollama pull qwen3:14b`), then:

```bash
set PYTHONPATH=src
python -m webtoon_ai.story novel.txt -o outputs/ep01/panels.json
```

- `characters/characters.json` is created on the first run. Fix the appearances there — they are reused for every episode.
- Edited panel fields in `panels.json`? Rebuild prompts:
  `python -m webtoon_ai.story outputs/ep01/panels.json --rebuild`
- Options: `--model`, `--base-url` (LM Studio: `http://localhost:1234/v1`), `--style`.

Panel format:

```json
{
  "id": 3,
  "shot": "close-up",
  "characters": ["minsu"],
  "scene": "rainy school rooftop at dusk",
  "action": "turns around quickly",
  "emotion": "shocked",
  "narration": "",
  "dialogue": [{"speaker": "minsu", "text": "뭐라고?"}],
  "prompt": "(auto-built)",
  "prompt_override": null
}
```

## Patch notes

Every change is logged in [PATCH_NOTES.md](PATCH_NOTES.md).

## Requirements (target)

- Windows 11, NVIDIA RTX 5080 (16 GB), CUDA 12.8+
- Python 3.13
- PyTorch (CUDA build), diffusers, transformers, accelerate, Pillow
