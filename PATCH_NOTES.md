# Patch Notes

All changes are logged here, newest first.

## v0.1.1 — 2026-11-09
### Changed
- README: project is now explicitly **local only** (no cloud servers / paid APIs); removed the API LLM option
- `localhost` LLM endpoint documented as the user's own machine (Ollama / LM Studio)

## v0.1.0 — 2026-11-09
### Added
- `src/webtoon_ai/story.py` — stage [1] Novel → Panels
  - Paste novel text; no hand-written prompts needed
  - Auto character extraction → `characters/characters.json` (existing entries are kept)
  - Long text split into ~3,000-char chunks; previous panel passed along for continuity
  - Panels: shot type, characters, scene, action, emotion, narration, dialogue (max 2 lines)
  - Image prompts built in code from panel + character sheet (consistent descriptions)
  - Manual prompt mode: `prompt_override` per panel
  - `--rebuild` to regenerate prompts after editing the JSON
  - Works with any OpenAI-compatible local server (Ollama default, LM Studio, llama.cpp, vLLM)
  - Robust JSON parsing (strips `<think>` blocks and code fences, one retry)
### Changed
- README: input is now novel text; added "How it works", usage, and panel format

## v0.0.1 — 2026-11-09
- Repository created
- README with pipeline design and model choices
- Project folder scaffolding
