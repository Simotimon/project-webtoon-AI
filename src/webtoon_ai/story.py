"""[1] Novel text -> panel list (JSON).

Feed in raw novel / story text. A local LLM (any OpenAI-compatible server:
Ollama, LM Studio, llama.cpp, vLLM ...) adapts it into webtoon panels:
shot type, characters, action, emotion, narration, dialogue.

The image prompt for each panel is built in code from those fields plus the
fixed character appearance sheet, so every panel describes a character the
same way. Edit the JSON before rendering, or set "prompt_override" on any
panel to use your own prompt instead (manual prompt mode).

Usage:
    python -m webtoon_ai.story novel.txt -o outputs/ep01/panels.json
    python -m webtoon_ai.story novel.txt --model qwen3:14b --base-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Callable

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_STYLE = "korean webtoon style, clean line art, cel shading, vivid colors"
CHUNK_CHARS = 3000  # novel text per LLM call

SHOT_TYPES = ["extreme close-up", "close-up", "medium", "full body", "wide", "establishing"]

LLMFn = Callable[[str, str], str]  # (system, user) -> reply text


# --------------------------------------------------------------------------- LLM

def make_openai_llm(base_url: str, model: str, temperature: float = 0.4) -> LLMFn:
    """Return a function that calls an OpenAI-compatible /chat/completions endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"

    def call(system: str, user: str) -> str:
        body = json.dumps({
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    return call


def parse_json(text: str):
    """Pull the JSON object/array out of an LLM reply (handles <think> and ``` fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in LLM reply")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1])


def ask_json(llm: LLMFn, system: str, user: str, retries: int = 1):
    last_err = None
    for _ in range(retries + 1):
        try:
            return parse_json(llm(system, user))
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            user += "\n\nYour previous reply was not valid JSON. Reply with JSON only."
    raise RuntimeError(f"LLM did not return valid JSON: {last_err}")


# ------------------------------------------------------------------------- text

def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries into chunks of about `size` characters."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------- characters

CHAR_SYSTEM = (
    "You extract characters from a novel for a webtoon adaptation. "
    "Reply with JSON only: a list of objects "
    '{"id": short lowercase ascii id, "name": name as written in the text, '
    '"appearance": concise English visual description (gender, age, hair, eyes, build, outfit)}. '
    "Only include characters who appear on screen. If the text gives no appearance, "
    "invent a reasonable, specific one."
)


def extract_characters(llm: LLMFn, text: str, known: list[dict]) -> list[dict]:
    """Find characters in the text; keep existing sheet entries untouched."""
    known_names = {c["name"] for c in known}
    found = ask_json(llm, CHAR_SYSTEM, text[:CHUNK_CHARS * 2])
    merged = list(known)
    for c in found if isinstance(found, list) else []:
        if c.get("name") and c["name"] not in known_names and c.get("id"):
            merged.append({"id": c["id"], "name": c["name"], "appearance": c.get("appearance", "")})
            known_names.add(c["name"])
    return merged


# ----------------------------------------------------------------------- panels

PANEL_SYSTEM = (
    "You are a webtoon storyboard artist adapting a novel into vertical-scroll panels. "
    "Reply with JSON only: a list of panels. Each panel: "
    '{"shot": one of ' + json.dumps(SHOT_TYPES) + ', '
    '"characters": list of character ids in the panel, '
    '"scene": English description of location and lighting, '
    '"action": English description of what happens / poses, '
    '"emotion": English facial expression or mood, '
    '"narration": narration box text in the ORIGINAL language or "", '
    '"dialogue": list of {"speaker": character id, "text": line in the ORIGINAL language}}. '
    "Rules: one moment per panel; max 2 dialogue lines per panel; keep dialogue short; "
    "vary shot types for rhythm; do not skip story beats."
)


def split_to_panels(llm: LLMFn, chunk: str, characters: list[dict], prev_tail: str = "") -> list[dict]:
    cast = "\n".join(f"- {c['id']}: {c['name']}" for c in characters)
    user = f"Characters (use these ids):\n{cast}\n\n"
    if prev_tail:
        user += f"Previous panel for continuity:\n{prev_tail}\n\n"
    user += f"Novel text:\n{chunk}"
    panels = ask_json(llm, PANEL_SYSTEM, user)
    return panels if isinstance(panels, list) else panels.get("panels", [])


def build_prompt(panel: dict, characters: dict[str, dict], style: str = DEFAULT_STYLE) -> str:
    """Compose the image prompt in code so character descriptions never drift."""
    parts = [style, f"{panel.get('shot', 'medium')} shot"]
    for cid in panel.get("characters", []):
        c = characters.get(cid)
        if c:
            parts.append(f"{c['name']} ({c['appearance']})")
    for key in ("action", "emotion", "scene"):
        if panel.get(key):
            parts.append(panel[key])
    parts.append("no text, no speech bubbles")
    return ", ".join(parts)


def normalize_panel(raw: dict, idx: int, characters: dict[str, dict], style: str) -> dict:
    p = {
        "id": idx,
        "shot": raw.get("shot") if raw.get("shot") in SHOT_TYPES else "medium",
        "characters": [c for c in raw.get("characters", []) if c in characters],
        "scene": raw.get("scene", ""),
        "action": raw.get("action", ""),
        "emotion": raw.get("emotion", ""),
        "narration": raw.get("narration", ""),
        "dialogue": [d for d in raw.get("dialogue", []) if isinstance(d, dict) and d.get("text")][:2],
        "prompt_override": None,
    }
    p["prompt"] = build_prompt(p, characters, style)
    return p


# ------------------------------------------------------------------------ main

def novel_to_panels(text: str, llm: LLMFn, characters: list[dict] | None = None,
                    style: str = DEFAULT_STYLE, log=print) -> dict:
    characters = extract_characters(llm, text, characters or [])
    log(f"characters: {', '.join(c['name'] for c in characters)}")
    by_id = {c["id"]: c for c in characters}

    panels, prev_tail = [], ""
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks, 1):
        raws = split_to_panels(llm, chunk, characters, prev_tail)
        for raw in raws:
            panels.append(normalize_panel(raw, len(panels) + 1, by_id, style))
        if panels:
            prev_tail = json.dumps(panels[-1], ensure_ascii=False)
        log(f"chunk {i}/{len(chunks)} -> {len(panels)} panels total")

    return {"style": style, "characters": characters, "panels": panels}


def rebuild_prompts(data: dict) -> dict:
    """Re-generate prompts after you hand-edit the JSON (overrides are kept)."""
    by_id = {c["id"]: c for c in data["characters"]}
    for p in data["panels"]:
        p["prompt"] = build_prompt(p, by_id, data.get("style", DEFAULT_STYLE))
    return data


def final_prompt(panel: dict) -> str:
    """The prompt the image stage should use: manual override wins."""
    return panel.get("prompt_override") or panel["prompt"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Novel text -> webtoon panel JSON")
    ap.add_argument("input", help="novel .txt file (UTF-8)")
    ap.add_argument("-o", "--output", default="outputs/panels.json")
    ap.add_argument("--characters", default="characters/characters.json",
                    help="character sheet JSON; created/extended automatically")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--style", default=DEFAULT_STYLE)
    ap.add_argument("--rebuild", action="store_true",
                    help="only rebuild prompts in an edited panels JSON (input = that JSON)")
    args = ap.parse_args(argv)

    if args.rebuild:
        path = Path(args.input)
        data = rebuild_prompts(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"prompts rebuilt: {path}")
        return

    text = Path(args.input).read_text(encoding="utf-8")
    char_path = Path(args.characters)
    known = json.loads(char_path.read_text(encoding="utf-8")) if char_path.exists() else []

    llm = make_openai_llm(args.base_url, args.model)
    data = novel_to_panels(text, llm, known, args.style)

    char_path.parent.mkdir(parents=True, exist_ok=True)
    char_path.write_text(json.dumps(data["characters"], ensure_ascii=False, indent=2), encoding="utf-8")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(data['panels'])} panels -> {out}")
    print("Review/edit the JSON, then run with --rebuild if you changed panel fields.")


if __name__ == "__main__":
    sys.exit(main())
