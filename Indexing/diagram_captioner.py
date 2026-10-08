# Indexing/diagram_captioner.py
from google import genai
from Indexing.cache_utils import load_cache, save_cache

CACHE_PATH = "Indexing/captions_cache.json"

CAPTION_PROMPT = """You are looking at a page from a hardware datasheet.
Describe everything relevant for an engineer troubleshooting this component:
- If it's a pinout diagram: list every labeled pin and its function
- If it's a schematic: describe every connection between labeled components
- If it's a block diagram: describe each block and how signals flow between them
- If it's a mechanical/package drawing: note the package type and key dimensions
Be specific and literal. Do not skip labels or pin names, even minor ones."""

def caption_diagram(image_path: str, client: genai.Client, model: str = "gemini-3.5-flash-lite") -> str | None:
    cache = load_cache(CACHE_PATH)

    if image_path in cache:
        print(f"[cache hit] {image_path}")
        return cache[image_path]

    print(f"[cache miss] {image_path} — calling Gemini")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=model,
        contents=[
            {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
            CAPTION_PROMPT,
        ],
    )
    caption = response.text

    if not caption:
        finish_reason = response.candidates[0].finish_reason if response.candidates else "unknown"
        print(f"[warning] Gemini returned no caption for {image_path} (finish_reason={finish_reason}). Skipping this page.")
        return None  # deliberately NOT cached, so a retry later can try again

    cache[image_path] = caption
    save_cache(CACHE_PATH, cache)

    return caption