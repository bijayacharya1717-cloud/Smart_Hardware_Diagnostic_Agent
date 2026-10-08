# Indexing/cache_utils.py
import json
from pathlib import Path


def load_cache(cache_path: str) -> dict:
    """Load the cache file if it exists, otherwise start with an empty dict."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_cache(cache_path: str, cache: dict) -> None:
    """Write the current cache dict back to disk as JSON."""
    Path(cache_path).write_text(json.dumps(cache, indent=2))
