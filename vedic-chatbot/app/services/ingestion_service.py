import os
import re
import json
from typing import List, Dict, Tuple

from app.config import settings


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "texts")


# ---------------------------------------------------------------------------
# Text splitting (no langchain dependency)
# ---------------------------------------------------------------------------

def _recursive_split(text: str, separators: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Recursively split text by separators until chunks are within chunk_size."""
    if not separators:
        # Base case: split by character
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    sep = separators[0]
    remaining_seps = separators[1:]

    parts = text.split(sep) if sep else list(text)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(part) > chunk_size:
                sub_chunks = _recursive_split(part, remaining_seps, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # Add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        return overlapped

    return chunks


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _recursive_split(text, separators, chunk_size, overlap)
    return [c for c in raw_chunks if len(c.strip()) > 20]


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

SOURCE_PATTERN = re.compile(r"\[SOURCE:\s*([^\]]+)\]")
CHAPTER_PATTERN = re.compile(r"\[CHAPTER:\s*([^\]]+)\]")
VERSE_PATTERN = re.compile(r"\[VERSE:\s*([^\]]+)\]")


def _parse_metadata_from_marker(line: str) -> Dict:
    meta = {}
    s = SOURCE_PATTERN.search(line)
    c = CHAPTER_PATTERN.search(line)
    v = VERSE_PATTERN.search(line)
    if s:
        meta["source"] = s.group(1).strip().lower().replace(" ", "_")
    if c:
        meta["chapter"] = c.group(1).strip()
    if v:
        meta["verse"] = v.group(1).strip()
    return meta


def _parse_txt_file(filepath: str) -> List[Tuple[str, Dict]]:
    """Parse a .txt file with [SOURCE:] markers. Returns list of (text_block, metadata)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = []
    current_meta: Dict = {"source": os.path.splitext(os.path.basename(filepath))[0]}
    current_lines: List[str] = []

    for line in content.splitlines():
        if SOURCE_PATTERN.search(line):
            # Save previous block
            if current_lines:
                blocks.append(("\n".join(current_lines).strip(), dict(current_meta)))
                current_lines = []
            # Update metadata
            new_meta = _parse_metadata_from_marker(line)
            current_meta.update(new_meta)
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append(("\n".join(current_lines).strip(), dict(current_meta)))

    return [(text, meta) for text, meta in blocks if text.strip()]


def _parse_json_file(filepath: str) -> List[Tuple[str, Dict]]:
    """Parse a .json file. Expects list of {text, source, chapter, verse} objects."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [(item.get("text", ""), {k: v for k, v in item.items() if k != "text"}) for item in data if item.get("text")]
    return []


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def load_all_documents() -> Tuple[List[str], List[Dict]]:
    """Load and chunk all Vedic texts from data/texts/. Returns (texts, metadatas)."""
    all_texts: List[str] = []
    all_metas: List[Dict] = []

    if not os.path.exists(DATA_DIR):
        print(f"Warning: Data directory not found: {DATA_DIR}")
        return all_texts, all_metas

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".json"))]
    if not files:
        print(f"Warning: No .txt or .json files found in {DATA_DIR}")
        return all_texts, all_metas

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"  Loading: {filename}")

        try:
            if filename.endswith(".txt"):
                blocks = _parse_txt_file(filepath)
            else:
                blocks = _parse_json_file(filepath)
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue

        for block_idx, (text, meta) in enumerate(blocks):
            chunks = chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = dict(meta)
                chunk_meta["chunk_index"] = f"{block_idx}_{chunk_idx}"
                chunk_meta["filename"] = filename
                all_texts.append(chunk)
                all_metas.append(chunk_meta)

    print(f"Total chunks prepared: {len(all_texts)}")
    return all_texts, all_metas
