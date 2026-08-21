from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^#{1,6}\s+.+$")


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    section: str


def split_structured_text(text: str, max_chars: int = 700, overlap_chars: int = 100) -> list[TextChunk]:
    """Split Chinese manuals without cutting the numbered FAQ entries whenever possible."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # PDF text extraction often removes blank lines between sections. Make every
    # Markdown heading start a paragraph so headings in the middle of a page can
    # update the section instead of leaking into the previous numbered entry.
    normalized = re.sub(r"(?m)(?=^#{1,6}\s+)", "\n\n", normalized).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    buffer = ""
    current_section = "正文"

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(TextChunk(text=buffer.strip(), section=current_section))
        buffer = ""

    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        while lines and HEADING_PATTERN.match(lines[0].strip()):
            flush()
            current_section = lines.pop(0).strip().lstrip("# ")
        paragraph = "\n".join(lines).strip()
        if not paragraph:
            continue

        plain_heading = (
            len(paragraph) <= 60
            and "\n" not in paragraph
            and not any(mark in paragraph for mark in "：；。！？")
            and any(word in paragraph for word in ("指南", "问答", "故障", "维护", "保养", "选购"))
        )
        if plain_heading:
            flush()
            current_section = paragraph
            continue

        numbered_items = [item.strip() for item in re.split(r"(?m)(?=^\d+\.\s*)", paragraph) if item.strip()]
        if len(numbered_items) > 1 or re.match(r"^\d+\.\s*", paragraph):
            flush()
            for item in numbered_items:
                if len(item) <= max_chars:
                    chunks.append(TextChunk(text=item, section=current_section))
                    continue
                start = 0
                while start < len(item):
                    piece = item[start : start + max_chars].strip()
                    if piece:
                        chunks.append(TextChunk(text=piece, section=current_section))
                    if start + max_chars >= len(item):
                        break
                    start += max_chars - overlap_chars
            continue

        if len(paragraph) > max_chars:
            flush()
            start = 0
            while start < len(paragraph):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    chunks.append(TextChunk(text=piece, section=current_section))
                if start + max_chars >= len(paragraph):
                    break
                start += max_chars - overlap_chars
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            flush()
            buffer = paragraph

    flush()
    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    """A dependency-light tokenizer that retains Latin words and Chinese uni/bi-grams."""
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9]+", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese: list[str] = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin + chinese
