from __future__ import annotations

from cleanbot.rag.chunking import split_structured_text, tokenize_for_bm25


def test_structured_chunking_preserves_numbered_entries() -> None:
    text = """# 使用指南
## 故障
1. 开机没有反应；检查电源。
2. 主刷不转；清理毛发。

## 保养
1. 每周擦拭传感器。
"""
    chunks = split_structured_text(text)
    assert [chunk.text for chunk in chunks] == [
        "1. 开机没有反应；检查电源。",
        "2. 主刷不转；清理毛发。",
        "1. 每周擦拭传感器。",
    ]
    assert chunks[0].section == "故障"
    assert chunks[-1].section == "保养"


def test_bm25_tokenizer_contains_chinese_bigrams_and_latin_words() -> None:
    tokens = tokenize_for_bm25("HEPA滤网堵塞 WiFi2.4G")
    assert "hepa" in tokens
    assert "滤网" in tokens
    assert "堵塞" in tokens
    assert "wifi2" in tokens
