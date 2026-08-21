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


def test_long_numbered_entry_uses_configured_overlap() -> None:
    text = """# 故障

1. 主刷缠绕宠物毛发时应先断电再拆下主刷清理轴承两端
"""

    chunks = split_structured_text(
        text,
        max_chars=12,
        overlap_chars=4,
    )

    assert [chunk.text for chunk in chunks] == [
        "1. 主刷缠绕宠物毛发时",
        "物毛发时应先断电再拆下主",
        "再拆下主刷清理轴承两端",
    ]

    assert chunks[0].text[-4:] == chunks[1].text[:4]
    assert chunks[1].text[-4:] == chunks[2].text[:4]
    assert all(chunk.section == "故障" for chunk in chunks)


def test_headings_inside_pdf_style_text_update_sections() -> None:
    text = """### 语音交互
1. 支持语音控制。
## 四、维护与保养类
### 日常维护
2. 滚刷缠绕毛发时应清理。
### 故障处理
3. 机器人被卡住时应断电检查。
"""

    chunks = split_structured_text(text)

    assert [(chunk.section, chunk.text) for chunk in chunks] == [
        ("语音交互", "1. 支持语音控制。"),
        ("日常维护", "2. 滚刷缠绕毛发时应清理。"),
        ("故障处理", "3. 机器人被卡住时应断电检查。"),
    ]
    assert all("###" not in chunk.text for chunk in chunks)


def test_bm25_tokenizer_contains_chinese_bigrams_and_latin_words() -> None:
    tokens = tokenize_for_bm25("HEPA滤网堵塞 WiFi2.4G")
    assert "hepa" in tokens
    assert "滤网" in tokens
    assert "堵塞" in tokens
    assert "wifi2" in tokens
