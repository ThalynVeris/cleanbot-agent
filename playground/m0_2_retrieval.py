"""M0.2 手敲练习：对照对话中给出的参考答案，边敲边写中文注释。

完成后运行：python -m pytest playground/test_m0_2_retrieval.py -q
"""

from __future__ import annotations

import re  # noqa: F401  手敲 tokenize_for_bm25 时会用到

from cleanbot.core.schemas import KnowledgeHit


def tokenize_for_bm25(text: str) -> list[str]:
    raise NotImplementedError("对照参考答案手敲实现")


def fuse_rrf(
    dense: list[KnowledgeHit],
    sparse: list[KnowledgeHit],
    rank_constant: int = 60,
) -> list[KnowledgeHit]:
    raise NotImplementedError("对照参考答案手敲实现")
