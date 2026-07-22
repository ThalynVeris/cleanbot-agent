from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cleanbot.api.container import get_container
from cleanbot.evaluation.runner import EvaluationRunner
from cleanbot.workflow.router import IntentRouter


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CleanBot routing and retrieval")
    parser.add_argument("--dataset", default="evaluation/questions.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--answer-sample-size", type=int, default=0)
    parser.add_argument("--json-output", default="reports/evaluation/latest.json")
    parser.add_argument("--markdown-output", default="reports/evaluation/latest.md")
    args = parser.parse_args()

    container = get_container()
    container.initialize()
    runner = EvaluationRunner(
        retriever=container.retriever,
        router=IntentRouter(container.model),
        model=container.model,
        settings=container.settings,
    )
    result = await runner.run(Path(args.dataset), args.limit, args.answer_sample_size)
    runner.save(result, Path(args.json_output), Path(args.markdown_output))
    print(Path(args.markdown_output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(async_main())
