from __future__ import annotations

import asyncio
import uuid

from cleanbot.api.container import get_container
from cleanbot.core.schemas import ChatRequest


async def main() -> None:
    container = get_container()
    container.initialize()
    request = ChatRequest(
        session_id=f"smoke-{uuid.uuid4()}",
        user_id="1003",
        message="扫地机器人主刷总被宠物毛发缠住，应该怎么处理？",
        month="2025-03",
    )
    text = ""
    sources = 0
    async for event in container.agent.stream(request):
        if event.event == "token":
            text += event.data.get("text", "")
        elif event.event == "source":
            sources += 1
        elif event.event == "error":
            raise RuntimeError(event.data.get("message", "smoke test failed"))
    print(f"sources={sources}")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())

