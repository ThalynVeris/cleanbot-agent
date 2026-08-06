from __future__ import annotations

import httpx
from langchain_core.messages import AIMessage

from cleanbot.core.config import Settings
from cleanbot.core.schemas import Intent
from cleanbot.tools.weather import WeatherClient
from cleanbot.workflow.router import IntentRouter


async def test_deterministic_routes_do_not_require_model() -> None:
    router = IntentRouter()
    assert await router.classify("给我生成使用报告") == Intent.REPORT
    assert await router.classify("现在湿度高吗") == Intent.ENVIRONMENT
    assert await router.classify("扫地机器人主刷不转") == Intent.KNOWLEDGE
    assert await router.classify("你好") == Intent.SMALLTALK
    assert await router.classify("写一个快速排序") == Intent.OUT_OF_SCOPE


async def test_model_fallback_parses_json_without_provider_structured_output() -> None:
    class JsonModel:
        async def ainvoke(self, prompt):
            return AIMessage(content='```json\n{"intent":"knowledge","reason":"product question"}\n```')

    router = IntentRouter(JsonModel())  # type: ignore[arg-type]
    assert await router.classify("卧室用多大噪声合适") == Intent.KNOWLEDGE


async def test_weather_success(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in str(request.url):
            return httpx.Response(200, json={"results": [{"latitude": 31.2, "longitude": 121.5}]})
        return httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 26.5,
                    "apparent_temperature": 31.2,
                    "relative_humidity_2m": 70,
                    "wind_speed_10m": 8.0,
                    "time": "2026-07-21T12:15",
                },
                "hourly": {
                    "time": ["2026-07-21T00:00", "2026-07-21T12:00"],
                    "precipitation_probability": [10, 70],
                },
            },
        )

    result = await WeatherClient(settings, httpx.MockTransport(handler)).current("上海")
    assert result.ok is True
    assert result.temperature_c == 26.5
    assert result.apparent_temperature_c == 31.2
    assert result.precipitation_probability == 70


async def test_weather_failure_is_explicit(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    result = await WeatherClient(settings, httpx.MockTransport(handler)).current("上海")
    assert result.ok is False
    assert "网络、DNS 或 VPN 代理" in (result.error or "")
