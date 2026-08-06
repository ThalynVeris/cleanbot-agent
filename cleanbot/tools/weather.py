from __future__ import annotations

import asyncio
from typing import Any

import httpx

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.schemas import WeatherResult


def _find_current_hour_probability(payload: dict[str, Any]) -> float | None:
    current_time = str(payload["current"]["time"])
    current_hour = current_time[:13]

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    probabilities = hourly.get("precipitation_probability", [])

    for time, probability in zip(times, probabilities, strict=False):
        if str(time).startswith(current_hour):
            if probability is None:
                return None
            return float(probability)
    return None


class WeatherClient:
    """Open-Meteo adapter. It uses live data and returns an explicit failure instead of fabricated weather."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings or get_settings()
        self.transport = transport

    async def current(self, city: str) -> WeatherResult:
        last_error = "weather service unavailable"
        for attempt in range(2):
            try:
                return await self._fetch(city)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = self._error_message(exc)
                if attempt == 0:
                    await asyncio.sleep(0.15)
        return WeatherResult(ok=False, city=city, error=last_error)

    async def _fetch(self, city: str) -> WeatherResult:
        timeout_seconds = self.settings.weather_timeout_seconds
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=min(3.0, timeout_seconds),
        )
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            geo_response = await client.get(
                self.GEOCODING_URL,
                params={"name": city, "count": 1, "language": "zh", "format": "json"},
            )
            geo_response.raise_for_status()
            results = geo_response.json().get("results", [])
            if not results:
                return WeatherResult(ok=False, city=city, error="city not found")
            location = results[0]

            weather_response = await client.get(
                self.FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": (
                        "temperature_2m,apparent_temperature,"
                        "relative_humidity_2m,precipitation,wind_speed_10m"
                    ),
                    "hourly": "precipitation_probability",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
            )
            weather_response.raise_for_status()
            payload: dict[str, Any] = weather_response.json()
            current = payload["current"]
            probability = _find_current_hour_probability(payload)
            return WeatherResult(
                ok=True,
                city=city,
                temperature_c=float(current["temperature_2m"]),
                apparent_temperature_c=float(current["apparent_temperature"]),
                relative_humidity=float(current["relative_humidity_2m"]),
                precipitation_probability=probability,
                wind_speed_kmh=float(current["wind_speed_10m"]),
                observed_at=str(current["time"]),
            )

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if isinstance(exc, httpx.ConnectError):
            return "无法连接 Open-Meteo，请检查网络、DNS 或 VPN 代理后重试"
        if isinstance(exc, httpx.TimeoutException):
            return "连接 Open-Meteo 超时，请检查网络或稍后重试"
        details = str(exc).strip()
        return f"{type(exc).__name__}: {details or '天气服务返回了无详细信息的错误'}"
