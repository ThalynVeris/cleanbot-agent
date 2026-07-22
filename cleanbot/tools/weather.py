from __future__ import annotations

import asyncio
from typing import Any

import httpx

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.schemas import WeatherResult


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
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    await asyncio.sleep(0.15)
        return WeatherResult(ok=False, city=city, error=last_error)

    async def _fetch(self, city: str) -> WeatherResult:
        timeout = httpx.Timeout(self.settings.weather_timeout_seconds)
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
                    "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                    "hourly": "precipitation_probability",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
            )
            weather_response.raise_for_status()
            payload: dict[str, Any] = weather_response.json()
            current = payload["current"]
            hourly_probability = payload.get("hourly", {}).get("precipitation_probability", [])
            probability = float(hourly_probability[0]) if hourly_probability else None
            return WeatherResult(
                ok=True,
                city=city,
                temperature_c=float(current["temperature_2m"]),
                relative_humidity=float(current["relative_humidity_2m"]),
                precipitation_probability=probability,
                wind_speed_kmh=float(current["wind_speed_10m"]),
                observed_at=str(current["time"]),
            )
