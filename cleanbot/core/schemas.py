from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    KNOWLEDGE = "knowledge"
    REPORT = "report"
    ENVIRONMENT = "environment"
    DEVICE = "device"
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"


class IntentDecision(BaseModel):
    intent: Intent
    reason: str = Field(description="Short classification reason; never include chain-of-thought")


class DeviceOperation(str, Enum):
    READ_STATUS = "read_status"
    READ_CONSUMABLE = "read_consumable"
    START_CLEANING = "start_cleaning"
    PAUSE_CLEANING = "pause_cleaning"
    RETURN_TO_DOCK = "return_to_dock"
    UNKNOWN = "unknown"


class DeviceIntentDecision(BaseModel):
    operation: DeviceOperation
    confidence: float = Field(ge=0, le=1)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)
    month: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value


class SourceRef(BaseModel):
    document_id: str
    chunk_id: str
    source: str
    page: int | None = None
    section: str | None = None
    score: float = 0.0
    excerpt: str = ""


class KnowledgeHit(SourceRef):
    content: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float | None = None

    def to_source(self) -> SourceRef:
        return SourceRef(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            source=self.source,
            page=self.page,
            section=self.section,
            score=self.score,
            excerpt=self.content[:180],
        )


class DeviceReport(BaseModel):
    user_id: str
    month: str
    features: str
    efficiency: str
    consumables: str
    comparison: str


class DeviceStatusView(BaseModel):
    device_id: str
    user_id: str
    model: str
    status: Literal[
        "docked",
        "cleaning",
        "paused",
        "returning_to_dock",
    ]
    battery_percent: int = Field(ge=0, le=100)
    simulated: bool = True


class ConsumableStatusView(BaseModel):
    device_id: str
    user_id: str
    consumable_percent: int = Field(ge=0, le=100)
    replacement_recommended: bool
    simulated: bool = True


class DeviceCapabilitiesView(BaseModel):
    device_id: str
    model: str
    supported_actions: list[str]
    readable_properties: list[str]
    simulated: bool = True


class DeviceActionResult(BaseModel):
    ok: bool
    action_id: str
    device_id: str
    action: Literal[
        "start_cleaning",
        "pause_cleaning",
        "return_to_dock",
    ]
    action_status: Literal["succeeded", "failed"]
    device_status: Literal[
        "docked",
        "cleaning",
        "paused",
        "returning_to_dock",
    ]
    idempotent_replay: bool = False
    error_type: str | None = None
    message: str


class DeviceActionView(BaseModel):
    id: str
    user_id: str
    device_id: str
    session_id: str
    action: Literal[
        "start_cleaning",
        "pause_cleaning",
        "return_to_dock",
    ]
    status: Literal[
        "pending",
        "approved",
        "rejected",
        "succeeded",
        "failed",
        "expired",
    ]
    checkpoint_thread_id: str
    approval_expires_at: datetime
    decided_at: datetime | None = None
    executed_at: datetime | None = None
    error_type: str | None = None
    created_at: datetime


class StoredMessage(BaseModel):
    id: int
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    sources: list[SourceRef] = Field(default_factory=list)
    created_at: datetime


class ChatSessionSummary(BaseModel):
    id: str
    user_id: str
    title: str
    message_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class DemoUser(BaseModel):
    id: str
    display_name: str
    city: str


class ChatEvent(BaseModel):
    event: Literal["status", "source", "token", "done", "error"]
    request_id: str
    data: dict[str, Any]


class IngestResult(BaseModel):
    document_id: str
    filename: str
    content_hash: str
    chunk_count: int
    status: Literal["created", "updated", "unchanged"]


class WeatherResult(BaseModel):
    ok: bool
    city: str
    temperature_c: float | None = None
    apparent_temperature_c: float | None = None
    relative_humidity: float | None = None
    precipitation_probability: float | None = None
    wind_speed_kmh: float | None = None
    observed_at: str | None = None
    error: str | None = None
