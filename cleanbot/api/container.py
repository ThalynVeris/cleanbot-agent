from __future__ import annotations

from contextlib import ExitStack
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.models import get_chat_model
from cleanbot.db.database import Database
from cleanbot.device_mcp.client import DeviceMCPClient
from cleanbot.device_mcp.server import create_device_mcp
from cleanbot.rag.knowledge_base import KnowledgeBase
from cleanbot.rag.retriever import HybridRetriever
from cleanbot.tools.weather import WeatherClient
from cleanbot.workflow.device_approval import (
    DeviceApprovalWorkflow,
)
from cleanbot.workflow.device_control import (
    DeviceControlService,
)
from cleanbot.workflow.graph import CleanBotGraph
from cleanbot.workflow.service import AgentService


class AppContainer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.database = Database(self.settings)
        self.model = get_chat_model()
        self.knowledge_base = KnowledgeBase(self.database, self.settings)
        self.retriever = HybridRetriever(self.knowledge_base, self.settings)
        self.weather = WeatherClient(self.settings)
        self._resources = ExitStack()

        self.device_checkpointer = self._resources.enter_context(
            SqliteSaver.from_conn_string(str(self.settings.device_checkpoint_path))
        )
        self.device_approval = DeviceApprovalWorkflow(self.device_checkpointer)
        mcp_transport = self.settings.device_mcp_url or create_device_mcp(self.database)
        self.device_mcp = DeviceMCPClient(
            mcp_transport,
            timeout_seconds=self.settings.device_mcp_timeout_seconds,
            token=self.settings.device_mcp_token,
        )

        self.device_control = DeviceControlService(
            database=self.database,
            approval_workflow=(self.device_approval),
            mcp_client=self.device_mcp,
            model=self.model,
        )
        self.graph = CleanBotGraph(
            database=self.database,
            retriever=self.retriever,
            weather=self.weather,
            model=self.model,
            settings=self.settings,
            device_control=self.device_control,
        )
        self.agent = AgentService(self.database, self.graph, self.model)

    def initialize(self) -> None:
        self.database.create_schema()
        self.database.seed_demo_data()

    def close(self) -> None:
        self._resources.close()


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer()
