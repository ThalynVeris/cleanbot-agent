from __future__ import annotations

from functools import lru_cache

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.models import get_chat_model
from cleanbot.db.database import Database
from cleanbot.rag.knowledge_base import KnowledgeBase
from cleanbot.rag.retriever import HybridRetriever
from cleanbot.tools.weather import WeatherClient
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
        self.graph = CleanBotGraph(
            database=self.database,
            retriever=self.retriever,
            weather=self.weather,
            model=self.model,
            settings=self.settings,
        )
        self.agent = AgentService(self.database, self.graph, self.model)

    def initialize(self) -> None:
        self.database.create_schema()
        self.database.seed_demo_data()


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer()
