from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel

from cleanbot.core.schemas import Intent, IntentDecision


class IntentRouter:
    REPORT_WORDS = ("报告", "月报", "使用记录", "耗材情况", "清洁数据")
    WEATHER_WORDS = ("天气", "湿度", "下雨", "气温", "潮湿", "干燥", "环境建议")
    ROBOT_WORDS = (
        "扫地机",
        "扫地机器人",
        "扫拖",
        "拖布",
        "主刷",
        "边刷",
        "尘盒",
        "滤网",
        "回充",
        "建图",
        "地图",
        "导航",
        "避障",
        "清扫",
        "吸力",
        "水箱",
        "地毯",
        "传感器",
        "电池",
        "续航",
        "防跌落",
        "耗材",
        "噪音",
        "分贝",
    )
    CAPABILITY_MESSAGES = {
        "你能做什么",
        "你会什么",
        "你有什么功能",
        "有什么功能",
        "支持哪些功能",
        "怎么使用",
        "如何使用",
        "怎么使用这个客服",
        "如何使用这个客服",
    }
    GREETINGS = {"你好", "您好", "hi", "hello", "在吗", "谢谢", "再见"}
    DEVICE_PHRASES = (
        "查看设备状态",
        "设备状态",
        "当前状态",
        "机器人状态",
        "剩余电量",
        "还剩多少电",
        "还有多少电",
        "查看耗材",
        "耗材剩余",
        "耗材还剩",
        "耗材状态",
        "开始清扫",
        "开始打扫",
        "启动清扫",
        "暂停清扫",
        "暂停打扫",
        "返回充电座",
        "回到充电座",
        "立即回充",
        "开始回充",
    )

    DEVICE_BLOCKERS = (
        "为什么",
        "怎么办",
        "怎么解决",
        "如何",
        "注意",
        "教程",
        "介绍",
        "说明",
        "无法",
        "不能",
        "失败",
        "故障",
        "异常",
        "不要",
    )

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.model = model

    def deterministic(self, message: str) -> Intent | None:
        normalized = re.sub(r"[，。！？!?\s]", "", message.lower())
        if any(word in message for word in self.REPORT_WORDS):
            return Intent.REPORT
        if any(word in message for word in self.WEATHER_WORDS):
            return Intent.ENVIRONMENT
        if normalized in self.CAPABILITY_MESSAGES:
            return Intent.SMALLTALK
        if normalized in self.GREETINGS:
            return Intent.SMALLTALK
        has_device_phrase = any(phrase in message for phrase in self.DEVICE_PHRASES)
        has_blocker = any(blocker in message for blocker in self.DEVICE_BLOCKERS)

        if has_device_phrase and not has_blocker:
            return Intent.DEVICE
        if any(word in message for word in self.ROBOT_WORDS):
            return Intent.KNOWLEDGE
        return None

    async def classify(self, message: str) -> Intent:
        direct = self.deterministic(message)
        if direct is not None:
            return direct
        if self.model is None:
            return Intent.OUT_OF_SCOPE
        try:
            response = await self.model.ainvoke(
                """Classify the user's request for a cleaning-robot customer-service system.
Allowed intents: knowledge, report, environment, device, smalltalk, out_of_scope.
Use device only when the user wants to read or control their current
simulated device.
Questions about device faults, maintenance, instructions or product
knowledge belong to knowledge, not device.
Use knowledge only for cleaning robots, their purchase, use, maintenance, or troubleshooting.
Return only one JSON object with exactly these fields:
{"intent":"knowledge","reason":"brief classification reason"}
Do not use Markdown fences and never return private reasoning.

User request: """
                + message
            )
            content = getattr(response, "content", response)
            if not isinstance(content, str):
                content = str(content)
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                content = content[start : end + 1]
            return IntentDecision.model_validate_json(content).intent
        except Exception:
            return Intent.OUT_OF_SCOPE
