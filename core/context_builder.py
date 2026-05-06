"""
聚合来自多个来源的上下文，构建 prompt-ready 结构。

供 decision_engine 和 reply_engine 使用。
实际上是 KnowledgeQuery + PersonaProvider 的简单聚合，
核心格式化逻辑保留在各 Engine 中。
"""
from .config import PluginConfig
from .memory.query import KnowledgeQuery
from .persona import PersonaProvider


class ContextBuilder:
    """集中构建 LLM 上下文。"""

    def __init__(
        self, config: PluginConfig,
        knowledge_query: KnowledgeQuery,
        persona_provider: PersonaProvider,
    ):
        self.cfg = config
        self.knowledge_query = knowledge_query
        self.persona_provider = persona_provider
