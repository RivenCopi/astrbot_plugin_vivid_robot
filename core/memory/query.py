"""
查询群聊知识，用于注入 LLM prompt。
"""
from .store import MemoryStore


class KnowledgeQuery:
    """检索并格式化群知识用于 prompt。"""

    def __init__(self, store: MemoryStore):
        self.store = store

    async def get_context_for_prompt(self, group_id: str) -> dict:
        """获取所有相关知识，格式化为可注入 prompt 的 dict。"""
        knowledge = await self.store.load(group_id)

        slang_entries = knowledge.get("slang", [])[:8]
        atmos = knowledge.get("atmosphere", {})
        impressions = knowledge.get("member_impressions", {})
        top_impressions = dict(
            sorted(
                impressions.items(),
                key=lambda x: len(x[1].get("traits", [])),
                reverse=True,
            )[:5]
        )

        return {
            "atmosphere": atmos,
            "slang": slang_entries,
            "member_impressions": top_impressions,
            "character_relationships": knowledge.get("character_relationships", {}),
            "notable_events": knowledge.get("notable_events", [])[-5:],
        }

    async def get_slang_meaning(self, group_id: str, term: str) -> str:
        """查找单个黑话的含义。"""
        knowledge = await self.store.load(group_id)
        for entry in knowledge.get("slang", []):
            if entry.get("term") == term:
                return entry.get("meaning", "")
        return ""

    async def get_member_impression(self, group_id: str, user_id: str) -> dict:
        """获取指定成员的印象。"""
        knowledge = await self.store.load(group_id)
        return knowledge.get("member_impressions", {}).get(user_id, {})
