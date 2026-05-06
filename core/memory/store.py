"""
KV 持久化的群聊知识存储。

每个群一个 KV blob: grp_knowledge_{group_id}
写入/读取均通过 AstrBot KV store 进行。
"""
import time
from typing import Any

from astrbot.api import logger


class MemoryStore:
    """KV-backed 群知识持久化。"""

    KNOWLEDGE_KEY_PREFIX = "grp_knowledge"

    def __init__(self, star_instance):
        self.plugin = star_instance
        self._cache: dict[str, dict] = {}

    def _key(self, group_id: str) -> str:
        return f"{self.KNOWLEDGE_KEY_PREFIX}_{group_id}"

    async def load(self, group_id: str) -> dict[str, Any]:
        """从 KV store 加载群知识，带内存缓存。"""
        if group_id in self._cache:
            return self._cache[group_id]

        data = await self.plugin.get_kv_data(self._key(group_id), None)
        if data is None:
            data = self._empty_knowledge(group_id)
        self._cache[group_id] = data
        return data

    async def save(self, group_id: str, knowledge: dict) -> None:
        """持久化群知识。"""
        knowledge["last_updated"] = time.time()
        self._cache[group_id] = knowledge
        await self.plugin.put_kv_data(self._key(group_id), knowledge)

    async def update_slang(self, group_id: str, new_slang: list[dict], max_entries: int = 30) -> dict:
        """合并新黑话条目。"""
        knowledge = await self.load(group_id)
        existing = {s["term"]: s for s in knowledge.get("slang", [])}
        for entry in new_slang:
            term = entry.get("term", "")
            if term:
                existing[term] = entry
        knowledge["slang"] = sorted(
            existing.values(),
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:max_entries]
        await self.save(group_id, knowledge)
        return knowledge

    async def update_atmosphere(self, group_id: str, atmosphere: dict) -> dict:
        """更新群氛围。"""
        knowledge = await self.load(group_id)
        knowledge["atmosphere"] = {**knowledge.get("atmosphere", {}), **atmosphere}
        await self.save(group_id, knowledge)
        return knowledge

    async def update_member_impressions(
        self, group_id: str, impressions: dict[str, dict]
    ) -> dict:
        """合并成员印象。兼容 LLM 直接返回 list 的格式。"""
        knowledge = await self.load(group_id)
        existing = knowledge.get("member_impressions", {})
        for uid, imp in impressions.items():
            # 归一化：LLM 可能直接返回 traits 列表而非 {"traits": [...]}
            if isinstance(imp, list):
                imp = {"traits": [str(t) for t in imp]}
            if uid in existing:
                existing_traits = set(existing[uid].get("traits", []))
                new_traits = set(imp.get("traits", []))
                existing[uid]["traits"] = list(existing_traits | new_traits)[:10]
                existing[uid]["last_updated"] = time.time()
            else:
                existing[uid] = {
                    **imp,
                    "first_seen": time.time(),
                    "last_updated": time.time(),
                }
        knowledge["member_impressions"] = existing
        await self.save(group_id, knowledge)
        return knowledge

    async def add_notable_events(self, group_id: str, events: list[str]) -> dict:
        """追加值得记录的事件。"""
        knowledge = await self.load(group_id)
        existing = knowledge.get("notable_events", [])
        existing.extend(events)
        knowledge["notable_events"] = existing[-20:]
        await self.save(group_id, knowledge)
        return knowledge

    async def update_relationships(self, group_id: str, relationships: dict[str, str]) -> dict:
        """更新角色关系。"""
        knowledge = await self.load(group_id)
        existing = knowledge.get("character_relationships", {})
        existing.update(relationships)
        knowledge["character_relationships"] = existing
        await self.save(group_id, knowledge)
        return knowledge

    @staticmethod
    def _empty_knowledge(group_id: str) -> dict[str, Any]:
        return {
            "group_id": group_id,
            "atmosphere": {
                "overall_vibe": "neutral",
                "activity_level": "moderate",
                "common_topics": [],
            },
            "slang": [],
            "member_impressions": {},
            "character_relationships": {},
            "notable_events": [],
            "last_extraction_time": 0.0,
            "last_updated": time.time(),
        }
