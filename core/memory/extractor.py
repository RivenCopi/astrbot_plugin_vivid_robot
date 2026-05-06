"""
使用 LLM 从对话批次中提取群聊知识。
每 N 条消息触发一次，异步执行，不阻塞回复路径。
提取后对黑话进行知识库补充（利用 LLM 训练数据中的百科知识，无需联网搜索）。
"""
import json
import time

from astrbot.api import logger
from astrbot.core.star.context import Context

from ..config import PluginConfig
from ..models import MessageRecord
from ...utils.llm_utils import safe_llm_generate
from .store import MemoryStore


EXTRACTION_PROMPT = """你是群聊分析师。分析以下对话，提取信息并输出 JSON。

—— 重要规则 ——
1. 黑话/术语(slang)：如果群友提到某个专有名词（如游戏名、角色名、圈内术语），请利用你的训练数据知识推断其完整含义。例如：
   - "黑龙" → 如果是怪物猎人语境，应写 "怪物猎人系列中的黑龙(Fatalis)，一只强大的古龙"
   - "圣杯战争" → "Fate系列中的圣杯战争概念"
   不要只写"黑龙"两个字，尽量用百科式语言补充说明。
2. 成员印象(member_impressions)：必须按 user_id 精确归因。谁是说话人就给谁加印象，不要错加到被提及的人身上。
   格式：{{"user_id": {{"traits": ["特点1"], "relationship_to_bot": "关系"}}}}
   其中 user_id 包含名字和QQ号，如 "JRPG爱好者(982843856)"。
3. 角色关系：描述成员之间的互动模式。
4. 值得记录的事件：群内发生的有趣/重要事件，包含参与者和简要描述。

已有知识（仅作参考，无需重复）：
{existing_knowledge}

对话记录（格式：[名字(QQ号)]: 内容）：
{conversation}

输出 JSON（只包含新的或有变化的信息，没有的字段输出空数组/对象）：
{{
  "slang": [{{"term": "词汇", "meaning": "详细含义(利用你的知识补充)", "example": "原句", "confidence": 0.8}}],
  "atmosphere": {{"overall_vibe": "氛围", "activity_level": "low/moderate/high/chaotic", "common_topics": ["话题"]}},
  "member_impressions": {{"用户ID": {{"traits": ["特点"], "relationship_to_bot": "对bot的态度"}}}},
  "character_relationships": {{"userA->userB": "关系描述"}},
  "notable_events": ["事件描述"]
}}"""


ENRICH_PROMPT = """你是一个知识库助手。请用你的训练数据中的百科知识，对以下群聊黑话/术语进行补充说明。
每条术语保持一行，格式：术语：补充说明（不要编造，如果确实不知道就写"无法确定"）

术语列表：
{terms}

请直接输出每条术语的补充说明："""


class KnowledgeExtractor:
    """从对话中提取群聊知识，并对黑话进行知识库补充。"""

    def __init__(self, context: Context, config: PluginConfig, store: MemoryStore):
        self.context = context
        self.cfg = config
        self.store = store

    async def extract_and_store(self, group_id: str, messages: list[MessageRecord]) -> None:
        if not self.cfg.memory.enable_memory:
            return

        try:
            conversation = "\n".join(
                f"[{m.sender_name}({m.sender_id})]: {m.text}"
                for m in messages[-200:]
            )

            existing = await self.store.load(group_id)
            existing_summary = json.dumps({
                "slang": existing.get("slang", [])[-10:],
                "atmosphere": existing.get("atmosphere", {}),
                "member_impressions": {
                    k: v.get("traits", []) for k, v in
                    existing.get("member_impressions", {}).items()
                },
            }, ensure_ascii=False, indent=2)

            provider_id = self.cfg.resolve_memory_provider()

            llm_resp = await safe_llm_generate(
                self.context,
                chat_provider_id=provider_id,
                prompt=EXTRACTION_PROMPT.format(
                    conversation=conversation,
                    existing_knowledge=existing_summary,
                ),
                system_prompt="你是一个群聊分析师。利用你的训练数据知识补充术语含义。只输出 JSON。",
            )

            text = llm_resp.completion_text.strip()
            data = self._parse_json(text)

            if not data:
                return

            # 黑话知识补充——用 LLM 训练数据中的百科知识丰富含义
            if data.get("slang"):
                data["slang"] = await self._enrich_slang(data["slang"], provider_id)

            # 更新每个知识类别
            if data.get("slang"):
                await self.store.update_slang(
                    group_id, data["slang"], self.cfg.memory.max_slang_entries
                )

            if data.get("atmosphere"):
                await self.store.update_atmosphere(group_id, data["atmosphere"])

            if data.get("member_impressions"):
                await self.store.update_member_impressions(group_id, data["member_impressions"])

            if data.get("character_relationships"):
                await self.store.update_relationships(group_id, data["character_relationships"])

            if data.get("notable_events"):
                await self.store.add_notable_events(group_id, data["notable_events"])

            knowledge = await self.store.load(group_id)
            knowledge["last_extraction_time"] = time.time()
            await self.store.save(group_id, knowledge)

            logger.info(
                f"[VividRobot] 群 {group_id} 知识已更新: "
                f"{len(data.get('slang', []))} 条黑话, "
                f"{len(data.get('member_impressions', {}))} 个成员印象"
            )

        except Exception as e:
            logger.error(f"[VividRobot] 群 {group_id} 知识提取失败: {e}", exc_info=True)

    async def _enrich_slang(self, slang_list: list, provider_id: str) -> list:
        """用 LLM 训练数据补充黑话含义。"""
        try:
            terms = []
            for s in slang_list:
                term = s.get("term", s) if isinstance(s, dict) else str(s)
                meaning = s.get("meaning", "") if isinstance(s, dict) else ""
                # 跳过含义已经足够详细的（>30字一般已经有百科补充）
                if len(meaning) > 30:
                    continue
                terms.append(term)

            if not terms:
                return slang_list

            terms_text = "\n".join(terms)
            llm_resp = await safe_llm_generate(
                self.context,
                chat_provider_id=provider_id,
                prompt=ENRICH_PROMPT.format(terms=terms_text),
                system_prompt="你是一个知识库助手。只输出补充说明，不要加前缀。",
            )

            enrich_text = llm_resp.completion_text.strip()
            enrich_map = {}
            for line in enrich_text.split("\n"):
                line = line.strip()
                if "：" in line or ":" in line:
                    sep = "：" if "：" in line else ":"
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        enrich_map[parts[0].strip()] = parts[1].strip()

            # 回填丰富后的含义
            for s in slang_list:
                term = s.get("term", s) if isinstance(s, dict) else str(s)
                if term in enrich_map and enrich_map[term] != "无法确定":
                    if isinstance(s, dict):
                        s["meaning"] = enrich_map[term]
                        s["confidence"] = min(s.get("confidence", 0.7) + 0.1, 1.0)

        except Exception as e:
            logger.warning(f"[VividRobot] 黑话知识补充失败: {e}")

        return slang_list

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}
