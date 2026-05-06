"""
使用 LLM 生成角色一致的回复。

回复会自然融入群聊语境（黑话、氛围、成员印象）。
"""
import re as re_mod
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.star.context import Context

from .config import PluginConfig
from .models import MessageRecord
from ..utils.llm_utils import safe_llm_generate


REPLY_PROMPT_TEMPLATE = """你是群聊中的一员。你的角色设定是：

{persona_prompt}

关于这个群的信息：
- 群内氛围：{group_atmosphere}
- 群内黑话：{group_slang}
- 成员印象：{member_impressions}

最近的对话：
{recent_messages}

刚刚有人说：「{trigger_message}」
你决定插话，原因是：{decision_reason}

请以你的角色身份说一句自然的回复。要求：
1. 不要说"作为XX角色"、"根据我的设定"等出戏的话
2. 使用自然的日常口语
3. 简短自然，1-3句话，不要长篇大论
4. 适当使用群内黑话（如果有的话）
5. 不要重复别人的话
6. 不要加前缀如"回复："、"bot："等
7. 如果有人在抱怨/倾诉痛苦，给予恰当共情，但不要过度说教
8. 如果群友在质疑你的性格/说话方式（如"你怎么说话怪怪的"、"你不是XX吗"），在回复末尾自然地加一句提示管理员如何修改人格的话，如"（管理员可以用 /persona 修改我的人格设定哦）"

直接输出回复内容，不要加任何前缀或说明："""


class ReplyEngine:
    """生成拟人化回复。"""

    def __init__(self, context: Context, config: PluginConfig, persona_provider):
        self.context = context
        self.cfg = config
        self.persona_provider = persona_provider

    async def generate(
        self,
        event: AstrMessageEvent,
        trigger_message: str,
        recent_messages: list[MessageRecord],
        group_knowledge: dict,
        persona_prompt: Optional[str],
        decision_reason: str = "",
    ) -> Optional[str]:
        try:
            provider_id = self.cfg.resolve_reply_provider(event.unified_msg_origin)

            recent_text = "\n".join(
                f"[{m.sender_name}]: {m.text}"
                for m in recent_messages[-15:]
            )

            # 格式化氛围
            atmosphere = group_knowledge.get("atmosphere", {})
            if isinstance(atmosphere, dict):
                atmosphere_str = atmosphere.get("overall_vibe", "未知")
            else:
                atmosphere_str = "未知"

            # 格式化黑话
            slang_list = group_knowledge.get("slang", [])
            if slang_list:
                slang_str = "; ".join(
                    f"{s.get('term', s) if isinstance(s, dict) else s}"
                    f"={' '.join(s.get('meaning', [])) if isinstance(s, dict) and isinstance(s.get('meaning'), list) else s.get('meaning', '') if isinstance(s, dict) else ''}"
                    for s in slang_list[:5]
                )
            else:
                slang_str = "暂无记录"

            # 格式化成员印象
            impressions = group_knowledge.get("member_impressions", {})
            if impressions:
                imp_parts = []
                for uid, imp in list(impressions.items())[:5]:
                    if isinstance(imp, dict):
                        traits = imp.get("traits", [])
                        traits_str = ", ".join(traits) if traits else "未知"
                        imp_parts.append(f"{uid}: {traits_str}")
                imp_str = "; ".join(imp_parts) if imp_parts else "暂无记录"
            else:
                imp_str = "暂无记录"

            reason_text = {
                "must_reply": "有人喊了你的名字，必须回应",
                "interject": "群友在表达情绪，适当插话关心",
                "conv_continue": "有人在接你的话，继续对话",
                "random": "觉得可以自然插句话",
                "attention_boost": "有人一直在和你聊天，保持参与",
            }.get(decision_reason, "自然参与群聊")

            prompt = REPLY_PROMPT_TEMPLATE.format(
                persona_prompt=persona_prompt or "你是一个友好的群聊成员",
                group_atmosphere=atmosphere_str,
                group_slang=slang_str,
                member_impressions=imp_str,
                recent_messages=recent_text or "（暂无对话）",
                trigger_message=trigger_message,
                decision_reason=reason_text,
            )

            llm_resp = await safe_llm_generate(
                self.context,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=persona_prompt or "",
            )

            reply = llm_resp.completion_text.strip()
            reply = self._clean_reply(reply)

            logger.info(f"[VividRobot] 生成回复 ({len(reply)} 字符)")
            return reply

        except Exception as e:
            logger.error(f"[VividRobot] 回复生成失败: {e}", exc_info=True)
            return None

    @staticmethod
    def _clean_reply(text: str) -> str:
        """清理 LLM 回复中的常见 artifacts。"""
        # 去掉整体引用包裹
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]

        # 去掉常见前缀
        for prefix in ("回复：", "bot：", "Bot：", "回复:", "回复 ", "bot:", "Bot:"):
            if text.startswith(prefix):
                text = text[len(prefix):]

        return text.strip()
