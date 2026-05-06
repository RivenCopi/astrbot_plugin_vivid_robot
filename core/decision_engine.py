"""
3 层决策管线，判断 bot 是否应该回复/插话。

Tier 1: 确定性规则（被喊名字、有人接话）—— 不调用 LLM
Tier 2: 概率预过滤（base × attention × psychology × keyword_bonus）—— 不调用 LLM
Tier 3: 小模型 LLM 判断 —— 前两层未决定时才调用
"""
import json
import random
import re
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.star.context import Context

from .config import PluginConfig
from .models import DecisionResult, ReplyReason, MessageRecord
from .state_manager import GroupState
from ..utils.llm_utils import safe_llm_generate


DECISION_PROMPT_TEMPLATE = """你是一个群聊插话判断助手。请根据上下文判断是否应该插话/回复。

你的角色设定：{persona_prompt}

群内氛围：{group_atmosphere}

最近的对话：
{recent_messages}

最新消息：「{current_message}」

请判断是否应该回复。以下情况应考虑回复：
1. 群友在抱怨/吐槽/表达负面情绪 → 适当插话安慰
2. 群友在讨论你熟悉的话题 → 可以插话
3. 群友提到了你/在对你说话 → 必须回复
4. 群友的话是接你之前的话 → 应该回复
5. 纯表情/刷屏/打卡/无关闲聊 → 不回复

请只输出 JSON：
{{"should_reply": true/false, "reason": "简短说明", "confidence": 0.0-1.0}}"""


class DecisionEngine:
    """判断是否回复的 3 层决策引擎。"""

    def __init__(self, context: Context, config: PluginConfig, persona_provider):
        self.context = context
        self.cfg = config
        self.persona_provider = persona_provider

    async def evaluate(
        self,
        event: AstrMessageEvent,
        current_message: str,
        recent_messages: list[MessageRecord],
        group_knowledge: dict,
        persona_prompt: Optional[str],
        psychology_modifier: float,
        state: GroupState,
    ) -> DecisionResult:
        cfg = self.cfg.decision

        # ---- Tier 1: 确定性规则 ----

        # 被喊名字(含别名) = 必回
        if self.cfg.is_personality_mentioned(current_message):
            if self.cfg.advanced.log_decision_reasoning:
                logger.info(
                    f"[VividRobot] MUST_REPLY: 人格名被提及 (群 {state.group_id})"
                )
            return DecisionResult(
                should_reply=True,
                reason=ReplyReason.MUST_REPLY,
                confidence=1.0,
            )

        # 有人在接 bot 的话 = 必回（往前看 5 条，跳过纯图片）
        if state.was_bot_last_speaker(lookback=5):
            bot_last = state.last_bot_message_text()
            if bot_last and self._is_reply_to_bot(current_message, bot_last):
                if self.cfg.advanced.log_decision_reasoning:
                    logger.info(
                        f"[VividRobot] CONV_CONTINUE: 有人接 bot 的话 (群 {state.group_id})"
                    )
                return DecisionResult(
                    should_reply=True,
                    reason=ReplyReason.CONVERSATION_CONTINUE,
                    confidence=0.85,
                )

        # ---- Tier 2: 概率预过滤 ----
        effective_prob = self._compute_effective_probability(
            cfg.base_reply_probability,
            state.attention_level,
            psychology_modifier,
            current_message,
        )

        if random.random() > effective_prob:
            return DecisionResult(
                should_reply=False,
                reason=ReplyReason.RANDOM,
            )

        # ---- Tier 3: LLM 判断 ----
        return await self._llm_judge(
            event, current_message, recent_messages,
            group_knowledge, persona_prompt, state,
        )

    # ---- Tier 2 helpers ----

    def _compute_effective_probability(
        self, base: float, attention: float,
        psych_mod: float, message: str,
    ) -> float:
        """计算有效回复概率。"""
        prob = base * attention * psych_mod

        # 情感关键词加成
        for kw in self.cfg.decision.trigger_keywords:
            if kw in message:
                prob *= 2.0
                break

        return min(prob, 1.0)

    def _is_reply_to_bot(self, new_msg: str, bot_last: str) -> bool:
        """Heuristic: 新消息看起来像在回复 bot？"""
        question_indicators = ("?", "？", "呢", "吗", "吧", "咋", "怎么", "啥", "什么")
        bot_asked = any(ind in bot_last for ind in question_indicators)
        # bot 问了问题 + 有人发了超过 3 个字 = 很可能是回复
        if bot_asked and len(new_msg) > 3:
            return True
        # 短消息跟在 bot 后面 = 很可能是回复
        if len(new_msg) < 20:
            return True
        # 消息中用"你"指代 bot（bot 刚说完话） = 很可能在跟 bot 说话
        if "你" in new_msg:
            return True
        return False

    # ---- Tier 3 helpers ----

    async def _llm_judge(
        self, event, current_message, recent_messages,
        group_knowledge, persona_prompt, state,
    ) -> DecisionResult:
        """用小模型判断是否回复。"""
        try:
            provider_id = self.cfg.resolve_decision_provider(event.unified_msg_origin)

            recent_text = "\n".join(
                f"[{m.sender_name}]: {m.text}" for m in recent_messages[-15:]
            )
            atmosphere = group_knowledge.get("atmosphere", {})
            atmosphere_str = atmosphere.get("overall_vibe", "未知") if isinstance(atmosphere, dict) else "未知"

            prompt = DECISION_PROMPT_TEMPLATE.format(
                persona_prompt=persona_prompt or "未设定",
                group_atmosphere=atmosphere_str,
                recent_messages=recent_text or "（无）",
                current_message=current_message,
            )

            llm_resp = await safe_llm_generate(
                self.context,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=self.cfg.decision.decision_system_prompt,
                temperature=self.cfg.decision.decision_llm_temperature,
            )

            text = llm_resp.completion_text.strip()
            data = self._parse_json_response(text)

            should_reply = data.get("should_reply", False)
            reason_str = data.get("reason", "")
            confidence = data.get("confidence", 0.5)

            # 归类
            reason = ReplyReason.RANDOM
            if any(w in reason_str for w in ("抱怨", "安慰", "emo", "吐槽", "累", "烦")):
                reason = ReplyReason.INTERJECT
            elif any(w in reason_str for w in ("对话", "回复", "接话", "继续")):
                reason = ReplyReason.CONVERSATION_CONTINUE

            if self.cfg.advanced.log_decision_reasoning:
                logger.info(
                    f"[VividRobot] LLM 判断: should_reply={should_reply} "
                    f"reason={reason.value} confidence={confidence:.2f} "
                    f"in group {state.group_id}: {reason_str}"
                )

            return DecisionResult(
                should_reply=should_reply,
                reason=reason,
                confidence=confidence,
                llm_raw_response=text,
                token_usage=getattr(llm_resp, "token_usage", 0),
            )

        except Exception as e:
            logger.error(f"[VividRobot] Decision LLM 调用失败: {e}", exc_info=True)
            return DecisionResult(should_reply=False)

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Robust JSON extraction from LLM output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}
