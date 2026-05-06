"""
astrbot_plugin_vivid_robot
让群聊机器人更加拟人——主动判断插话时机、人格化回复、
积累群内黑话与成员印象、产生成长感。
"""
import asyncio
from typing import Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api import logger, AstrBotConfig
from .utils.llm_utils import safe_llm_generate


@register(
    "astrbot_plugin_vivid_robot",
    "RivenCopi",
    "让群聊机器人更加拟人——主动判断插话时机、人格化回复、积累群内黑话与成员印象、产生成长感",
    "1.0.0",
    "https://github.com/RivenCopi/astrbot_plugin_vivid_robot",
)
class VividRobotPlugin(Star):
    """拟人化群聊机器人插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.cfg: Optional[object] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        self.memory_store = None
        self.knowledge_extractor = None
        self.knowledge_query = None
        self.persona_provider = None
        self.psychology = None
        self.debouncer = None
        self.decision_engine = None
        self.reply_engine = None
        self.data_dir = ""

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return

            from .core.config import PluginConfig

            self.cfg = PluginConfig(self.config, self.context)
            self.data_dir = StarTools.get_data_dir("astrbot_plugin_vivid_robot")

            from .core.memory.store import MemoryStore
            from .core.memory.extractor import KnowledgeExtractor
            from .core.memory.query import KnowledgeQuery
            from .core.persona import PersonaProvider
            from .core.psychology import PsychologyModule
            from .core.debouncer import Debouncer
            from .core.state_manager import StateManager

            self.memory_store = MemoryStore(self)
            self.knowledge_extractor = KnowledgeExtractor(self.context, self.cfg, self.memory_store)
            self.knowledge_query = KnowledgeQuery(self.memory_store)
            self.persona_provider = PersonaProvider(self.context, self.cfg)
            self.psychology = PsychologyModule(self.cfg)
            self.debouncer = Debouncer(self.cfg)
            self.state_manager = StateManager()

            from .core.decision_engine import DecisionEngine
            from .core.reply_engine import ReplyEngine

            self.decision_engine = DecisionEngine(self.context, self.cfg, self.persona_provider)
            self.reply_engine = ReplyEngine(self.context, self.cfg, self.persona_provider)

            self._initialized = True
            logger.info("[VividRobot] 插件初始化完成")

    async def terminate(self) -> None:
        logger.info("[VividRobot] 插件正在卸载...")
        self._initialized = False

    # ===== Event Handler =====

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent) -> None:
        """处理所有群消息，判断是否插话/回复。

        注意：这是 event_message_type 处理器，是普通 async 函数（非 generator），
        AstrBot 会 await 此函数。回复通过 event.chain_result() 设置。
        """
        if not self._initialized:
            logger.info("[VividRobot] 插件尚未初始化，跳过消息")
            return
        if not self.cfg or not self.cfg.basic.enabled:
            return

        group_id = event.get_group_id()
        if not group_id:
            return

        # 1. Whitelist gate
        umo = event.unified_msg_origin or group_id
        if not self.cfg.is_group_allowed(umo):
            if self.cfg.advanced.debug_mode:
                logger.info(
                    f"[VividRobot] 群不在白名单: umo={umo}, "
                    f"group_list={self.cfg.basic.group_list}"
                )
            return

        # 2. Filter self messages
        sender_id = event.get_sender_id()
        if sender_id == self.cfg.basic.self_id:
            return

        message_text = event.get_message_str()

        # 2.5. Forward/link guard
        from .utils.message_filter import should_skip_message_content
        skip_reason = should_skip_message_content(
            event, message_text, self.cfg,
            skip_forward=self.cfg.decision.skip_forward_content,
            skip_link=self.cfg.decision.skip_link_content,
        )
        if skip_reason:
            if self.cfg.advanced.log_decision_reasoning:
                logger.info(f"[VividRobot] 跳过 {skip_reason} 类消息 (群 {group_id})")
            state = self.state_manager.get_group(group_id)
            state.record_message(event, f"[{skip_reason}消息]")
            return

        # 2.6. OCR
        if self.cfg.is_ocr_group_allowed(umo):
            ocr_text = await self._run_ocr_if_images(event)
            if ocr_text:
                message_text = f"{message_text}\n[图片内容：{ocr_text}]"

        # 3. Update attention (before recording message, so was_bot_last_speaker
        #    can see the buffer state before the current message is added)
        state = self.state_manager.get_group(group_id)

        if self.cfg.is_personality_mentioned(message_text):
            state.boost_attention()
        else:
            state.decay_attention()

        # 4. Cooldown gate
        if not self.debouncer.may_proceed(group_id):
            state.record_message(event, message_text)
            self.debouncer.on_other_speaks(group_id)
            if self.cfg.advanced.debug_mode:
                logger.info(f"[VividRobot] 冷却中，跳过回复 (群 {group_id})")
            return

        # 5. Decision (runs BEFORE recording current message, so Tier 1
        #    was_bot_last_speaker check isn't polluted by the current message)
        decision = await self._evaluate_reply(event, message_text, state)
        if not decision.should_reply:
            state.record_message(event, message_text)
            self.debouncer.on_other_speaks(group_id)
            if self.cfg.advanced.debug_mode:
                logger.info(
                    f"[VividRobot] 决策引擎判断不回复 (群 {group_id}, "
                    f"reason={decision.reason.value if decision.reason else '?'})"
                )
            return

        # Record message now that we've decided to reply
        state.record_message(event, message_text)
        self.debouncer.on_other_speaks(group_id)

        # 6. Generate reply
        reply_text = await self._generate_reply(event, message_text, state, decision)
        if not reply_text:
            logger.info(f"[VividRobot] 生成回复为空 (群 {group_id})")
            return

        # 7. Send reply
        await event.send(event.chain_result([Plain(reply_text)]))

        # 8. Post-reply
        self.debouncer.record_reply(group_id)
        state.record_bot_reply(reply_text, self.cfg.basic.self_id)

        if self.cfg.advanced.log_decision_reasoning:
            logger.info(
                f"[VividRobot] 已回复 (群 {group_id}, "
                f"reason={decision.reason.value if decision.reason else '?'}, "
                f"confidence={decision.confidence:.2f})"
            )

        # 9. Periodic memory update (fire-and-forget)
        if (
            self.cfg.memory.enable_memory
            and state.message_count_since_update >= self.cfg.memory.memory_update_interval
        ):
            asyncio.create_task(self._update_memory(group_id))
            state.reset_message_counter()

    async def _evaluate_reply(
        self, event: AstrMessageEvent, message_text: str, state
    ):
        recent_messages = state.get_recent_messages(self.cfg.decision.max_context_messages)
        group_knowledge = await self.knowledge_query.get_context_for_prompt(state.group_id)
        persona_prompt = await self.persona_provider.get_persona_prompt(event)
        psych_mod = self.psychology.get_reply_modifier(state)

        return await self.decision_engine.evaluate(
            event=event,
            current_message=message_text,
            recent_messages=recent_messages,
            group_knowledge=group_knowledge,
            persona_prompt=persona_prompt,
            psychology_modifier=psych_mod,
            state=state,
        )

    async def _generate_reply(
        self, event: AstrMessageEvent, trigger_msg: str, state, decision,
    ) -> Optional[str]:
        recent_messages = state.get_recent_messages(self.cfg.decision.max_context_messages)
        group_knowledge = await self.knowledge_query.get_context_for_prompt(state.group_id)
        persona_prompt = await self.persona_provider.get_persona_prompt(event)

        return await self.reply_engine.generate(
            event=event,
            trigger_message=trigger_msg,
            recent_messages=recent_messages,
            group_knowledge=group_knowledge,
            persona_prompt=persona_prompt,
            decision_reason=decision.reason.value if decision.reason else "",
        )

    async def _run_ocr_if_images(self, event: AstrMessageEvent) -> str:
        try:
            from astrbot.api.message_components import Image

            # 提取 Image 组件，过滤 GIF 表情包
            image_comps: list[Image] = []
            try:
                for comp in event.get_messages():
                    if isinstance(comp, Image):
                        url = getattr(comp, "url", None) or ""
                        file = getattr(comp, "file", None) or ""
                        if url.lower().endswith(".gif") or file.lower().endswith(".gif"):
                            continue
                        image_comps.append(comp)
            except Exception:
                pass

            if not image_comps:
                return ""

            # 下载图片并转为 base64（QQ 图片 URL 有访问鉴权，直接传 URL 会失败）
            base64_urls = []
            for img in image_comps:
                try:
                    bs64 = await img.convert_to_base64()
                    if bs64:
                        base64_urls.append(f"base64://{bs64}")
                except Exception as e:
                    logger.warning(f"[VividRobot] 图片转换 base64 失败: {e}")

            if not base64_urls:
                logger.info("[VividRobot] 所有图片转换失败，跳过 OCR")
                return ""

            provider_id = self.cfg.resolve_ocr_provider()
            logger.info(f"[VividRobot] OCR 识别 {len(base64_urls)} 张图片...")

            llm_resp = await safe_llm_generate(
                self.context,
                chat_provider_id=provider_id,
                prompt="请简要描述这张图片的内容。如果有文字，请提取出来。不要加前缀或说明，直接描述。",
                image_urls=base64_urls,
                system_prompt="你是一个图片识别助手。只输出图片内容描述。",
            )

            ocr_text = llm_resp.completion_text.strip()
            logger.info(f"[VividRobot] OCR 结果 ({len(ocr_text)} 字符)")
            return ocr_text

        except Exception as e:
            logger.error(f"[VividRobot] OCR 识别失败: {e}", exc_info=True)
            return ""

    async def _update_memory(self, group_id: str) -> None:
        state = self.state_manager.get_group(group_id)
        messages_for_extraction = state.get_messages_for_memory_update()
        if messages_for_extraction:
            await self.knowledge_extractor.extract_and_store(group_id, messages_for_extraction)
