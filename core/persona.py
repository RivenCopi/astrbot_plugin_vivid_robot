"""
人格回退解析。

优先级（从高到低）：
1. UMO 默认 persona（persona_manager.get_default_persona_v3 已包含会话/对话级回退）
2. 插件全局 persona_id —— 兜底默认值
"""
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.star.context import Context

from .config import PluginConfig


class PersonaProvider:
    """解析 bot 当前应使用的人格设定。"""

    def __init__(self, context: Context, config: PluginConfig):
        self.context = context
        self.cfg = config

    async def get_persona_prompt(self, event: AstrMessageEvent) -> Optional[str]:
        if not self.cfg.persona.use_persona:
            return None

        persona_mgr = getattr(self.context, "persona_manager", None)
        if persona_mgr is None:
            return None

        umo = event.unified_msg_origin

        # UMO 默认人格（persona_mgr 内部已处理会话/对话/UMO 级 fallback）
        if umo:
            prompt = await self._get_umo_default_persona(persona_mgr, umo)
            if prompt:
                return prompt

        # 插件全局指定（兜底）
        persona_id = self.cfg.persona.persona_id
        if persona_id:
            prompt = await self._get_persona_by_id(persona_mgr, persona_id)
            if prompt:
                return prompt

        return None

    async def _get_persona_by_id(self, persona_mgr, persona_id: str) -> Optional[str]:
        try:
            persona_obj = await persona_mgr.get_persona(persona_id)
            if persona_obj and hasattr(persona_obj, "system_prompt"):
                prompt = persona_obj.system_prompt
                if prompt and isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
        except Exception as e:
            logger.warning(f"[VividRobot] 获取人格 {persona_id} 失败: {e}")
        return None

    async def _get_umo_default_persona(self, persona_mgr, umo: str) -> Optional[str]:
        try:
            personality = await persona_mgr.get_default_persona_v3(umo)
            if isinstance(personality, dict):
                prompt = personality.get("prompt")
            else:
                prompt = getattr(personality, "prompt", None)
            if prompt and isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        except Exception as e:
            logger.warning(f"[VividRobot] UMO 默认人格查找失败: {e}")
        return None
