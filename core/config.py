"""
配置节点, 把 AstrBotConfig dict 变成强类型对象。

规则：
- schema 来自子类类型注解
- 声明字段：读写，写回底层 dict
- 未声明字段和下划线字段：仅挂载属性，不写回
- 支持 ConfigNode 多层嵌套（lazy + cache）
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from types import MappingProxyType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.context import Context


class ConfigNode:
    """配置节点基类。把 dict 变成强类型对象。"""

    _SCHEMA_CACHE: dict[type, dict[str, type]] = {}
    _FIELDS_CACHE: dict[type, set[str]] = {}

    @classmethod
    def _schema(cls) -> dict[str, type]:
        return cls._SCHEMA_CACHE.setdefault(cls, get_type_hints(cls))

    @classmethod
    def _fields(cls) -> set[str]:
        return cls._FIELDS_CACHE.setdefault(
            cls,
            {k for k in cls._schema() if not k.startswith("_")},
        )

    @staticmethod
    def _is_optional(tp: type) -> bool:
        if get_origin(tp) in (Union, UnionType):
            return type(None) in get_args(tp)
        return False

    def __init__(self, data: MutableMapping[str, Any]):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_children", {})
        for key, tp in self._schema().items():
            if key.startswith("_"):
                continue
            if key in data:
                continue
            if hasattr(self.__class__, key):
                continue
            if self._is_optional(tp):
                continue
            logger.warning(f"[config:{self.__class__.__name__}] 缺少字段: {key}")

    def __getattr__(self, key: str) -> Any:
        if key in self._fields():
            value = self._data.get(key)
            tp = self._schema().get(key)

            if isinstance(tp, type) and issubclass(tp, ConfigNode):
                children: dict[str, ConfigNode] = self.__dict__["_children"]
                if key not in children:
                    if not isinstance(value, MutableMapping):
                        raise TypeError(
                            f"[config:{self.__class__.__name__}] "
                            f"字段 {key} 期望 dict，实际是 {type(value).__name__}"
                        )
                    children[key] = tp(value)
                return children[key]

            return value

        if key in self.__dict__:
            return self.__dict__[key]

        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._fields():
            self._data[key] = value
            return
        object.__setattr__(self, key, value)

    def raw_data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._data)

    def save_config(self) -> None:
        if not isinstance(self._data, AstrBotConfig):
            raise RuntimeError("[config] 非根节点无法调用 save_config()")
        self._data.save()


# ===== Plugin-specific config nodes =====


class BasicConfig(ConfigNode):
    enabled: bool
    group_list_mode: str
    group_list: list
    personality_name: str
    personality_aliases: list
    self_id: str


class DecisionConfig(ConfigNode):
    decision_provider_id: str
    decision_system_prompt: str
    reply_provider_id: str
    base_reply_probability: float
    attention_boost_multiplier: float
    max_consecutive_replies: int
    cooldown_seconds: int
    max_context_messages: int
    trigger_keywords: list
    decision_llm_temperature: float
    skip_forward_content: bool
    skip_link_content: bool


class OCRConfig(ConfigNode):
    enable_ocr: bool
    ocr_provider_id: str
    ocr_group_list_mode: str
    ocr_group_list: list

    def is_group_allowed(self, umo_or_id: str) -> bool:
        mode = (self.ocr_group_list_mode or "whitelist").lower()
        if mode == "none":
            return True
        if not self.ocr_group_list:
            return False
        target = str(umo_or_id).strip()
        in_list = any(
            str(item).strip() in target or target in str(item).strip()
            for item in self.ocr_group_list
        )
        return in_list if mode == "whitelist" else not in_list


class PersonaConfig(ConfigNode):
    use_persona: bool
    persona_id: str
    reply_system_prompt_template: str


class MemoryConfig(ConfigNode):
    enable_memory: bool
    memory_update_interval: int
    max_slang_entries: int
    max_member_impressions: int
    memory_llm_provider_id: str


class PsychologyConfig(ConfigNode):
    enable_psychology: bool
    dunbar_intimate_size: int
    dunbar_friend_size: int
    social_penetration_depth: float
    interaction_novelty_decay: float


class AdvancedConfig(ConfigNode):
    debug_mode: bool
    log_decision_reasoning: bool


class PluginConfig(ConfigNode):
    basic: BasicConfig
    decision: DecisionConfig
    ocr: OCRConfig
    persona: PersonaConfig
    memory: MemoryConfig
    psychology: PsychologyConfig
    advanced: AdvancedConfig

    def __init__(self, cfg: AstrBotConfig, context: Context):
        super().__init__(cfg)
        self.context = context

    def get_personality_names(self) -> list[str]:
        """获取所有人格名和别名。"""
        names = []
        if self.basic.personality_name and self.basic.personality_name.strip():
            names.append(self.basic.personality_name.strip())
        for alias in (self.basic.personality_aliases or []):
            alias = alias.strip()
            if alias and alias not in names:
                names.append(alias)
        return names

    def is_personality_mentioned(self, text: str) -> bool:
        """检查文本是否提到了 bot 的人格名或任意别名。"""
        for name in self.get_personality_names():
            if name in text:
                return True
        return False

    def is_group_allowed(self, umo_or_id: str) -> bool:
        """检查群是否在白名单中 / 不在黑名单中。

        UMO 格式如 "aiocqhttp:GroupMessage:466195457"，配置中的 group_list
        通常只填纯数字群号。用 substring 匹配兼容两种写法。
        """
        mode = (self.basic.group_list_mode or "whitelist").lower()
        if mode == "none":
            return True
        if not self.basic.group_list:
            return True
        target = str(umo_or_id).strip()
        in_list = any(
            str(item).strip() in target or target in str(item).strip()
            for item in self.basic.group_list
        )
        return in_list if mode == "whitelist" else not in_list

    def resolve_decision_provider(self, umo) -> str:
        """解析判断提供商：配置 > 空（使用默认）。"""
        return self.decision.decision_provider_id.strip() or ""

    def resolve_reply_provider(self, umo) -> str:
        """解析回复提供商：reply > decision > 空（使用默认）。"""
        return (
            self.decision.reply_provider_id.strip()
            or self.decision.decision_provider_id.strip()
            or ""
        )

    def resolve_memory_provider(self) -> str:
        """解析知识提取提供商：memory > decision > 空（使用默认）。"""
        return (
            self.memory.memory_llm_provider_id.strip()
            or self.decision.decision_provider_id.strip()
            or ""
        )

    def resolve_ocr_provider(self) -> str:
        """解析 OCR 提供商：ocr > decision > 空（使用默认）。"""
        return (
            self.ocr.ocr_provider_id.strip()
            or self.decision.decision_provider_id.strip()
            or ""
        )

    def is_ocr_group_allowed(self, umo_or_id: str) -> bool:
        """检查群是否在 OCR 白名单中。"""
        if not self.ocr.enable_ocr:
            return False
        return self.ocr.is_group_allowed(umo_or_id)
