"""
内存中的群聊状态跟踪。

使用类级别 dict（类似 outputpro 的 StateManager）实现快速访问，
无需数据库开销。状态在重启后丢失（可接受——知识通过 KV store 持久化）。
"""
import time
from collections import deque
from typing import Optional

from .models import MessageRecord


class GroupState:
    """单个群的对话状态。"""

    MAX_BUFFER_SIZE = 100

    __slots__ = (
        "group_id", "message_buffer", "bot_reply_count", "last_bot_reply_time",
        "last_user_interaction_time", "interacting_users", "attention_level",
        "message_count_since_update",
    )

    def __init__(self, group_id: str):
        self.group_id = group_id
        self.message_buffer: deque[MessageRecord] = deque(maxlen=self.MAX_BUFFER_SIZE)
        self.bot_reply_count: int = 0
        self.last_bot_reply_time: float = 0.0
        self.last_user_interaction_time: float = 0.0
        self.interacting_users: set[str] = set()
        self.attention_level: float = 1.0
        self.message_count_since_update: int = 0

    # ---- 消息记录 ----

    def record_message(self, event, message_text: str) -> None:
        sender_id = event.get_sender_id()
        self.message_buffer.append(MessageRecord(
            sender_id=sender_id,
            sender_name=event.get_sender_name(),
            text=message_text,
            timestamp=time.time(),
            is_bot=False,
        ))
        self.message_count_since_update += 1

    def record_bot_reply(self, reply_text: str = "", sender_id: str = "") -> None:
        self.bot_reply_count += 1
        self.last_bot_reply_time = time.time()
        if reply_text:
            self.message_buffer.append(MessageRecord(
                sender_id=sender_id or "bot",
                sender_name="bot",
                text=reply_text,
                timestamp=time.time(),
                is_bot=True,
            ))

    # ---- 消息查询 ----

    def get_recent_messages(self, count: int) -> list[MessageRecord]:
        return list(self.message_buffer)[-count:]

    def get_messages_for_memory_update(self) -> list[MessageRecord]:
        return list(self.message_buffer)

    def reset_message_counter(self) -> None:
        self.message_count_since_update = 0

    # ---- 注意力 ----

    def boost_attention(self) -> None:
        """有人直接和 bot 对话时提升注意力。"""
        self.attention_level = min(self.attention_level * 1.5, 5.0)
        self.last_user_interaction_time = time.time()

    def decay_attention(self) -> None:
        """注意力随时间衰减。"""
        elapsed = time.time() - self.last_user_interaction_time
        if elapsed > 120:
            self.attention_level = max(self.attention_level * 0.95, 1.0)

    # ---- 对话续接 ----

    def is_bot_being_addressed(self, message_text: str, personality_name: str) -> bool:
        """检查消息是否在对 bot 说话（喊了人格名）。"""
        if not personality_name:
            return False
        return personality_name in message_text

    def was_bot_last_speaker(self, lookback: int = 3) -> bool:
        """bot 是否是最后一个（有效）发言的人。
        跳过纯图片/系统消息，往前看最多 lookback 条。
        """
        if not self.message_buffer:
            return False
        for msg in reversed(list(self.message_buffer)[-lookback:]):
            # 跳过纯图片、系统占位消息
            if msg.text and msg.text.strip() in ("[图片]", "[forward消息]", "[link消息]"):
                continue
            return msg.is_bot
        return False

    def last_bot_message_text(self) -> Optional[str]:
        """获取 bot 最近一次发言的内容。"""
        for msg in reversed(self.message_buffer):
            if msg.is_bot:
                return msg.text
        return None


class StateManager:
    """全局群状态注册表（类级别）。"""

    _groups: dict[str, GroupState] = {}

    @classmethod
    def get_group(cls, group_id: str) -> GroupState:
        if group_id not in cls._groups:
            cls._groups[group_id] = GroupState(group_id)
        return cls._groups[group_id]

    @classmethod
    def cleanup_idle(cls, max_idle_seconds: int = 3600) -> None:
        """移除超过 max_idle_seconds 没有 bot 活动的群状态。"""
        now = time.time()
        to_remove = [
            gid for gid, state in cls._groups.items()
            if state.last_bot_reply_time and (now - state.last_bot_reply_time) > max_idle_seconds
        ]
        for gid in to_remove:
            del cls._groups[gid]
