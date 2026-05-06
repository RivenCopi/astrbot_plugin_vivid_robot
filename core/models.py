"""Pydantic 数据模型。"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReplyReason(str, Enum):
    """回复决策的原因。"""
    MUST_REPLY = "must_reply"
    INTERJECT = "interject"
    CONVERSATION_CONTINUE = "conv_continue"
    RANDOM = "random"
    ATTENTION_BOOST = "attention_boost"


@dataclass
class DecisionResult:
    """判断结果。"""
    should_reply: bool
    reason: Optional[ReplyReason] = None
    confidence: float = 0.0
    llm_raw_response: str = ""
    token_usage: int = 0


class MessageRecord(BaseModel):
    """消息缓冲区中的单条记录。"""
    sender_id: str
    sender_name: str
    text: str
    timestamp: float
    is_bot: bool = False


class SlangEntry(BaseModel):
    """群内黑话/惯用语。"""
    term: str
    meaning: str
    example: str = ""
    confidence: float = 1.0
    created_at: float = 0.0


class GroupAtmosphere(BaseModel):
    """群内氛围。"""
    overall_vibe: str = "neutral"
    activity_level: str = "moderate"
    common_topics: list[str] = Field(default_factory=list)
    last_updated: float = 0.0


class MemberImpression(BaseModel):
    """成员印象。"""
    user_id: str
    nickname: str
    traits: list[str] = Field(default_factory=list)
    relationship_to_bot: str = ""
    interaction_count: int = 0
    first_seen: float = 0.0
    last_updated: float = 0.0


class GroupKnowledge(BaseModel):
    """群聊知识总汇。"""
    group_id: str
    atmosphere: GroupAtmosphere = Field(default_factory=GroupAtmosphere)
    slang: list[SlangEntry] = Field(default_factory=list)
    member_impressions: dict[str, MemberImpression] = Field(default_factory=dict)
    character_relationships: dict[str, str] = Field(default_factory=dict)
    notable_events: list[str] = Field(default_factory=list)
    last_extraction_time: float = 0.0
    last_updated: float = 0.0
