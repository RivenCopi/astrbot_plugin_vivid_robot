"""文本处理工具。"""
import re


_CQ_CODE_RE = re.compile(r'\[CQ:[^\]]+\]')
_WHITESPACE_RE = re.compile(r'\s+')


def clean_message_text(text: str) -> str:
    """移除 CQ 码和多余空白。"""
    text = _CQ_CODE_RE.sub('', text)
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()


def is_emotional_message(text: str) -> bool:
    """快速启发式检测情感内容。"""
    emotional_indicators = [
        "好累", "好烦", "不想", "emo", "难受", "崩溃",
        "受不了", "笑死", "哈哈哈", "草", "绝了", "太强了",
        "感动", "破防", "无语", "麻了",
    ]
    return any(ind in text for ind in emotional_indicators)
