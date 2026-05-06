"""
消息类型检测与过滤。

检测合并转发、链接、图片等消息类型，
判断 bot 是否被明确要求查看特定内容。
"""
import re
from typing import Optional

from astrbot.api.event import AstrMessageEvent

# 匹配 URL
_URL_RE = re.compile(r'https?://[^\s]+')

# 被明确叫到的关键词
_DIRECTIVE_KEYWORDS = [
    "看看", "查看", "看下", "看一下", "看了吗",
    "看看这个", "看这个", "看下这个",
    "讲了什么", "说了什么", "说什么", "讲的啥",
    "里面有什么", "写了什么",
    "帮我看看", "帮我看下", "帮我查查",
]


def has_forward_content(event: AstrMessageEvent) -> bool:
    """检测消息是否包含合并转发节点。"""
    try:
        messages = event.get_messages()
        for comp in messages:
            type_name = type(comp).__name__
            if type_name in ("Node", "Nodes"):
                return True
    except Exception:
        pass
    return False


def extract_urls(text: str) -> list[str]:
    """从文本中提取 URL。"""
    return _URL_RE.findall(text)


def has_urls(text: str) -> bool:
    """检测文本是否包含 URL。"""
    return bool(_URL_RE.search(text))


def extract_images(event: AstrMessageEvent) -> list[str]:
    """从消息中提取图片 URL/file 列表，跳过 GIF/表情包。
    返回可用于 LLM image_urls 参数的 URL/file 列表。
    """
    images = []
    try:
        from astrbot.api.message_components import Image
        messages = event.get_messages()
        for comp in messages:
            if isinstance(comp, Image):
                url = getattr(comp, "url", None)
                file = getattr(comp, "file", None)
                # 跳过 GIF 表情包（vision 模型通常不支持）
                if url and url.lower().endswith(".gif"):
                    continue
                if file and file.lower().endswith(".gif"):
                    continue
                if url:
                    images.append(url)
                elif file:
                    images.append(file)
    except Exception:
        pass
    return images


def has_images(event: AstrMessageEvent) -> bool:
    """检测消息是否包含图片。"""
    return len(extract_images(event)) > 0


def is_explicitly_asked(
    message_text: str,
    cfg,
) -> bool:
    """检测 bot 是否被明确要求查看/处理某内容。

    条件：消息中同时包含人格名(或别名) + 指示词。
    """
    if not cfg.is_personality_mentioned(message_text):
        return False
    return any(kw in message_text for kw in _DIRECTIVE_KEYWORDS)


def should_skip_message_content(
    event: AstrMessageEvent,
    message_text: str,
    cfg,
    skip_forward: bool = True,
    skip_link: bool = True,
) -> Optional[str]:
    """判断是否应跳过消息内容处理。

    返回 None 表示正常处理；
    返回 skip_reason 字符串表示跳过原因。

    即使包含转发/链接，如果 bot 被明确叫到，也不跳过。
    """
    if is_explicitly_asked(message_text, cfg):
        return None  # 明确被叫到，不跳过

    if skip_forward and has_forward_content(event):
        return "forward"

    if skip_link and has_urls(message_text):
        return "link"

    return None
