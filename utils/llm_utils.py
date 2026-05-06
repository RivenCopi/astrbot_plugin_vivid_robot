"""
LLM 调用工具：带 provider fallback 的 safe_llm_generate。
"""
from astrbot.api import logger
from astrbot.core.exceptions import ProviderNotFoundError


async def safe_llm_generate(context, chat_provider_id: str, **kwargs):
    """调用 llm_generate，若 provider 不存在则 fallback 到第一个可用 provider。"""
    try:
        return await context.llm_generate(
            chat_provider_id=chat_provider_id, **kwargs
        )
    except ProviderNotFoundError:
        provider_insts = context.provider_manager.provider_insts
        if provider_insts:
            fallback_id = provider_insts[0].provider_id
            logger.warning(
                f"[VividRobot] Provider '{chat_provider_id}' 未找到，"
                f"fallback 到 '{fallback_id}'"
            )
            return await context.llm_generate(
                chat_provider_id=fallback_id, **kwargs
            )
        raise
