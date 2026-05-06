"""
防刷屏 / 冷却控制。

每个群独立跟踪冷却状态和连续回复计数。
"""
import time
from collections import defaultdict

from .config import PluginConfig


class Debouncer:
    """每群冷却时间和连续回复计数。"""

    def __init__(self, config: PluginConfig):
        self.cfg = config
        self._last_reply_time: dict[str, float] = defaultdict(float)
        self._consecutive_count: dict[str, int] = defaultdict(int)

    def may_proceed(self, group_id: str) -> bool:
        """检查 bot 是否被允许考虑回复。"""
        cfg = self.cfg.decision

        # 冷却检查
        elapsed = time.time() - self._last_reply_time.get(group_id, 0)
        if elapsed < cfg.cooldown_seconds:
            return False

        # 最大连续回复检查
        if self._consecutive_count.get(group_id, 0) >= cfg.max_consecutive_replies:
            return False

        return True

    def record_reply(self, group_id: str) -> None:
        """bot 成功发送回复后调用。"""
        self._last_reply_time[group_id] = time.time()
        self._consecutive_count[group_id] = self._consecutive_count.get(group_id, 0) + 1

    def on_other_speaks(self, group_id: str) -> None:
        """当其他人（非 bot）在 bot 之后发言时调用，重置连续计数。"""
        self._consecutive_count[group_id] = 0

    def get_cooldown_remaining(self, group_id: str) -> float:
        """返回剩余冷却秒数。"""
        elapsed = time.time() - self._last_reply_time.get(group_id, 0)
        return max(0.0, self.cfg.decision.cooldown_seconds - elapsed)
