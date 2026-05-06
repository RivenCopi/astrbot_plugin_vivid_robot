"""
基于社会学/心理学的交互频率调节。

参考理论：
- 邓巴数 (Dunbar's Number): 认知上限约 150 段稳定关系
  - 亲密圈 (5): 最深度交互，频率最高
  - 朋友圈 (15): 中等深度
  - 熟人圈 (50): 较轻交互
  - 超出: 最小化
- 社会渗透理论 (Social Penetration Theory): 关系通过
  自我披露的广度和深度逐步发展
- 不确定性减少理论 (Uncertainty Reduction Theory):
  初始交互中人们通过信息寻求来减少不确定性
"""
import math
import time
import random

from .config import PluginConfig
from .state_manager import GroupState


class PsychologyModule:
    """无状态心理学规则，调节交互行为。"""

    def __init__(self, config: PluginConfig):
        self.cfg = config

    def get_reply_modifier(self, state: GroupState) -> float:
        """计算基于心理因素的概率修正因子。
        返回倍率: 1.0 = 基线, >1 = 更可能回复, <1 = 不太可能回复。
        """
        if not self.cfg.psychology.enable_psychology:
            return 1.0

        modifier = 1.0

        # 1. 新鲜感衰减: bot 刚进群时更活跃，逐渐趋于自然节奏
        modifier *= self._novelty_decay(state)

        # 2. 社交渗透深度
        depth = self.cfg.psychology.social_penetration_depth
        if depth > 1.5:
            modifier *= 0.8   # 深度交流：少说但更有意义
        elif depth > 1.0:
            modifier *= 1.0
        else:
            modifier *= 1.2   # 浅层：更随意，频率更高

        # 3. 活跃用户数调节 (Dunbar)
        active_users = len(state.interacting_users)
        if active_users > self.cfg.psychology.dunbar_friend_size:
            modifier *= 0.7
        elif active_users > self.cfg.psychology.dunbar_intimate_size:
            modifier *= 0.85

        # 4. 时间节律
        hour = time.localtime().tm_hour
        if 0 <= hour < 6:
            modifier *= 0.5   # 深夜
        elif 6 <= hour < 9:
            modifier *= 0.7   # 清晨
        elif 12 <= hour < 14:
            modifier *= 1.1   # 午休
        elif 19 <= hour < 23:
            modifier *= 1.15  # 晚间黄金时间

        return max(modifier, 0.1)

    def _novelty_decay(self, state: GroupState) -> float:
        """基于总互动次数的新鲜感衰减。
        使用指数衰减：互动越多 → 越"淡定" → 概率降低。
        """
        total = state.bot_reply_count
        if total < 10:
            return 1.5  # 刚加入，很兴奋
        decay = self.cfg.psychology.interaction_novelty_decay
        return max(decay ** (total / 10), 0.5)

    def compute_interaction_depth(self, state: GroupState) -> str:
        """确定当前上下文的合适交互深度。
        返回: "superficial", "casual", "personal", "intimate"
        """
        total = state.bot_reply_count
        if total < 5:
            return "superficial"
        elif total < 30:
            return "casual"
        elif total < 100:
            return "personal"
        else:
            return "intimate"

    @staticmethod
    def should_initiate_new_topic(state: GroupState) -> bool:
        """决定 bot 是否应主动发起新话题。
        基于不确定性减少理论：定期主动发起以维持社交存在感。
        """
        if state.bot_reply_count > 0 and state.last_bot_reply_time:
            silence_duration = time.time() - state.last_bot_reply_time
            if silence_duration > 1800:  # 30分钟没人说话
                return random.random() < 0.3
        return False
