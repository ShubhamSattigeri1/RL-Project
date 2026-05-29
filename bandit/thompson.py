import numpy as np
from typing import Dict

class ThompsonArm:
    def __init__(self, arm_id: str):
        self.arm_id = arm_id
        self.alpha = 1.0
        self.beta = 1.0

    def sample(self) -> float:
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, reward: float):
        if reward > 0.5:
            self.alpha += 1
        else:
            self.beta += 1

class ThompsonBandit:
    """Thompson Sampling bandit (non-contextual).

    Note: Thompson Sampling ignores context vectors — it is a pure bandit,
    not contextual. Use when context is not required.
    """
    def __init__(self, arm_ids: list):
        self.arms: Dict[str, ThompsonArm] = {aid: ThompsonArm(aid) for aid in arm_ids}

    def select(self, arm_ids: list) -> str:
        best = None
        best_s = -1
        for aid in arm_ids:
            s = self.arms[aid].sample()
            if s > best_s:
                best_s = s
                best = aid
        return best

    def update(self, arm_id: str, reward: float):
        arm = self.arms.get(arm_id)
        if arm is None:
            return
        arm.update(reward)
