import numpy as np
import threading
from typing import Dict

class LinUCBArm:
    def __init__(self, d: int, alpha: float = 1.0):
        self.d = d
        self.A = np.eye(d)
        self.A_inv = np.eye(d)
        self.b = np.zeros(d)
        self.alpha = float(alpha)

    def score(self, context: np.ndarray) -> float:
        # use cached inverse for speed
        theta = self.A_inv @ self.b
        est = float(theta @ context)
        conf = self.alpha * np.sqrt(float(context @ (self.A_inv @ context)))
        return est + conf

    def update(self, context: np.ndarray, reward: float):
        # A <- A + x x^T
        x = context
        self.A += np.outer(x, x)
        # Sherman-Morrison update for inverse: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        Au = self.A_inv @ x
        denom = 1.0 + float(x @ Au)
        if denom != 0.0:
            self.A_inv = self.A_inv - np.outer(Au, Au) / denom
        self.b += reward * x

    def to_dict(self) -> dict:
        return {'A': self.A.tolist(), 'b': self.b.tolist(), 'alpha': self.alpha}

    @classmethod
    def from_dict(cls, arm_id: str, d: int, data: dict, alpha: float = 1.0):
        arm = cls(d, alpha=alpha)
        arm.A = np.array(data['A'])
        try:
            arm.A_inv = np.linalg.inv(arm.A)
        except Exception:
            arm.A_inv = np.eye(d)
        arm.b = np.array(data['b'])
        return arm

class LinUCBBandit:
    def __init__(self, arm_ids: list, d: int, alpha: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.arms: Dict[str, LinUCBArm] = {aid: LinUCBArm(d, alpha=alpha) for aid in arm_ids}
        self.lock = threading.Lock()

    def select(self, context_vectors: Dict[str, np.ndarray]) -> str:
        best = None
        best_score = -np.inf
        for aid, vec in context_vectors.items():
            if aid not in self.arms:
                continue
            sc = self.arms[aid].score(vec)
            if sc > best_score:
                best_score = sc
                best = aid
        return best

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        with self.lock:
            arm = self.arms.get(arm_id)
            if arm is None:
                return
            arm.update(context, reward)

    def save(self, db):
        for aid, arm in self.arms.items():
            db.save_bandit_arm(aid, arm.A, arm.b)

    def load(self, db):
        for aid in list(self.arms.keys()):
            data = db.load_bandit_arm(aid)
            if data is not None:
                self.arms[aid].A = data['A']
                self.arms[aid].b = data['b']
