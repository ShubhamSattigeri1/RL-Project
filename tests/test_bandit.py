import numpy as np
from artwork_bandit.bandit.linucb import LinUCBArm, LinUCBBandit
from artwork_bandit.bandit.thompson import ThompsonBandit

def test_linucb_score_and_update():
    d=10
    arm = LinUCBArm(d=d, alpha=0.5)
    ctx = np.ones(d)
    s = arm.score(ctx)
    assert isinstance(s, float)
    A0 = arm.A.copy()
    b0 = arm.b.copy()
    arm.update(ctx, 1.0)
    assert not np.allclose(arm.A, A0)
    assert not np.allclose(arm.b, b0)

def test_linucb_bandit_select():
    arms = ['a','b','c']
    d=5
    bandit = LinUCBBandit(arms, d=d, alpha=1.0)
    contexts = {a: np.ones(d) * i for i,a in enumerate(arms, start=1)}
    chosen = bandit.select(contexts)
    assert chosen in arms

def test_linucb_exploit_after_updates():
    arms = ['a','b','c']
    d=5
    bandit = LinUCBBandit(arms, d=d, alpha=0.1)
    ctx = np.ones(d)
    # reward arm 'b' heavily
    for _ in range(100):
        bandit.update('b', ctx, 1.0)
    counts = {a:0 for a in arms}
    for _ in range(100):
        chosen = bandit.select({a:ctx for a in arms})
        counts[chosen]+=1
    assert counts['b'] > 80

def test_thompson_select():
    arms = [f'a{i}' for i in range(4)]
    t = ThompsonBandit(arms)
    chosen = t.select(arms)
    assert chosen in arms

def test_linucb_serialization_roundtrip():
    d=3
    arm = LinUCBArm(d=d, alpha=1.0)
    arm.update(np.ones(d), 1.0)
    data = arm.to_dict()
    arm2 = LinUCBArm.from_dict('x', d, data, alpha=1.0)
    assert np.allclose(arm.A, arm2.A)
    assert np.allclose(arm.b, arm2.b)
