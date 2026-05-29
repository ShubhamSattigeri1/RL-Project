import pytest
from fastapi.testclient import TestClient
from artwork_bandit.api.main import app
from artwork_bandit.db import database
import time

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'

def test_recommend_and_feedback_cycle():
    # ensure at least one user and content exist
    state = app.state
    users = state.users
    contents = state.contents
    if not users or not contents:
        pytest.skip('No data')
    user = users[0]['user_id']
    content = contents[0]['content_id']
    start = time.time()
    r = client.post('/recommend', json={'user_id': user, 'content_id': content})
    elapsed = (time.time() - start) * 1000
    assert r.status_code == 200
    data = r.json()
    assert 'artwork_id' in data
    imp_id = data['impression_id']
    # feedback
    r2 = client.post('/feedback', json={'impression_id': imp_id, 'reward': 1.0})
    assert r2.status_code == 200
    # stats
    r3 = client.get('/stats')
    assert r3.status_code == 200
    stats = r3.json()
    assert 'overall_ctr' in stats
    # latency
    assert elapsed < 200

def test_recommend_invalid_user():
    r = client.post('/recommend', json={'user_id': 'no_such_user', 'content_id': 'content_001'})
    assert r.status_code == 404

def test_feedback_invalid_reward():
    # create an impression
    state = app.state
    users = state.users
    contents = state.contents
    if not users or not contents:
        pytest.skip('No data')
    user = users[0]['user_id']
    content = contents[0]['content_id']
    r = client.post('/recommend', json={'user_id': user, 'content_id': content})
    imp = r.json()['impression_id']
    r2 = client.post('/feedback', json={'impression_id': imp, 'reward': 2.0})
    assert r2.status_code == 422
