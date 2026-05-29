from fastapi.testclient import TestClient
from artwork_bandit.api.main import app

client = TestClient(app)
state = app.state
if not getattr(state, 'users', None):
    print('No users available')
    raise SystemExit(1)
user = state.users[0]['user_id']
content = state.contents[0]['content_id']
print('Using', user, content)
# Recommend
r = client.post('/recommend', json={'user_id': user, 'content_id': content})
print('RECOMMEND:', r.status_code, r.json())
impression_id = r.json().get('impression_id')
# Feedback
f = client.post('/feedback', json={'impression_id': impression_id, 'reward': 1.0})
print('FEEDBACK:', f.status_code, f.json())
# Stats
s = client.get('/stats')
print('STATS:', s.status_code, s.json())
