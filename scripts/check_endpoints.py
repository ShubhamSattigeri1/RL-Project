from fastapi.testclient import TestClient
from artwork_bandit.api.main import app

client = TestClient(app)
print('GET /users')
r = client.get('/users')
print(r.status_code)
print(r.json()[:5])
print('\nGET /contents')
r = client.get('/contents')
print(r.status_code)
print(r.json()[:5])
