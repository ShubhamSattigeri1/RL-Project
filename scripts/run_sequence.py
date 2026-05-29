import requests
BASE='http://127.0.0.1:8000'

print('GET /users')
r = requests.get(BASE + '/users')
print(r.status_code)
users = r.json()
print(users[:3])

print('\nGET /contents')
r = requests.get(BASE + '/contents')
print(r.status_code)
contents = r.json()
print(contents[:3])

user_id = users[0]['user_id']
content_id = contents[0]['content_id']
print(f'\nPOST /recommend user={user_id} content={content_id}')
r = requests.post(BASE + '/recommend', json={'user_id': user_id, 'content_id': content_id, 'algorithm': 'linucb'})
print(r.status_code)
rec = r.json()
print(rec)
imp_id = rec.get('impression_id')

print(f'\nPOST /feedback impression_id={imp_id} reward=1.0')
r = requests.post(BASE + '/feedback', json={'impression_id': imp_id, 'reward': 1.0})
print(r.status_code)
print(r.json())

print('\nGET /stats')
r = requests.get(BASE + '/stats')
print(r.status_code)
print(r.json())
