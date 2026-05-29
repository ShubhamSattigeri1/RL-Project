# Artwork Bandit

Project implementing a contextual bandit system for dynamic artwork thumbnail selection.

Architecture:

Data -> Feature Store -> Bandit -> API -> User -> Feedback -> Bandit

Setup:

```bash
pip install -r requirements.txt
python -m api.main
```

API Endpoints:

- `POST /recommend` - payload: `{"user_id":..., "content_id":..., "algorithm":"linucb"}` - returns artwork recommendation and impression id
- `POST /feedback` - payload: `{"impression_id":..., "reward": 0.0|1.0}` - records feedback and updates bandit
- `GET /stats` - returns CTR and counts

Run tests:

```bash
pytest tests/ -v
```

Run evaluation:

```bash
python -m simulation.evaluate
```

Key hyperparameters:

- alpha (LinUCB): 1.0 (exploration)
- d (context dim): 1280
- n_arms: 5
