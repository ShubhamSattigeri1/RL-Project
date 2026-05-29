"""
FastAPI application entry point.
"""
import os
import json
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from artwork_bandit.db import database
from artwork_bandit.features.nlp_encoder import NLPEncoder
from artwork_bandit.features.vision_encoder import VisionEncoder
from artwork_bandit.features.feature_store import FeatureStore
from artwork_bandit.bandit.linucb import LinUCBBandit
from artwork_bandit.bandit.thompson import ThompsonBandit

APP = FastAPI()

APP.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
APP.include_router(router)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
USERS_F = os.path.join(DATA_DIR, 'users.json')
CONTENTS_F = os.path.join(DATA_DIR, 'content.json')
ARTWORK_F = os.path.join(DATA_DIR, 'artwork.json')

LINUCB_ALPHA = float(os.getenv('LINUCB_ALPHA', '1.0'))

@APP.on_event('startup')
def startup():
    try:
        print("[STARTUP] Loading contents...")
        # load data
        with open(CONTENTS_F, 'r', encoding='utf-8') as f:
            contents = json.load(f)
        print(f"[STARTUP] Loaded {len(contents)} contents")
        
        print("[STARTUP] Checking users...")
        # generate users/artworks if missing
        if not os.path.exists(USERS_F) or os.path.getsize(USERS_F) == 0:
            try:
                from artwork_bandit.data.generate_synthetic import main as gen_main
                gen_main()
            except Exception as e:
                print(f"[STARTUP] Warning: Failed to generate users: {e}")
                with open(USERS_F, 'w', encoding='utf-8') as f:
                    json.dump([], f)
        with open(USERS_F, 'r', encoding='utf-8') as f:
            users = json.load(f)
        print(f"[STARTUP] Loaded {len(users)} users")
        
        print("[STARTUP] Checking artworks...")
        if not os.path.exists(ARTWORK_F) or os.path.getsize(ARTWORK_F) == 0:
            try:
                from artwork_bandit.data.generate_synthetic import main as gen_main
                gen_main()
            except Exception as e:
                print(f"[STARTUP] Warning: Failed to generate artworks: {e}")
                with open(ARTWORK_F, 'w', encoding='utf-8') as f:
                    json.dump([], f)
        with open(ARTWORK_F, 'r', encoding='utf-8') as f:
            artworks = json.load(f)
        print(f"[STARTUP] Loaded {len(artworks)} artworks")
        
        print("[STARTUP] Initializing database...")
        # init db
        database.init_db()
        print("[STARTUP] Database initialized")
        
        print("[STARTUP] Creating encoders...")
        # encoders - these are now lazy-loaded
        nlp = NLPEncoder()
        vis = VisionEncoder()
        print("[STARTUP] Encoders created (lazy-loaded)")
        
        print("[STARTUP] Creating feature store...")
        fs = FeatureStore(database, nlp, vis)
        print("[STARTUP] Feature store created")
        
        print("[STARTUP] Precomputing embeddings...")
        # precompute
        fs.precompute_all(users, contents, artworks)
        print("[STARTUP] Embeddings precomputed")
        
        print("[STARTUP] Creating bandits...")
        # bandits
        # collect all artwork ids
        all_arts = [a['artwork_id'] for a in artworks]
        d = 384 + 384 + 512
        linucb = LinUCBBandit(all_arts, d=d, alpha=LINUCB_ALPHA)
        try:
            linucb.load(database)
        except Exception as e:
            print(f"[STARTUP] Warning: Failed to load linucb state: {e}")
        thompson = ThompsonBandit(all_arts)
        print(f"[STARTUP] Bandits created (LinUCB with {len(all_arts)} arms, Thompson with {len(all_arts)} arms)")
        
        print("[STARTUP] Attaching state...")
        # attach
        APP.state.feature_store = fs
        APP.state.linucb = linucb
        APP.state.thompson = thompson
        APP.state.users = users
        APP.state.contents = contents
        APP.state.artworks = artworks
        print("[STARTUP] State attached - startup complete!")
        
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise


def init_app():
    """Idempotent initialisation run at import time to help tests and CLI use.
    Runs a safe subset of the startup routine and ensures DB tables exist.
    """
    try:
        # create DB tables early so tests don't fail if startup() isn't invoked
        database.init_db()
    except Exception:
        pass
    try:
        # load minimal data and set placeholder state if startup hasn't run
        if not hasattr(APP.state, 'feature_store'):
            with open(CONTENTS_F, 'r', encoding='utf-8') as f:
                contents = json.load(f)
            # ensure users/artworks exist
            if not os.path.exists(USERS_F) or os.path.getsize(USERS_F) == 0:
                try:
                    from artwork_bandit.data.generate_synthetic import main as gen_main
                    gen_main()
                except Exception:
                    with open(USERS_F, 'w', encoding='utf-8') as f:
                        json.dump([], f)
            with open(USERS_F, 'r', encoding='utf-8') as f:
                users = json.load(f)
            if not os.path.exists(ARTWORK_F) or os.path.getsize(ARTWORK_F) == 0:
                try:
                    from data.generate_synthetic import main as gen_main
                    gen_main()
                except Exception:
                    with open(ARTWORK_F, 'w', encoding='utf-8') as f:
                        json.dump([], f)
            with open(ARTWORK_F, 'r', encoding='utf-8') as f:
                artworks = json.load(f)
            nlp = NLPEncoder()
            vis = VisionEncoder()
            fs = FeatureStore(database, nlp, vis)
            fs.precompute_all(users, contents, artworks)
            all_arts = [a['artwork_id'] for a in artworks]
            d = 384 + 384 + 512
            linucb = LinUCBBandit(all_arts, d=d, alpha=LINUCB_ALPHA)
            thompson = ThompsonBandit(all_arts)
            APP.state.feature_store = fs
            APP.state.linucb = linucb
            APP.state.thompson = thompson
            APP.state.users = users
            APP.state.contents = contents
            APP.state.artworks = artworks
    except Exception as e:
        print(f"ERROR in init_app: {e}")
        import traceback
        traceback.print_exc()


# NOTE: Do NOT call init_app() at import time - let the startup event handle initialization
# This prevents encoder loading and model downloads from blocking the import

@APP.get('/health')
def health():
    # impressions count
    from artwork_bandit.db.database import SessionLocal, Impression
    db = SessionLocal()
    try:
        cnt = db.query(Impression).count()
    finally:
        db.close()
    return {"status": "ok", "impressions": cnt, "features_loaded": True}

@APP.on_event('shutdown')
def shutdown():
    try:
        APP.state.linucb.save(database)
    except Exception:
        pass

app = APP

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=True)
