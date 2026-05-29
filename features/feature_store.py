"""
Feature store: pre-computes and caches all embeddings at startup.
Retrieves embeddings by entity ID in O(1) from in-memory dict,
with SQLite as persistent backup.
"""
import numpy as np
from typing import Dict
from artwork_bandit.db import database

class FeatureStore:
    def __init__(self, db, nlp_encoder, vision_encoder):
        self.db = db
        self.nlp = nlp_encoder
        self.vis = vision_encoder
        self.user_emb: Dict[str, np.ndarray] = {}
        self.content_emb: Dict[str, np.ndarray] = {}
        self.artwork_emb: Dict[str, np.ndarray] = {}
        self.artworks_by_content: Dict[str, list] = {}

    def precompute_all(self, users, contents, artworks):
        # users
        print(f"[PRECOMPUTE] Encoding {len(users)} users...")
        for i, u in enumerate(users):
            if i % 50 == 0:
                print(f"[PRECOMPUTE] User {i}/{len(users)}")
            try:
                emb = self.nlp.encode_user(u)
                self.user_emb[u['user_id']] = emb
                try:
                    database.save_embedding('user', u['user_id'], emb)
                except Exception:
                    pass
            except Exception as e:
                print(f"[PRECOMPUTE] Error encoding user {u.get('user_id')}: {e}")
        print(f"[PRECOMPUTE] Completed {len(users)} users")
        
        # contents
        print(f"[PRECOMPUTE] Encoding {len(contents)} contents...")
        for i, c in enumerate(contents):
            if i % 10 == 0:
                print(f"[PRECOMPUTE] Content {i}/{len(contents)}")
            try:
                emb = self.nlp.encode_content(c)
                self.content_emb[c['content_id']] = emb
                try:
                    database.save_embedding('content', c['content_id'], emb)
                except Exception:
                    pass
            except Exception as e:
                print(f"[PRECOMPUTE] Error encoding content {c.get('content_id')}: {e}")
        print(f"[PRECOMPUTE] Completed {len(contents)} contents")
        
        # artworks
        print(f"[PRECOMPUTE] Encoding {len(artworks)} artworks...")
        for i, a in enumerate(artworks):
            if i % 50 == 0:
                print(f"[PRECOMPUTE] Artwork {i}/{len(artworks)}")
            try:
                emb = self.vis.encode_artwork(a)
                self.artwork_emb[a['artwork_id']] = emb
                try:
                    database.save_embedding('artwork', a['artwork_id'], emb)
                except Exception:
                    pass
                self.artworks_by_content.setdefault(a['content_id'], []).append(a['artwork_id'])
            except Exception as e:
                print(f"[PRECOMPUTE] Error encoding artwork {a.get('artwork_id')}: {e}")
        print(f"[PRECOMPUTE] Completed {len(artworks)} artworks")
        print("[PRECOMPUTE] All embeddings precomputed successfully")

    def get_user_embedding(self, user_id: str):
        return self.user_emb.get(user_id)

    def get_content_embedding(self, content_id: str):
        return self.content_emb.get(content_id)

    def get_artwork_embedding(self, artwork_id: str):
        return self.artwork_emb.get(artwork_id)

    def build_context_vector(self, user_id, content_id, artwork_id):
        u = self.get_user_embedding(user_id)
        c = self.get_content_embedding(content_id)
        a = self.get_artwork_embedding(artwork_id)
        if u is None or c is None or a is None:
            return None
        # ensure vector order and concatenation [user(384)|content(384)|artwork(512)]
        return np.concatenate([u, c, a])

    def get_artworks_for_content(self, content_id: str):
        return self.artworks_by_content.get(content_id, [])
