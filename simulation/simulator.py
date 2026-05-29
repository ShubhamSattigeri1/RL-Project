"""
Simulates user click behaviour for offline evaluation.
"""
import numpy as np

MOOD_MAP = {
    'intense': ['action', 'thriller'],
    'warm': ['romance', 'comedy'],
    'mysterious': ['thriller', 'horror', 'scifi'],
    'comedic': ['comedy'],
    'epic': ['action', 'scifi', 'drama']
}

class ArtworkSimulator:
    def __init__(self, users, contents, artworks):
        self.users = {u['user_id']: u for u in users}
        self.contents = {c['content_id']: c for c in contents}
        self.artworks = {a['artwork_id']: a for a in artworks}

    def simulate_click(self, user_id, artwork_id) -> float:
        user = self.users[user_id]
        art = self.artworks[artwork_id]
        mood = art.get('mood')
        mapped = MOOD_MAP.get(mood, [])
        affinity_sum = sum(user['genre_affinity'].get(g, 0.0) for g in mapped)
        if len(mapped) > 0:
            click_prob = 0.10 + (0.70 * affinity_sum / len(mapped))
        else:
            click_prob = 0.10
        click_prob = np.clip(click_prob + np.random.normal(0, 0.05), 0.05, 0.95)
        return 1.0 if np.random.random() < click_prob else 0.0

    def get_oracle_artwork(self, user_id, content_id):
        # among artworks for content, pick highest true click_prob
        arts = [a for a in self.artworks.values() if a['content_id'] == content_id]
        best = None
        best_p = -1
        for a in arts:
            mood = a.get('mood')
            mapped = MOOD_MAP.get(mood, [])
            affinity_sum = sum(self.users[user_id]['genre_affinity'].get(g, 0.0) for g in mapped)
            if len(mapped) > 0:
                p = 0.10 + (0.70 * affinity_sum / len(mapped))
            else:
                p = 0.10
            if p > best_p:
                best_p = p
                best = a['artwork_id']
        return best
