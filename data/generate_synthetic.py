import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent

def load_contents():
    with open(BASE / 'content.json', 'r', encoding='utf-8') as f:
        return json.load(f)

first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Drew", "Cameron", "Avery", "Quinn", "Hayden", "Parker", "Skyler", "Reese", "Kendall", "Blake", "Logan", "Rowan", "Elliot"]
last_names = ["Smith", "Johnson", "Brown", "Lee", "Garcia", "Martinez", "Davis", "Lopez", "Wilson", "Anderson"]

genres = ["action", "romance", "thriller", "comedy", "drama", "scifi", "horror"]

def gen_users(n=200):
    contents = load_contents()
    titles = [c['title'] for c in contents]
    users = []
    for i in range(1, n+1):
        uid = f'user_{i:03d}'
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        # pick a favored genre
        fav = genres[(i-1) % len(genres)]
        weights = [0.05] * len(genres)
        fav_idx = genres.index(fav)
        weights[fav_idx] = 0.6
        # normalize
        s = sum(weights)
        affinity = {g: round(w/s, 3) for g,w in zip(genres, weights)}
        # watch history: 5-10 titles
        k = random.randint(5,10)
        watch = [random.choice(titles) for _ in range(k)]
        profile = f"Enjoys {fav} and similar genres; prefers engaging, character-led stories."
        users.append({
            'user_id': uid,
            'name': name,
            'genre_affinity': affinity,
            'watch_history': watch,
            'profile_text': profile
        })
    return users

moods = ["intense", "warm", "mysterious", "comedic", "epic"]
focal_points = ["solo_hero", "ensemble_cast", "explosion_action", "romantic_leads", "villain_close_up"]
phrases = ["Only one survives", "Love conquers all", "The city burns tonight", "A secret revealed", "Nothing is as it seems"]

def gen_artworks():
    contents = load_contents()
    artworks = []
    art_idx = 1
    for c in contents:
        for variant in range(5):
            aid = f'art_{art_idx:03d}'
            artworks.append({
                'artwork_id': aid,
                'content_id': c['content_id'],
                'variant_index': variant,
                'focal_point': random.choice(focal_points),
                'mood': random.choice(moods),
                'text_overlay': random.choice(phrases),
                'description': f"{random.choice(focal_points)} with {random.choice(moods)} lighting and dramatic composition."
            })
            art_idx += 1
    return artworks


def main():
    users = gen_users()
    with open(BASE / 'users.json', 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)
    artworks = gen_artworks()
    with open(BASE / 'artwork.json', 'w', encoding='utf-8') as f:
        json.dump(artworks, f, indent=2)
    print('Generated users.json and artwork.json')

if __name__ == '__main__':
    main()
