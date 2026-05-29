from fastapi import APIRouter, HTTPException, Request
from .schemas import RecommendRequest, RecommendResponse, FeedbackRequest, FeedbackResponse, StatsResponse
import time
from artwork_bandit.db import database
import numpy as np

router = APIRouter()


@router.get('/users')
async def list_users(request: Request):
    app = request.app
    users = getattr(app.state, 'users', None)
    if users is None:
        raise HTTPException(status_code=500, detail='users not loaded')
    # return compact list
    return [{'user_id': u.get('user_id'), 'age': u.get('age'), 'mood': u.get('mood')} for u in users]


@router.get('/contents')
async def list_contents(request: Request):
    app = request.app
    contents = getattr(app.state, 'contents', None)
    if contents is None:
        raise HTTPException(status_code=500, detail='contents not loaded')
    return [{'content_id': c.get('content_id'), 'title': c.get('title', ''), 'genre': c.get('genre')} for c in contents]

@router.post('/recommend', response_model=RecommendResponse)
async def recommend(req: RecommendRequest, request: Request):
    app = request.app
    start = time.time()
    fs = app.state.feature_store
    if fs is None:
        raise HTTPException(status_code=500, detail='Feature store not initialised')
    # validate
    if fs.get_user_embedding(req.user_id) is None:
        raise HTTPException(status_code=404, detail='user_id not found')
    if fs.get_content_embedding(req.content_id) is None:
        raise HTTPException(status_code=404, detail='content_id not found')
    artworks = fs.get_artworks_for_content(req.content_id)
    if not artworks or len(artworks) < 1:
        raise HTTPException(status_code=404, detail='no artworks for content')
    # build contexts
    contexts = {}
    for aid in artworks:
        vec = fs.build_context_vector(req.user_id, req.content_id, aid)
        if vec is None:
            continue
        contexts[aid] = vec
    # select
    algorithm = req.algorithm.lower()
    if algorithm == 'linucb':
        bandit = app.state.linucb
        chosen = bandit.select(contexts)
    elif algorithm == 'thompson':
        bandit = app.state.thompson
        chosen = bandit.select(list(contexts.keys()))
    else:
        raise HTTPException(status_code=422, detail='unknown algorithm')
    impression_id = database.log_impression(req.user_id, req.content_id, chosen)
    latency_ms = (time.time() - start) * 1000.0
    if latency_ms > 100:
        app.logger = getattr(app, 'logger', None)
        if app.logger:
            app.logger.warning(f"High latency: {latency_ms}ms")

    artwork = next((a for a in app.state.artworks if a.get('artwork_id') == chosen), None)
    if artwork is None or artwork.get('artwork_image') is None:
        raise HTTPException(status_code=500, detail='artwork image not found')

    return RecommendResponse(
        artwork_id=chosen,
        content_id=req.content_id,
        user_id=req.user_id,
        algorithm=algorithm,
        latency_ms=latency_ms,
        impression_id=impression_id,
        artwork_image=artwork['artwork_image'],
    )

@router.post('/feedback', response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest, request: Request):
    app = request.app
    imp = database.get_impression(req.impression_id)
    if imp is None:
        raise HTTPException(status_code=404, detail='impression not found')
    if req.reward not in (0.0, 1.0):
        raise HTTPException(status_code=422, detail='reward must be 0.0 or 1.0')
    database.log_reward(req.impression_id, req.reward)
    fs = app.state.feature_store
    ctx = fs.build_context_vector(imp.user_id, imp.content_id, imp.artwork_id)
    if ctx is None:
        raise HTTPException(status_code=500, detail='context not found')
    # update both bandits
    if hasattr(app.state, 'linucb'):
        app.state.linucb.update(imp.artwork_id, ctx, req.reward)
    if hasattr(app.state, 'thompson'):
        app.state.thompson.update(imp.artwork_id, req.reward)
    return FeedbackResponse(status='ok', impression_id=req.impression_id, updated_arm=imp.artwork_id)

@router.get('/stats', response_model=StatsResponse)
async def stats(request: Request):
    rows = database.impressions_with_rewards()
    total = len(rows)
    clicks = sum(1 for r in rows if r.reward and r.reward > 0)
    impressions_by = {}
    clicks_by = {}
    for r in rows:
        impressions_by[r.artwork_id] = impressions_by.get(r.artwork_id, 0) + 1
        if r.reward and r.reward > 0:
            clicks_by[r.artwork_id] = clicks_by.get(r.artwork_id, 0) + 1
    overall_ctr = float(clicks) / total if total > 0 else 0.0
    return StatsResponse(total_impressions=total, total_clicks=clicks, overall_ctr=overall_ctr, impressions_by_artwork=impressions_by, clicks_by_artwork=clicks_by)
