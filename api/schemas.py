from pydantic import BaseModel, Field
from typing import Dict

class RecommendRequest(BaseModel):
    user_id: str
    content_id: str
    algorithm: str = Field(default="linucb")

class RecommendResponse(BaseModel):
    artwork_id: str
    content_id: str
    user_id: str
    algorithm: str
    latency_ms: float
    impression_id: int
    artwork_image: str

class FeedbackRequest(BaseModel):
    impression_id: int
    reward: float

class FeedbackResponse(BaseModel):
    status: str
    impression_id: int
    updated_arm: str

class StatsResponse(BaseModel):
    total_impressions: int
    total_clicks: int
    overall_ctr: float
    impressions_by_artwork: Dict[str, int]
    clicks_by_artwork: Dict[str, int]
