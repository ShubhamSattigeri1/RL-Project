import json
import datetime
import numpy as np
from sqlalchemy import (create_engine, Column, Integer, String, DateTime, Text, Float)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import NoResultFound
import os

Base = declarative_base()

DB_FILE = os.path.join(os.path.dirname(__file__), '..', 'artwork_bandit.db')
ENGINE = create_engine(f'sqlite:///{DB_FILE}', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE)

class Impression(Base):
    __tablename__ = 'impressions'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, index=True)
    content_id = Column(String, index=True)
    artwork_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    reward = Column(Float, nullable=True)

class BanditState(Base):
    __tablename__ = 'bandit_state'
    id = Column(Integer, primary_key=True)
    arm_id = Column(String, unique=True, index=True)
    A_matrix = Column(Text)
    b_vector = Column(Text)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class FeatureCache(Base):
    __tablename__ = 'feature_cache'
    id = Column(Integer, primary_key=True)
    entity_type = Column(String, index=True)
    entity_id = Column(String, index=True)
    embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=ENGINE)

def log_impression(user_id: str, content_id: str, artwork_id: str):
    db = SessionLocal()
    imp = Impression(user_id=user_id, content_id=content_id, artwork_id=artwork_id)
    db.add(imp)
    db.commit()
    db.refresh(imp)
    db.close()
    return imp.id

def log_reward(impression_id: int, reward: float):
    db = SessionLocal()
    imp = db.query(Impression).filter(Impression.id == impression_id).one()
    imp.reward = float(reward)
    db.commit()
    db.close()

def save_bandit_arm(arm_id: str, A: np.ndarray, b: np.ndarray):
    db = SessionLocal()
    textA = json.dumps(A.tolist())
    textb = json.dumps(b.tolist())
    try:
        state = db.query(BanditState).filter(BanditState.arm_id == arm_id).one()
        state.A_matrix = textA
        state.b_vector = textb
        state.updated_at = datetime.datetime.utcnow()
    except NoResultFound:
        state = BanditState(arm_id=arm_id, A_matrix=textA, b_vector=textb)
        db.add(state)
    db.commit()
    db.close()

def load_bandit_arm(arm_id: str):
    db = SessionLocal()
    try:
        state = db.query(BanditState).filter(BanditState.arm_id == arm_id).one()
        A = np.array(json.loads(state.A_matrix))
        b = np.array(json.loads(state.b_vector))
        db.close()
        return {'arm_id': arm_id, 'A': A, 'b': b}
    except NoResultFound:
        db.close()
        return None

def save_embedding(entity_type: str, entity_id: str, embedding: np.ndarray):
    db = SessionLocal()
    text = json.dumps(embedding.tolist())
    try:
        existing = db.query(FeatureCache).filter(FeatureCache.entity_type==entity_type, FeatureCache.entity_id==entity_id).one()
        existing.embedding = text
    except NoResultFound:
        entry = FeatureCache(entity_type=entity_type, entity_id=entity_id, embedding=text)
        db.add(entry)
    db.commit()
    db.close()

def load_embedding(entity_type: str, entity_id: str):
    db = SessionLocal()
    try:
        entry = db.query(FeatureCache).filter(FeatureCache.entity_type==entity_type, FeatureCache.entity_id==entity_id).one()
        emb = np.array(json.loads(entry.embedding))
        db.close()
        return emb
    except NoResultFound:
        db.close()
        return None

def get_impression(impression_id: int):
    db = SessionLocal()
    try:
        imp = db.query(Impression).filter(Impression.id==impression_id).one()
        db.expunge(imp)
        db.close()
        return imp
    except NoResultFound:
        db.close()
        return None

def impressions_with_rewards():
    db = SessionLocal()
    rows = db.query(Impression).filter(Impression.reward != None).all()
    db.close()
    return rows
