import numpy as np
from artwork_bandit.features.nlp_encoder import NLPEncoder
from artwork_bandit.features.vision_encoder import VisionEncoder
from artwork_bandit.features.feature_store import FeatureStore


def test_nlp_encoder_shape():
    enc = NLPEncoder()
    v = enc.encode('Hello world')
    assert v.shape == (384,)
    assert v.dtype == 'float32'

def test_vision_encoder_shape():
    enc = VisionEncoder()
    v = enc.encode_text('A dramatic close up')
    assert v.shape == (512,)
    assert v.dtype == 'float32'

def test_feature_store_context_vector():
    # minimal synthetic
    users = [{'user_id':'user_001','profile_text':'Likes action','watch_history':['Midnight Chase'],'genre_affinity':{'action':1.0,'romance':0,'thriller':0,'comedy':0,'drama':0,'scifi':0,'horror':0}}]
    contents = [{'content_id':'content_001','synopsis':'An action film.'}]
    artworks = [{'artwork_id':'art_001','content_id':'content_001','description':'explosion action','variant_index':0,'focal_point':'explosion_action','mood':'intense','text_overlay':'Boom'}]
    n = NLPEncoder(); v = VisionEncoder(); fs = FeatureStore(None, n, v)
    fs.precompute_all(users, contents, artworks)
    ctx = fs.build_context_vector('user_001','content_001','art_001')
    assert ctx.shape == (384+384+512,)
    assert np.all(np.isfinite(ctx))
