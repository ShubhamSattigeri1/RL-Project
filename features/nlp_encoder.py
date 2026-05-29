"""
NLP encoder using sentence-transformers (all-MiniLM-L6-v2).
Encodes user profile_text and content synopsis into 384-dim vectors.
Falls back to deterministic hash-based encoding if models can't be loaded.
"""

import numpy as np
import sys

HAS_ST = False  # Will be set to True when successfully loaded
_st_model = None  # Will hold SentenceTransformer once loaded
_load_attempted = False

def _ensure_st_loaded():
    """Lazy import sentence_transformers on first use with timeout."""
    global HAS_ST, _st_model, _load_attempted
    if _load_attempted:
        return  # Already tried, don't retry
    
    _load_attempted = True
    print("[NLPEncoder] Loading sentence-transformers model...", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        print("[NLPEncoder] Attempting to import sentence_transformers...", file=sys.stderr)
        sys.stderr.flush()
        from sentence_transformers import SentenceTransformer
        print("[NLPEncoder] Importing SentenceTransformer...", file=sys.stderr)
        sys.stderr.flush()
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[NLPEncoder] Model loaded successfully!", file=sys.stderr)
        sys.stderr.flush()
        HAS_ST = True
    except Exception as e:
        print(f"[NLPEncoder] Failed to load SentenceTransformer: {e}", file=sys.stderr)
        print(f"[NLPEncoder] Will use fallback deterministic embeddings", file=sys.stderr)
        sys.stderr.flush()
        HAS_ST = False
        _st_model = None

class NLPEncoder:
    def __init__(self):
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if not self._loaded:
            _ensure_st_loaded()
            self.model = _st_model
            self._loaded = True

    def encode(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        if self.model is not None:
            try:
                emb = self.model.encode(text, convert_to_numpy=True)
                emb = emb.astype('float32')
            except Exception as e:
                print(f"[NLPEncoder] Encoding failed: {e}, using fallback", file=sys.stderr)
                sys.stderr.flush()
                # fallback
                vec = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
                rng = np.random.default_rng(np.sum(vec))
                emb = rng.standard_normal(384).astype('float32')
        else:
            # fallback deterministic hash-based vector
            vec = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
            rng = np.random.default_rng(np.sum(vec))
            emb = rng.standard_normal(384).astype('float32')
        # L2-normalise
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def encode_user(self, user: dict) -> np.ndarray:
        text = user.get('profile_text', '') + ' | ' + ','.join(user.get('watch_history', []))
        return self.encode(text)

    def encode_content(self, content: dict) -> np.ndarray:
        return self.encode(content.get('synopsis', ''))
