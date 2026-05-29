"""
Vision encoder using open-clip (ViT-B-32).
Encodes artwork description text into 512-dim vectors.
(Using text-based CLIP encoding so no actual images are required.)
Falls back to deterministic hash-based encoding if models can't be loaded.
"""
import numpy as np
import sys

HAS_CLIP = False  # Will be set to True when successfully loaded
_clip_model = None
_clip_tokenizer = None
_clip_preprocess = None
_load_attempted = False

def _ensure_clip_loaded():
    """Lazy import open_clip on first use with error handling."""
    global HAS_CLIP, _clip_model, _clip_tokenizer, _clip_preprocess, _load_attempted
    if _load_attempted:
        return  # Already tried, don't retry
    
    _load_attempted = True
    print("[VisionEncoder] Loading open_clip model...", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        print("[VisionEncoder] Attempting to import open_clip...", file=sys.stderr)
        sys.stderr.flush()
        import open_clip
        import torch
        print("[VisionEncoder] Creating ViT-B-32 model...", file=sys.stderr)
        sys.stderr.flush()
        # create model; choose a commonly available pretrained
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s32b_b79k')
        _clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        _clip_model.eval()
        HAS_CLIP = True
        print("[VisionEncoder] Model loaded successfully!", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[VisionEncoder] Failed to load open_clip: {e}", file=sys.stderr)
        print(f"[VisionEncoder] Will use fallback deterministic embeddings", file=sys.stderr)
        sys.stderr.flush()
        HAS_CLIP = False
        _clip_model = None
        _clip_tokenizer = None
        _clip_preprocess = None

class VisionEncoder:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if not self._loaded:
            _ensure_clip_loaded()
            global _clip_model, _clip_tokenizer, _clip_preprocess
            self.model = _clip_model
            self.tokenizer = _clip_tokenizer
            self.preprocess = _clip_preprocess
            self._loaded = True

    def encode_text(self, description: str) -> np.ndarray:
        self._ensure_loaded()
        if self.model is not None:
            try:
                import torch
                tokens = self.tokenizer(description)
                with torch.no_grad():
                    txt = self.model.encode_text(tokens)
                    emb = txt.cpu().numpy().astype('float32')
            except Exception as e:
                print(f"[VisionEncoder] Encoding failed: {e}, using fallback", file=sys.stderr)
                sys.stderr.flush()
                # fallback
                vec = np.frombuffer(description.encode('utf-8'), dtype=np.uint8)
                rng = np.random.default_rng(np.sum(vec))
                emb = rng.standard_normal(512).astype('float32')
        else:
            vec = np.frombuffer(description.encode('utf-8'), dtype=np.uint8)
            rng = np.random.default_rng(np.sum(vec))
            emb = rng.standard_normal(512).astype('float32')
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def encode_artwork(self, artwork: dict) -> np.ndarray:
        return self.encode_text(artwork.get('description', ''))
