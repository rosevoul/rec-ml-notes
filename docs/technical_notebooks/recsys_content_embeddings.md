# Content Embeddings (Text, Vision, and CLIP)
**Scope:** Builds item embeddings from content (text/images) for retrieval and cold-start coverage. Produces vectors for the embedding store and ANN indexing. Does not personalize or rank.

## What this component does (and does not do)
- Generates item vectors from item content fields (title/description/images).
- Supports multiple backends: minimal baselines and modern encoders.
- Produces versioned embedding artifacts consumable by retrieval.
- Does not train rankers or apply business rules.

## When this component is used
- Cold-start items need vectors before interaction data accrues.
- You need semantic coverage beyond behavior-only embeddings.
- You want multi-modal retrieval (text query → image-like items, etc.).
- Embedding computation runs offline with periodic refresh.

## Integration points

```
Item catalog + content
   ↓
Content embedding job (text/image/CLIP)
   ↓
Embedding store (versioned)
   ↓
ANN index build + publish
   ↓
Serving retrieval / ranking
```

Embedding versioning is the safety mechanism. Rollback is a version switch.

## Example inputs / outputs

Input (catalog row):
```json
{
  "item_id": 712,
  "title": "Ergonomic Mesh Office Chair",
  "description": "Adjustable lumbar support and breathable mesh.",
  "image_urls": ["https://.../712_main.jpg"]
}
```

Output:
```json
{
  "item_id": 712,
  "embedding": {"dim": 512, "norm": 1.0},
  "content_emb_version": "content_v12"
}
```

## Core implementation (handoff-grade)

### 1) Shared contracts
All embedding backends must return fixed-dim, L2-normalized vectors.

```python
import numpy as np

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)

class EmbeddingBackend:
    dim: int
    def encode_text(self, texts: list[str]) -> np.ndarray: ...
    def encode_image(self, images: list[np.ndarray]) -> np.ndarray: ...
```

### 2) Minimal text baseline: TF‑IDF → SVD (offline-friendly)
Use this when you need a deterministic baseline without model serving.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def fit_tfidf_svd(texts: list[str], *, dim: int = 128):
    tfidf = TfidfVectorizer(min_df=2, max_features=200_000, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=dim, random_state=0)
    Z = svd.fit_transform(X)
    return tfidf, svd, l2_normalize(Z)

def encode_tfidf_svd(tfidf, svd, texts: list[str]) -> np.ndarray:
    X = tfidf.transform(texts)
    Z = svd.transform(X)
    return l2_normalize(Z)
```

Why this exists:
- Works offline.
- No GPU dependency.
- Gives decent lexical-semantic coverage for titles/short text.

### 3) Modern text embeddings: Sentence-BERT style (served)
Treat this as a model service with versioning and batching.

```python
def encode_text_sbert(texts: list[str], *, model_version: str) -> np.ndarray:
    resp = text_embed_svc.batch_encode(
        texts=texts,
        model_version=model_version,
        max_batch=256,
        timeout_ms=5000
    )
    E = np.asarray(resp["embeddings"], dtype=np.float32)
    return l2_normalize(E)
```

### 4) Minimal vision baseline: downsample + gradients (cheap sanity)
This is not SOTA. It’s a baseline to validate plumbing and indexing.

```python
def encode_image_minimal(imgs: list[np.ndarray], *, out_dim: int = 128) -> np.ndarray:
    feats = []
    for im in imgs:
        small = im[::4, ::4]                     # downsample
        gx = np.diff(small, axis=1, prepend=0)
        gy = np.diff(small, axis=0, prepend=0)
        f = np.concatenate([small.ravel(), gx.ravel(), gy.ravel()])[:out_dim]
        feats.append(f.astype(np.float32))
    return l2_normalize(np.vstack(feats))
```

### 5) Modern vision embeddings: ViT encoder (served)
Same pattern as text: batch, version, normalize.

```python
def encode_image_vit(img_bytes: list[bytes], *, model_version: str) -> np.ndarray:
    resp = vision_embed_svc.batch_encode(
        images=img_bytes,
        model_version=model_version,
        max_batch=64,
        timeout_ms=8000
    )
    E = np.asarray(resp["embeddings"], dtype=np.float32)
    return l2_normalize(E)
```

### 6) CLIP: joint space for text ↔ image retrieval
CLIP is the cleanest way to align query text and item images without custom training.

```python
def encode_clip_text(texts: list[str], *, model_version: str) -> np.ndarray:
    E = clip_svc.encode_text(texts=texts, model_version=model_version, max_batch=256)
    return l2_normalize(np.asarray(E, dtype=np.float32))

def encode_clip_image(img_bytes: list[bytes], *, model_version: str) -> np.ndarray:
    E = clip_svc.encode_image(images=img_bytes, model_version=model_version, max_batch=64)
    return l2_normalize(np.asarray(E, dtype=np.float32))
```

### 7) Item vector assembly and persistence
Decide upfront whether you store separate modalities or one fused vector.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ContentConfig:
    version: str = "content_v12"
    backend: str = "clip"         # "tfidf_svd" | "sbert" | "vit" | "clip"
    dim: int = 512

CFG = ContentConfig()

def build_item_embedding(row: dict) -> np.ndarray:
    if CFG.backend == "clip":
        txt = row["title"] + " " + (row.get("description") or "")
        e_txt = encode_clip_text([txt], model_version="clip_v3")[0]
        e_img = encode_clip_image([download_image(row["image_urls"][0])], model_version="clip_v3")[0]
        e = l2_normalize(0.3 * e_txt + 0.7 * e_img)  # simple, stable fusion
        return e

    raise NotImplementedError(CFG.backend)

def write_embeddings(rows: list[dict]):
    for r in rows:
        e = build_item_embedding(r)
        embedding_store.put(item_id=r["item_id"], vec=e, version=CFG.version)
```

## Guardrails and failure modes
- **Version skew:** retrieval index built from content_v11 while serving uses content_v12; enforce pointer consistency.
- **Batching bugs:** embedding services return out-of-order results; include item_id alignment checks.
- **Normalization drift:** some backends return unnormalized vectors; enforce L2 at boundary.
- **Missing images/text:** do not “invent” content; fall back to the available modality or mark as missing.
- **Distribution shift:** catalog text changes (templates, spam); baselines (TF‑IDF) can degrade sharply.

## Known limitations
- Content embeddings capture similarity, not preference.
- CLIP fusion weights are heuristic unless trained; keep them stable and versioned.
- Encoding jobs can be expensive; plan refresh cadence and incremental updates.
