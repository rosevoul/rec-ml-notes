# User Embeddings (for Retrieval)
**Scope:** Builds a user vector from recent interaction history in the same embedding space as items. Used to query ANN for candidate retrieval. Does not rank or gate items.

## What this component does (and does not do)
- Produces a single L2-normalized vector per request (or per cached user state).
- Uses recent item interactions as the primary signal.
- Does not apply business rules, filtering, or final ordering.
- Must fail open to a fallback retrieval strategy if history or embeddings are missing.

## When this component is used
- Retrieval is embedding-based (ANN over item vectors).
- Users have sparse/noisy histories and you need a stable default.
- Latency requires cheap request-time compute or cached embeddings.
- You want failure isolation: ranking should still run if user embeddings fail.

## Integration points

```
User events → History buffer → User embedding → ANN retrieval → Candidate merge → Ranker
```

User embedding is a query vector. It should not depend on retrieval outcomes (avoid feedback loops on the request path).

## Example inputs / outputs

Input (cached history + item embedding store):
```json
{
  "user_id": 42,
  "history_item_ids": [311, 98, 712, 45, 45, 901],
  "history_ts":       [1700,1705,1710,1713,1715,1718],
  "item_embedding_version": "items_v42"
}
```

Output (what retrieval consumes):
```json
{
  "user_embedding": {"dim": 48, "norm": 1.0},
  "user_embedding_version": "user_v7",
  "depends_on": {"item_embedding_version": "items_v42"}
}
```

## Core implementation (handoff-grade)

### 1) Shared helpers (normalization and safety)
User vectors must be normalized consistently with item vectors.

```python
import numpy as np

def l2_normalize(x: np.ndarray, axis=-1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def safe_take_item_emb(item_ids: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    # item_emb shape: [n_items, d]; item_ids shape: [L] or [B, L]
    return item_emb[item_ids]
```

### 2) Baseline: recency-weighted average (ships first)
This is the default because it is stable, cheap, and hard to break.

```python
def user_embed_avg(history_item_ids: np.ndarray,
                   item_emb: np.ndarray,
                   *,
                   decay: float = 0.05) -> np.ndarray:
    # history_item_ids: [L], newest is last
    L = int(history_item_ids.shape[0])
    if L == 0:
        raise ValueError("empty_history")

    t = np.arange(L, dtype=np.float32)
    w = np.exp(decay * (t - (L - 1))).astype(np.float32)   # newest ~ 1
    w = w / (w.sum() + 1e-12)

    X = safe_take_item_emb(history_item_ids, item_emb)     # [L, d]
    u = (X * w[:, None]).sum(axis=0)                       # [d]
    return l2_normalize(u)
```

Design intent encoded:
- Recency weighting handles mild intent drift without learning.
- One vector, one norm, deterministic output.

### 3) Upgrade: short-term + long-term mixture
This is the minimal “more modern” variant without training.

```python
def user_embed_mix(history_item_ids: np.ndarray,
                   item_emb: np.ndarray,
                   *,
                   alpha_short: float = 0.6) -> np.ndarray:
    u_long  = user_embed_avg(history_item_ids, item_emb, decay=0.01)
    u_short = user_embed_avg(history_item_ids, item_emb, decay=0.15)
    u = (1 - alpha_short) * u_long + alpha_short * u_short
    return l2_normalize(u)
```

Why this exists:
- Long-term stabilizes.
- Short-term reacts.
- Still debuggable and deterministic.

### 4) Learned encoder (two-tower style): contract only
You do not need the full training loop in the note, but you do need the feature contract.

```python
def make_user_features(history_item_ids: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    # Example: concatenate summary stats that are cheap to compute
    X = safe_take_item_emb(history_item_ids, item_emb)     # [L, d]
    mean = X.mean(axis=0)
    last = X[-1]
    short = (X[-5:]).mean(axis=0) if X.shape[0] >= 5 else mean
    return np.concatenate([mean, short, last], axis=0)     # [3d]

def user_embed_learned(history_item_ids, item_emb, mlp) -> np.ndarray:
    feat = make_user_features(history_item_ids, item_emb)
    u = mlp(feat)                                          # output dim = d
    return l2_normalize(u)
```

Key invariants:
- Output dimension must equal item embedding dimension.
- L2-normalize before retrieval.
- The encoder must be versioned against the item embedding space.

### 5) Fail-open behavior + fallback strategy
Never block retrieval. If user embedding fails, retrieval should still produce candidates.

```python
def fallback_user_embedding(d: int) -> np.ndarray:
    # Prefer a stable, versioned default (e.g., global centroid) over random.
    return l2_normalize(np.ones((d,), dtype=np.float32))

def build_user_embedding(user_id: int, history_item_ids: list[int], item_emb: np.ndarray) -> np.ndarray:
    try:
        h = np.asarray(history_item_ids, dtype=np.int64)
        if h.size < 2:
            raise ValueError("too_short_history")
        return user_embed_mix(h, item_emb)
    except Exception as e:
        log.warning("user_embed_fallback", extra={"user_id": user_id, "err": str(e)})
        return fallback_user_embedding(item_emb.shape[1])
```

### 6) Caching (required for real systems)
Compute once, reuse. Cache key must include item embedding version.

```python
def user_cache_key(user_id: int, item_emb_version: str, user_emb_version: str) -> str:
    return f"uemb:{user_emb_version}:{item_emb_version}:{user_id}"

def get_or_build_user_embedding(user_id: int, item_emb_version: str, user_emb_version: str):
    key = user_cache_key(user_id, item_emb_version, user_emb_version)
    cached = redis.get_vec(key)
    if cached is not None:
        return cached

    history = history_store.get_recent_items(user_id, limit=100)  # ordered oldest→newest
    item_emb = item_embedding_store.load(item_emb_version)
    u = build_user_embedding(user_id, history, item_emb)
    redis.set_vec(key, u, ttl_s=6 * 3600)
    return u
```

## Guardrails and failure modes
- **Space mismatch:** user encoder trained on different item embeddings; vectors look fine but neighbors are wrong.
- **History leakage:** using post-click signals in features; offline looks great, online drifts.
- **Cache staleness:** user intent changes but cached embedding doesn’t; short TTL or event-driven invalidation.
- **Popularity collapse:** averaging amplifies popular items; needs debiasing or reweighting in some domains.
- **Feedback loops:** embedding depends on retrieved items; keep query vector independent of retrieval results.

## Known limitations
- Averaging ignores order beyond recency weighting.
- Learned encoders add versioning and skew risk.
- Cold-start users still need a non-embedding fallback (popularity/lexical/rules).
