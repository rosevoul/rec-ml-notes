# Behavioral Item Embeddings (Co-occurrence, Item2Vec, Graph, Fusion)
**Scope:** Learns item vectors from user interaction sequences. Used for retrieval and similarity. Produces versioned item embeddings for ANN. Does not personalize by itself.

## What this component does (and does not do)
- Converts sessions / interaction sequences into item embeddings.
- Supports multiple methods with increasing complexity.
- Produces a single embedding table keyed by item_id and version.
- Does not build user vectors (that’s a separate component).
- Does not rank or enforce policy.

## When this component is used
- You have enough interaction volume to learn meaningful co-occurrence.
- Metadata is weak or noisy, but behavior is strong.
- You need better “people who viewed this also viewed” geometry for retrieval.
- You can run offline training jobs and publish immutable artifacts.

## Integration points

```
Interaction logs → Sessionize → Behavioral embedding job → Embedding store → ANN index → Retrieval
                                    └─(optional) fuse with content embeddings → same store/version
```

## Example inputs / outputs

Input (sessionized):
```json
{ "session_id": "abc", "items": [12, 45, 98, 45, 311] }
```

Output:
```json
{ "item_id": 45, "embedding": {"dim": 128, "norm": 1.0}, "behavior_emb_version": "beh_v9" }
```

## Core implementation (handoff-grade)

### 1) Common helpers
All methods converge to: build vectors → L2 normalize → write versioned table.

```python
import numpy as np

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def write_item_table(item_ids: np.ndarray, E: np.ndarray, *, version: str):
    for i, item_id in enumerate(item_ids):
        embedding_store.put(item_id=int(item_id), vec=E[i], version=version)
```

### 2) Method A: Co-occurrence → PPMI → SVD (baseline)
This is the “ships-first” behavioral method: deterministic, offline-friendly, solid geometry.

```python
def build_cooccurrence(sessions: list[list[int]], n_items: int, window: int = 5) -> np.ndarray:
    C = np.zeros((n_items, n_items), dtype=np.float32)
    for seq in sessions:
        L = len(seq)
        for i, center in enumerate(seq):
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                C[center, seq[j]] += 1.0
    return C

def ppmi(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = C.sum() + eps
    row = C.sum(axis=1, keepdims=True) + eps
    col = C.sum(axis=0, keepdims=True) + eps
    pmi = np.log((C * total) / (row * col))
    return np.maximum(pmi, 0.0).astype(np.float32)

def svd_embed(PPMI: np.ndarray, k: int = 128) -> np.ndarray:
    U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    E = U[:, :k] * np.sqrt(S[:k])[None, :]
    return l2_normalize(E.astype(np.float32))
```

### 3) Method B: Item2Vec (skip-gram + negative sampling)
Use when you want order/context learning without building a full graph pipeline.

```python
def make_skipgram_pairs(sessions: list[list[int]], window: int = 5) -> np.ndarray:
    pairs = []
    for seq in sessions:
        for i, center in enumerate(seq):
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, seq[j]))
    return np.asarray(pairs, dtype=np.int64)

def train_item2vec(pairs: np.ndarray, n_items: int, d: int = 128, neg_k: int = 10, steps: int = 50_000):
    rng = np.random.default_rng(0)
    W_in  = 0.01 * rng.normal(size=(n_items, d)).astype(np.float32)
    W_out = 0.01 * rng.normal(size=(n_items, d)).astype(np.float32)

    centers = pairs[:, 0]
    contexts = pairs[:, 1]

    for t in range(steps):
        idx = rng.integers(0, len(pairs))
        c = int(centers[idx]); o = int(contexts[idx])

        neg = rng.integers(0, n_items, size=neg_k)

        # logistic loss gradients (sketched; production uses torch/jax)
        vc = W_in[c]
        vo = W_out[o]
        score_pos = 1 / (1 + np.exp(-np.dot(vc, vo)))
        grad = (score_pos - 1.0)

        W_in[c]  -= 0.05 * grad * vo
        W_out[o] -= 0.05 * grad * vc

        for n in neg:
            vn = W_out[int(n)]
            score_neg = 1 / (1 + np.exp(-np.dot(vc, vn)))
            gradn = score_neg
            W_in[c]      -= 0.05 * gradn * vn
            W_out[int(n)] -= 0.05 * gradn * vc

    return l2_normalize(W_in)
```

This code is intentionally “shape-correct” for replication. In production, use torch/jax, proper batching, and sampling.

### 4) Method C: Graph embeddings (random walks + skip-gram)
Use when co-occurrence is better expressed as a graph (e.g., basket co-purchase).

```python
def build_graph(C: np.ndarray, *, topk: int = 50) -> list[list[int]]:
    nbrs = []
    for i in range(C.shape[0]):
        idx = np.argsort(-C[i])[:topk]
        nbrs.append([int(j) for j in idx if C[i, j] > 0])
    return nbrs

def random_walks(nbrs: list[list[int]], *, n_walks: int = 10, walk_len: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    walks = []
    for start in range(len(nbrs)):
        for _ in range(n_walks):
            cur = start
            seq = [cur]
            for _ in range(walk_len - 1):
                if not nbrs[cur]:
                    break
                cur = int(rng.choice(nbrs[cur]))
                seq.append(cur)
            walks.append(seq)
    return walks
```

Feed walks into the same Item2Vec trainer as sequences.

### 5) Method D: Fuse with content embeddings
This is often the “real system” move: behavior for preference geometry, content for cold start.

```python
def fuse_behavior_content(E_beh: np.ndarray, E_cont: np.ndarray, *, w_beh: float = 0.7) -> np.ndarray:
    if E_beh.shape != E_cont.shape:
        raise ValueError("shape mismatch for fusion")
    E = w_beh * E_beh + (1.0 - w_beh) * E_cont
    return l2_normalize(E.astype(np.float32))
```

## Guardrails and failure modes
- **Popularity domination:** PPMI wrong or window too wide; embeddings collapse to popularity.
- **Sessionization bugs:** cross-user leakage creates false co-occurrence.
- **Cold-start holes:** behavior-only methods produce no vectors for new items; fuse with content or backfill.
- **Index skew:** embeddings published but ANN not rebuilt; serving uses stale neighbors.
- **Silent regressions:** method changes produce plausible vectors; require offline sanity checks (NN inspection, stability diffs).

## Known limitations
- Behavior embeddings reflect exposure bias.
- Methods assume stationarity; drift requires retraining and versioning.
- Fusion weights are heuristic unless learned; keep them stable and versioned.
