```md
# Behavioral Item Embeddings (Co-occurrence, Item2Vec, Graph, Fusion)


## TLDR
This component builds item embeddings from user interaction data for retrieval.  
It runs offline and the output is a versioned embedding table used by ANN retrieval.

**Methods**
- **Co-occurrence + SVD:** deterministic offline embeddings from item co-occurrence.
- **Item2Vec:** sequence-based embeddings that capture order and local context.
- **Graph embeddings:** random-walk embeddings from co-purchase or co-cart graphs.
- **Behavior + content fusion:** combines behavior and content embeddings into a new version.


## What this component does and does not do
- Converts sessionized interaction data into fixed-size item embeddings.
- Supports multiple embedding methods that all produce the same output format.
- Produces a single embedding table keyed by item_id and version.
- Does not build user embeddings. That is handled by a separate component.
- Does not rank items or apply policy.

## When this component is used
- There is enough behavioral data to learn item relationships.
- Content or metadata alone is not enough for good retrieval.
- Retrieval quality depends on item-to-item similarity.
- Offline training jobs can run and publish new embedding versions.

## Integration points

```

Interaction logs → Sessionize → Behavioral embedding job → Embedding store → ANN index → Retrieval
└─ optional fusion with content embeddings → writes a new embedding version

````

## Example inputs and outputs

Input (sessionized):
```json
{ "session_id": "abc", "items": [12, 45, 98, 45, 311] }
````

Output:

```json
{ "item_id": 45, "embedding": [0.013, -0.091, ...], "version": "beh_v9" }
```

## Core implementation


### Method A: Co-occurrence → PPMI → SVD (baseline)

Deterministic offline method that produces stable embeddings, for example from items frequently viewed together.

```python
def build_cooccurrence(
    sessions: list[list[int]],
    n_items: int,
    window: int = 5
) -> np.ndarray:
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

### Method B: Item2Vec (skip-gram with negative sampling)

Use this method when order and local context matter, for example when users view a phone and then a phone case in the same session.


```python
def make_skipgram_pairs(
    sessions: list[list[int]],
    window: int = 5
) -> np.ndarray:
    pairs = []
    for seq in sessions:
        for i, center in enumerate(seq):
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, seq[j]))
    return np.asarray(pairs, dtype=np.int64)

def train_item2vec(
    pairs: np.ndarray,
    n_items: int,
    dim: int = 128,
    neg_k: int = 10,
    steps: int = 50_000
) -> np.ndarray:
    rng = np.random.default_rng(0)
    W_in = 0.01 * rng.normal(size=(n_items, dim)).astype(np.float32)
    W_out = 0.01 * rng.normal(size=(n_items, dim)).astype(np.float32)

    centers = pairs[:, 0]
    contexts = pairs[:, 1]

    for _ in range(steps):
        idx = rng.integers(0, len(pairs))
        c = int(centers[idx])
        o = int(contexts[idx])

        neg = rng.integers(0, n_items, size=neg_k)

        vc = W_in[c]
        vo = W_out[o]

        score_pos = 1 / (1 + np.exp(-np.dot(vc, vo)))
        grad = score_pos - 1.0

        W_in[c] -= 0.05 * grad * vo
        W_out[o] -= 0.05 * grad * vc

        for n in neg:
            vn = W_out[int(n)]
            score_neg = 1 / (1 + np.exp(-np.dot(vc, vn)))
            gradn = score_neg
            W_in[c] -= 0.05 * gradn * vn
            W_out[int(n)] -= 0.05 * gradn * vc

    return l2_normalize(W_in)
```

### Method C: Graph embeddings (random walks + skip-gram)

Use when item relationships are better represented as a graph, for example co-purchase or co-cart data.

```python
def build_graph(
    C: np.ndarray,
    topk: int = 50
) -> list[list[int]]:
    neighbors = []
    for i in range(C.shape[0]):
        idx = np.argsort(-C[i])[:topk]
        neighbors.append([int(j) for j in idx if C[i, j] > 0])
    return neighbors

def random_walks(
    neighbors: list[list[int]],
    n_walks: int = 10,
    walk_len: int = 20,
    seed: int = 0
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    walks = []
    for start in range(len(neighbors)):
        for _ in range(n_walks):
            cur = start
            seq = [cur]
            for _ in range(walk_len - 1):
                if not neighbors[cur]:
                    break
                cur = int(rng.choice(neighbors[cur]))
                seq.append(cur)
            walks.append(seq)
    return walks
```

The resulting walks are passed into the same Item2Vec training logic.

### Method D: Fuse with content embeddings

Fusion controls what retrieval sees. Behavior embeddings capture preferences, while content embeddings help with new or unseen items.

```python
def fuse_behavior_content(
    behavior_embeddings: np.ndarray,
    content_embeddings: np.ndarray,
    behavior_weight: float = 0.7
) -> np.ndarray:
    if behavior_embeddings.shape != content_embeddings.shape:
        raise ValueError("embedding shape mismatch")
    fused = (
        behavior_weight * behavior_embeddings
        + (1.0 - behavior_weight) * content_embeddings
    )
    return l2_normalize(fused.astype(np.float32))
```


### Common helpers

For this component, the output is processed into fixed-size vectors, L2-normalized, and written as a new versioned table.

```python
import numpy as np

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def write_item_table(
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    version: str
) -> None:
    for i, item_id in enumerate(item_ids):
        embedding_store.put(
            item_id=int(item_id),
            vec=embeddings[i],
            version=version
        )
```

## Guardrails and failure modes

* **Popularity domination:** PPMI misconfigured or window too wide, embeddings collapse toward popularity.
* **Sessionization errors:** cross-user leakage creates false co-occurrence.
* **Cold-start gaps:** behavior-only methods produce no vectors for new items. Fusion or backfill is required.
* **Index skew:** embeddings updated without rebuilding ANN, serving stale neighbors.
* **Silent regressions:** changes produce plausible vectors. Require offline checks such as nearest-neighbor inspection and stability diffs.

This component owns embedding quality and stability. If retrieval quality drops because of bad or stale embeddings, this component is responsible.

## Known limitations

* Behavior embeddings encode exposure bias.
* Data is assumed stable between training runs. If data changes, a new version must be trained and published.
* Fusion weights are heuristic unless learned. A good practice is to keep them stable and versioned.
