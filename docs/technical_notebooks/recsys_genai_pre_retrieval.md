# GenAI Query Expansion (Pre-Retrieval)

## TLDR
This component expands a raw text query into a small set of safe rewrites for retrieval. 
For example, it turns "office chair" into 2 extra queries that widen recall. 
It runs before embedding and ANN retrieval and must fail open. 

**Methods**
- **Query normalization:** clean input so caching and prompting are stable.
- **Prompted rewrite:** produce short query variants with deterministic generation.
- **Post-processing:** dedupe, cap, and keep the original query first.
- **Fail-open bypass:** return the original query on any error.
- **Caching:** cache variants by normalized query, locale, surface, and version.
- **Wiring into retrieval:** embed each variant and merge candidates.

---

## What this component builds
- Produces 1 to 3 query variants for the existing embedding and retrieval stack.
- Keeps the original query as the anchor and always includes it first.
- Adds no new retrieval logic. It only changes the text fed into embedding.

## When this component is needed
- Query is short, generic, or ambiguous, for example 1 to 2 tokens.
- No strong user profile signal exists for the request.
- Retrieval recall is the bottleneck, not ranking.
- The surface can tolerate extra latency, or expansions can be cached.

## How this component fits in the retrieval flow

```
Raw query
   ↓
LLM expansion (optional) ──┐
   ↓                      │
Query normalization        │
   ↓                      │
Embed each query variant   │
   ↓                      │
ANN retrieval per variant  │
   ↓                      │
Merge + dedupe candidates ─┘
   ↓
Ranker
```

The rest of the pipeline stays unchanged. If this component is disabled, the original query flows through.

## Inputs & Outputs

Input:
```json
{ "query": "office chair", "locale": "en_US", "surface": "search" }
```

Output:
```json
{
  "queries": [
    "office chair",
    "ergonomic office chair",
    "adjustable desk chair lumbar support"
  ],
  "expansion_version": "qe_v3"
}
```

For this component, the output is a small list of short query strings with the original query preserved first.

## How query expansion works

### 1) Cleaning queries before calling the model
**Method:** Query normalization. 
Use to avoid garbage input, for example repeated spaces or very long text.

```python
import re
from dataclasses import dataclass

_ws = re.compile(r"\s+")

def normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = _ws.sub(" ", q)
    # keep punctuation minimal; avoid removing meaning (e.g., "c++", "4k")
    return q[:256]  # hard cap; prevents prompt bloat
```

### 2) Setting config and versioning
**Method:** Request contract and versioning. 
Use versioning because prompts drift and caches must stay consistent.

```python
@dataclass(frozen=True)
class QEConfig:
    expansion_version: str = "qe_v3"
    max_variants: int = 3
    max_tokens_out: int = 80
    temperature: float = 0.0
    timeout_ms: int = 120
    model: str = "gpt-4.1-mini"

CFG = QEConfig()
```

### 3) Asking for rewrites with strict rules
**Method:** Prompted rewrite. 
Use a boring prompt so outputs are repeatable.

```python
SYSTEM_PROMPT = """
You rewrite search queries.
Return 2 concise alternatives to the input query.
Rules:
- No brands.
- No explanations.
- Output one query per line.
- Keep each line under 10 words.
"""
```

### 4) Calling the model and cleaning its output
**Method:** LLM call with output hygiene. 
Treat the model like an unreliable dependency.

```python
def llm_expand(raw_query: str, *, cfg: QEConfig = CFG) -> list[str]:
    resp = client.responses.create(
        model=cfg.model,
        input={"system": SYSTEM_PROMPT, "user": raw_query},
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_tokens_out,
        # In production: enforce request timeout at the client/transport layer.
    )
    text = (resp.output_text or "").strip()
    if not text:
        return []
    lines = [normalize_query(x) for x in text.splitlines() if x.strip()]
    return lines
```

### 5) Keeping the original query first and bounding cost
**Method:** Post-processing. 
Use to dedupe, cap variants, and preserve the anchor query.

```python
def dedupe_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def build_variants(query: str, *, cfg: QEConfig = CFG) -> list[str]:
    q0 = normalize_query(query)
    # Anchor always first
    variants = [q0]

    # Expansion is additive
    expansions = llm_expand(q0, cfg=cfg)
    variants.extend(expansions)

    variants = dedupe_keep_order(variants)

    # Hard cap to bound cost downstream (embed + ANN + ranking)
    variants = variants[:cfg.max_variants]

    # Invariant: anchor must exist
    assert variants and variants[0] == q0, "Anchor query must be preserved"
    return variants
```

### 6) What to do when the model fails
**Method:** Fail-open bypass. 
Use to return baseline retrieval instead of an error.

```python
def safe_build_variants(query: str, *, cfg: QEConfig = CFG) -> list[str]:
    q0 = normalize_query(query)
    try:
        return build_variants(q0, cfg=cfg)
    except Exception as e:
        log.warning("qe_bypass", extra={"err": str(e), "expansion_version": cfg.expansion_version})
        return [q0]
```

### 7) Caching variants so this is usable at scale
**Method:** Caching. 
Use to make the expensive part optional.

```python
def cache_key(query: str, locale: str, surface: str, version: str) -> str:
    return f"qe:{version}:{surface}:{locale}:{query}"

def get_variants(query: str, locale: str, surface: str, *, cfg: QEConfig = CFG) -> list[str]:
    q0 = normalize_query(query)
    key = cache_key(q0, locale, surface, cfg.expansion_version)
    cached = redis.get_json(key)
    if cached:
        return cached["queries"]

    variants = safe_build_variants(q0, cfg=cfg)
    redis.set_json(key, {"queries": variants}, ttl_s=7 * 24 * 3600)
    return variants
```

### 8) Wiring into retrieval
**Method:** Expand, embed, retrieve, merge. 
Use to run one retrieval per variant and merge candidates.

```python
def pre_retrieval_candidates(query: str, locale: str, surface: str) -> list[int]:
    queries = get_variants(query, locale, surface, cfg=CFG)

    all_ids: list[int] = []
    for q in queries:
        vec = embed_text(q)                 # existing embedding service
        ids, _scores = ann_retrieve(vec)    # existing ANN service
        all_ids.extend(ids)

    return dedupe_candidates(all_ids)
```

## What downstream receives
Downstream still receives candidates for ranking. The difference is candidate coverage.

```json
{
  "candidates": [712, 45, 98, 311, 901],
  "sources": {
    "query": "office chair",
    "variants": ["office chair", "ergonomic office chair", "adjustable desk chair lumbar support"],
    "expansion_version": "qe_v3"
  }
}
```

## What can go wrong and how to notice it
- Semantic drift where expansions do not match inventory.
- Over-expansion where too many variants increase cost.
- Cache miss storms when normalization changes and hit rate collapses.
- Prompt drift when small edits change the output distribution.
- Latency creep when model p95 rises.
- Locale leakage where expansions use the wrong language.

## What this approach does not handle well
- Inventory grounding. Expansions can be reasonable but irrelevant.
- Long queries. Long queries often need constraint preservation instead of expansion.
- Operational overhead. Cache and versioning add surface area.
