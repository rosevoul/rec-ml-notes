# GenAI Reranker (Post-Ranking)

## TLDR
This component applies an LLM as a constrained semantic judge to adjust the ordering of an already-scored candidate list.
For example, it fixes obvious semantic mistakes in the top 20 to 50 results. 
It runs after the primary ranker and must be safe to bypass at any time. 
It does not add, remove, or filter items.

**Methods**
- **Judge prompt:** ask for item_ids only with no explanations.
- **Deterministic generation:** schema enforcement and zero temperature.
- **Validation and bypass:** ignore invalid outputs and keep the original order.
- **Bounded influence:** blend LLM order with the primary ranking.
- **Guardrails and metrics:** swap rate, latency, timeouts, and allowlisting.

---

## When this component is needed
- Small candidate sets, for example 50 or fewer, where ordering errors are user visible.
- Queries with semantic nuance or constraints not captured in learned features.
- Surfaces with explicit latency and cost budgets.
- A refinement step that must be safe to bypass at any time.

## How this component fits in the retrieval flow

```
Candidates + scores
        ↓
  GenAI reranker
        ↓
 Final ordering
        ↓
 UI / policy
```

Consumes ranked candidates and query context. Emits a reordered list only. Does not add, remove, or filter items.

## What goes in and what comes out

Input:
```json
{
  "query": "ergonomic office chair",
  "candidates": [
    {"item_id": 712, "title": "Mesh Office Chair", "score": 1.82},
    {"item_id": 45,  "title": "Executive Leather Chair", "score": 1.75},
    {"item_id": 98,  "title": "Drafting Chair", "score": 1.61}
  ]
}
```

Output:
```json
{
  "final_rank": [45, 712, 98]
}
```

For this component, the output is a reordered list of the same item_ids with bounded changes.

## How reranking works

### 1) Asking for a strict ranked list
**Method:** Judge prompt. 
Use a strict prompt so output is only item_ids with no explanations.

```python
SYSTEM_PROMPT = """
You are a ranking function.
You output only item_ids in ranked order.
No explanations.
"""

def build_prompt(query, candidates):
    items = [
        {"id": c["item_id"], "title": c["title"]}
        for c in candidates
    ]
    return {
        "system": SYSTEM_PROMPT,
        "user": {
            "query": query,
            "items": items
        }
    }
```

Notes
- Only titles are provided to reduce leakage from price, popularity, or model scores.
- The primary ranker remains the main source of relevance signals.

### 2) Getting deterministic output with schema checks
**Method:** Deterministic generation with schema enforcement. 
Use zero temperature and hard schema validation.

```python
def llm_rank(prompt):
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.0,
        max_output_tokens=50,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ranking",
                "schema": {
                    "type": "object",
                    "properties": {
                        "order": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["order"]
                }
            }
        }
    )
    return resp.output_parsed["order"]
```

### 3) What to do when the output is wrong
**Method:** Validation and fail-open bypass. 
Use to ignore invalid outputs and keep the original ranking.

```python
def validate(order, candidates):
    ids = {c["item_id"] for c in candidates}
    return (
        len(order) == len(candidates) and
        set(order) == ids
    )
```

Invalid outputs are logged and ignored. The system falls back to the original ranking.

### 4) Limiting how much the LLM can change
**Method:** Bounded influence via blending. 
Use a small alpha so the primary ranker stays dominant.

```python
def blend(primary_rank, llm_rank, alpha=0.2):
    score = {}
    for i, item in enumerate(primary_rank):
        score[item] = 1.0 - alpha * i
    for i, item in enumerate(llm_rank):
        score[item] = score.get(item, 0) + alpha * (1.0 - i)
    return sorted(score, key=score.get, reverse=True)
```

Notes
- Alpha is tuned offline and fixed per surface.
- This limits blast radius from prompt or model drift.

## What to measure to keep this safe

### Swap rate
**Method:** Swap rate guardrail. 
Use to detect over-intervention.

```python
def swap_rate(before, after):
    swaps = sum(1 for i in range(len(before)) if before[i] != after[i])
    return swaps / len(before)
```

### Incremental lift in experiments
**Method:** Experiment-only evaluation. 
Disable if there is no sustained lift.

```python
delta_ctr = ctr_llm - ctr_baseline
if delta_ctr < 0:
    disable("llm_reranker")
```

### Latency and timeouts
**Method:** Latency and timeout bypass. 
Bypass when p95 is too high or timeouts rise.

```python
if llm_latency_p95 > 150:
    bypass("llm_reranker")
```

## What can go wrong and how to notice it
- Schema failures that produce invalid outputs.
- Prompt edits that change output distribution.
- High swap rates that create visible churn.
- Latency creep that violates budgets.
- Cost growth that scales linearly with QPS.


## What this approach does not handle well
- Recall. It only reorders candidates.
- Interpretability of individual swaps.
- Cost at high QPS.
