# Spec 051: Advanced Confidence Scoring Functions from fd-shifts

**Status**: Implemented (2026-01-03)
**Priority**: Medium (enriches confidence signal library)
**Depends on**: None (standalone)
**Estimated effort**: Low-Medium
**Research basis**: [fd-shifts (ICLR 2023, NeurIPS 2024)](https://github.com/IML-DKFZ/fd-shifts)

## 0. Problem Statement

The [fd-shifts benchmark](https://github.com/IML-DKFZ/fd-shifts) implements **13+ Confidence Scoring Functions (CSFs)** that are validated across multiple datasets. Our current implementation has only 5 CSFs.

Some fd-shifts CSFs are directly applicable to our LLM-based system:
- **Maximum Softmax Probability (MSP)** → Token-level confidence
- **Predictive Entropy (PE)** → Token-level uncertainty
- **Energy Score** → Alternative to softmax
- **External Confidence** → Our retrieval signals

Others require model modifications not available via Ollama:
- **Monte Carlo Dropout (MCD)** → Requires dropout at inference
- **Deep Ensembles** → Requires multiple models

This spec focuses on **portable CSFs** that can enhance our confidence signal library.

## 1. Goals / Non-Goals

### 1.1 Goals

- Port applicable CSFs from fd-shifts to our codebase
- Add **token-level confidence** signals (requires Ollama logprobs)
- Add **secondary combination** CSFs (average, product of signals)
- Provide consistent API for registering and using CSFs
- Enable ablation across CSF variants

### 1.2 Non-Goals

- MCD or ensemble methods (not available via Ollama)
- Training custom confidence networks (e.g., ConfidNet)
- Mahalanobis distance or other representation-space methods

## 2. CSF Inventory

### 2.1 Currently Implemented

| CSF | Signal | Source |
|-----|--------|--------|
| `llm` | `llm_evidence_count` | Spec 046 |
| `total_evidence` | Legacy alias for `llm` | Spec 047 |
| `retrieval_similarity_mean` | Mean similarity of retrieved refs | Spec 046 |
| `retrieval_similarity_max` | Max similarity of retrieved refs | Spec 046 |
| `hybrid_evidence_similarity` | 0.5 * e + 0.5 * s | Spec 046 |

### 2.2 Proposed Additions (This Spec)

| CSF | Signal | Source | Requires |
|-----|--------|--------|----------|
| `token_msp` | Max softmax probability of predicted tokens | fd-shifts | Ollama logprobs |
| `token_pe` | Predictive entropy of predicted tokens | fd-shifts | Ollama logprobs |
| `token_energy` | Energy score (logsumexp of logits) | fd-shifts | Ollama logprobs |
| `secondary_average` | Average of two CSFs | fd-shifts | Two base CSFs |
| `secondary_product` | Product of two CSFs | fd-shifts | Two base CSFs |

### 2.3 fd-shifts CSFs NOT Portable

| CSF | Why Not Portable |
|-----|------------------|
| `mcd_*` | Requires dropout at inference |
| `maha` | Requires hidden representations |
| `vim` | Requires hidden representations |
| `dknn` | Requires hidden representations |
| `tcp` | Requires trained confidence head |
| `dg` | Requires deep generative model |
| `devries` | Requires trained confidence head |

## 3. Proposed Solution

### 3.1 Token-Level CSFs via Ollama Logprobs

Ollama supports `logprobs` in the response when enabled (Requires Ollama >= 0.12.11).
The response JSON structure contains a `logprobs` field which is a list of objects.

```python
# Ollama API call with logprobs
response = ollama.chat(
    model="gemma3:27b-it-qat",
    messages=[...],
    options={"logprobs": True, "top_logprobs": 5}
)

# Response structure (verified):
# {
#   "message": { ... },
#   "logprobs": [
#     {
#       "token": "The",
#       "logprob": -0.001,
#       "bytes": [84, 104, 101],
#       "top_logprobs": [ ... ]
#     },
#     ...
#   ]
# }
```

**Token MSP (Maximum Softmax Probability):**
```python
def compute_token_msp(logprobs: list[dict]) -> float:
    """Mean of max softmax probability across tokens."""
    probs = [np.exp(lp["logprob"]) for lp in logprobs]
    return float(np.mean(probs))
```

**Predictive Entropy:**
```python
def compute_token_pe(logprobs: list[dict]) -> float:
    """Mean predictive entropy across tokens (lower = more confident)."""
    entropies = []
    for lp in logprobs:
        # Entropy from top_logprobs distribution
        probs = np.array([np.exp(t["logprob"]) for t in lp["top_logprobs"]])
        probs = probs / probs.sum()  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
    return float(np.mean(entropies))
```

**Energy Score:**
```python
def compute_token_energy(logprobs: list[dict]) -> float:
    """Mean energy score (logsumexp of logits)."""
    energies = []
    for lp in logprobs:
        logits = [t["logprob"] for t in lp["top_logprobs"]]
        energy = scipy.special.logsumexp(logits)
        energies.append(energy)
    return float(np.mean(energies))
```

### 3.2 CSF Registry

Port the fd-shifts pattern for registering CSFs:

```python
# src/ai_psychiatrist/confidence/csf_registry.py

_csf_funcs: dict[str, Callable] = {}

def register_csf(name: str) -> Callable:
    """Decorator to register a CSF."""
    def wrapper(func: Callable) -> Callable:
        _csf_funcs[name] = func
        return func
    return wrapper

def get_csf(name: str) -> Callable:
    """Get a registered CSF by name."""
    if name not in _csf_funcs:
        raise ValueError(f"Unknown CSF: {name}. Available: {list(_csf_funcs.keys())}")
    return _csf_funcs[name]

@register_csf("llm")
def csf_llm(item_signals: dict) -> float:
    return float(item_signals.get("llm_evidence_count", 0))

@register_csf("token_msp")
def csf_token_msp(item_signals: dict) -> float:
    value = item_signals.get("token_msp")
    if value is None:
        raise ValueError("token_msp not available in item_signals")
    return float(value)

@register_csf("retrieval_similarity_mean")
def csf_retrieval_similarity_mean(item_signals: dict) -> float:
    return float(item_signals.get("retrieval_similarity_mean", 0.0))
```

### 3.3 Secondary Combinations

Port fd-shifts' secondary combination pattern:

```python
# src/ai_psychiatrist/confidence/csf_registry.py

_combine_opts = {
    "average": lambda x, y: (x + y) / 2,
    "product": lambda x, y: x * y,
}

def create_secondary_csf(csf1: str, csf2: str, combine: str) -> Callable:
    """Create a secondary CSF that combines two base CSFs."""
    if combine not in _combine_opts:
        raise ValueError(f"Unknown combine method: {combine}")

    func1 = get_csf(csf1)
    func2 = get_csf(csf2)
    combine_func = _combine_opts[combine]

    def secondary(item_signals: dict) -> float:
        return combine_func(func1(item_signals), func2(item_signals))

    return secondary

# Usage:
# csf = create_secondary_csf("token_msp", "retrieval_similarity_mean", "average")
# confidence = csf(item_signals)
```

### 3.4 Run Artifact Extension

Add token-level signals to `item_signals`:

```json
{
  "item_signals": {
    "Sleep": {
      "llm_evidence_count": 2,
      "retrieval_similarity_mean": 0.82,
      "verbalized_confidence": 4,
      "token_msp": 0.91,
      "token_pe": 0.23,
      "token_energy": 2.1
    }
  }
}
```

### 3.5 Evaluation Script Updates

Update `scripts/evaluate_selective_prediction.py`:

```python
CONFIDENCE_VARIANTS = {
    # Existing...

    # NEW (Spec 051)
    "token_msp",
    "token_pe",
    "token_energy",
    "secondary:llm+token_msp:average",
    "secondary:retrieval_similarity_mean+token_msp:product",
}

# Secondary CSF parsing
def parse_confidence_variant(variant: str):
    if variant.startswith("secondary:"):
        # Format: secondary:csf1+csf2:method
        parts = variant[10:].split(":")
        csfs = parts[0].split("+")
        method = parts[1]
        return create_secondary_csf(csfs[0], csfs[1], method)
    return get_csf(variant)
```

## 4. Implementation Plan

### Phase 1: CSF Registry

1. Create `src/ai_psychiatrist/confidence/__init__.py`
2. Create `src/ai_psychiatrist/confidence/csf_registry.py`
3. Port existing CSFs to registry pattern
4. Implement secondary combinations

### Phase 2: Token-Level Signals

5. Update `OllamaClient` to request logprobs
6. Implement `compute_token_msp`, `compute_token_pe`, `compute_token_energy`
7. Persist token signals in `ItemAssessment`
8. Export to run artifacts

### Phase 3: Evaluation Integration

9. Update `evaluate_selective_prediction.py` to use CSF registry
10. Add secondary CSF parsing
11. Document available CSFs

## 5. Test Plan

### 5.1 Unit Tests

- `test_csf_registry`: Register and retrieve CSFs
- `test_secondary_csf`: Average, product combinations
- `test_token_msp`: Correct computation from logprobs
- `test_token_pe`: Entropy calculation

### 5.2 Integration Tests

- Mock Ollama response with logprobs
- Verify end-to-end token signal extraction

## 6. Expected Outcomes

| CSF | Expected Correlation with Correctness |
|-----|---------------------------------------|
| `token_msp` | Moderate-High (0.3-0.5) |
| `token_pe` | Moderate (0.2-0.4) |
| `secondary:llm+token_msp:average` | High (0.4-0.6) |

Based on fd-shifts: "softmax response baseline is overall best performing" (their MSP finding).

## 7. Acceptance Criteria

- [ ] CSF registry with `register_csf` and `get_csf`
- [ ] Token-level signals extracted from Ollama logprobs
- [ ] Token signals persisted in run artifacts
- [ ] Secondary CSF combinations work
- [ ] `evaluate_selective_prediction.py` supports new variants
- [ ] Documentation in `docs/statistics/metrics-and-evaluation.md`
- [ ] Tests pass: `make ci`

## 8. File Changes

### New Files

- `src/ai_psychiatrist/confidence/__init__.py`
- `src/ai_psychiatrist/confidence/csf_registry.py`
- `src/ai_psychiatrist/confidence/token_csfs.py`
- `tests/unit/confidence/test_csf_registry.py`

### Modified Files

- `src/ai_psychiatrist/infrastructure/ollama_client.py` (add logprobs)
- `src/ai_psychiatrist/agents/quantitative.py` (extract token signals)
- `scripts/evaluate_selective_prediction.py` (use CSF registry)

## 9. References

- [fd-shifts confid_scores.py](../../_reference/fd-shifts/fd_shifts/analysis/confid_scores.py)
- [fd-shifts ICLR 2023](https://openreview.net/forum?id=YnkGMIh0gvX)
- [fd-shifts NeurIPS 2024](https://arxiv.org/abs/2407.01032)
- [Ollama API - logprobs](https://github.com/ollama/ollama/blob/main/docs/api.md)
