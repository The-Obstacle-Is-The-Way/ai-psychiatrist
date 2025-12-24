# AI-Psychiatrist: Paper vs. Code Discrepancy Analysis

**Analysis Date:** December 24, 2025
**Repo Commit Audited:** `a966ce5878b9c4cbefc05fa97596fd0afa4311c7`
**Paper Source:** `_literature/ai_psychiatrist.md`
**Paper:** AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews
**Paper-Reported MAE:** 0.619 (few-shot) vs 0.796 (zero-shot) (`_literature/ai_psychiatrist.md:9`, `_literature/ai_psychiatrist.md:129`)
**Reproduction Attempt MAE (external, not in repo):** 0.778
**Status:** Multiple discrepancies identified requiring author clarification

---

## Executive Summary

This document consolidates findings comparing the published paper claims against the released codebase. We identified **at least 8 critical discrepancies** that affect reproducibility, with the most significant being:

1. **Undocumented keyword backfill** in production agent code (not in experimental notebooks)
2. **Hardware mismatch** (A100 GPUs vs MacBook M3 Pro claims)
3. **Server defaults don't match paper** (llama3 vs Gemma 3 27B)
4. **MAE computation methodology** not clearly specified in paper

---

## Critical Finding 1: Keyword Backfill Mechanism

### Severity: CRITICAL

### Paper Claim
> "During evidence retrieval, Gemma 3 27B was provided with information about the PHQ-8, a given transcript, and instructions to retrieve relevant evidence associated with each individual PHQ-8 question. If no relevant evidence was found for a given PHQ-8 item, the model produced no output." (`_literature/ai_psychiatrist.md:73`)

### Code Reality

**Location:** `agents/quantitative_assessor_f.py:29-38`, `agents/quantitative_assessor_f.py:84-102`, `agents/quantitative_assessor_f.py:478`

The released few-shot agent contains a keyword-based evidence augmentation step (`DOMAIN_KEYWORDS` + `_keyword_backfill`) that runs **unconditionally** inside `QuantitativeAssessor.extract_evidence()`.

**1) Keyword dictionary** (`agents/quantitative_assessor_f.py:29-38`):
```python
DOMAIN_KEYWORDS = {
    "PHQ8_NoInterest": ["can't be bothered", "no interest", "nothing really", "not enjoy", "no pleasure", "what's the point", "can't be bothered", "cant be bothered"],
    "PHQ8_Depressed": ["fed up", "miserable", "depressed", "very black", "hopeless", "low"],
    "PHQ8_Sleep": ["sleep", "fall asleep", "wake up", "insomnia", "clock", "tired in the morning"],
    "PHQ8_Tired": ["exhausted", "tired", "little energy", "fatigue", "no energy"],
    "PHQ8_Appetite": ["appetite", "weight", "lost weight", "eat", "eating", "don't bother", "dont bother", "looser"],
    "PHQ8_Failure": ["useless", "failure", "bad about myself", "burden"],
    "PHQ8_Concentrating": ["concentrat", "memory", "forgot", "thinking of something else", "focus"],
    "PHQ8_Moving": ["moving slowly", "restless", "fidget", "speaking slowly", "psychomotor"]
}
```

**2) Backfill function** (`agents/quantitative_assessor_f.py:84-102`):
```python
def _keyword_backfill(transcript: str, current: Dict[str, List[str]], per_item_cap: int = 3) -> Dict[str, List[str]]:
    sents = _sentences(transcript.lower())
    orig_sents = _sentences(transcript)
    out = {k: list(v) for k, v in current.items()}
    for key, kws in DOMAIN_KEYWORDS.items():
        need = max(0, per_item_cap - len(out.get(key, [])))
        if need == 0:
            continue
        hits = []
        for idx, s in enumerate(sents):
            if any(kw in s for kw in kws):
                hits.append(orig_sents[idx].strip())
            if len(hits) >= need:
                break
        if hits:
            seen = set(out.get(key, []))
            merged = out.get(key, []) + [h for h in hits if h not in seen]
            out[key] = merged[:per_item_cap]
    return out
```

**3) Unconditional call in evidence extraction** (`agents/quantitative_assessor_f.py:478`):
```python
        enriched = _keyword_backfill(transcript, out, per_item_cap=3)
```

**What `_keyword_backfill()` does (verified):**
- Splits transcript into "sentences" by punctuation and newlines (`agents/quantitative_assessor_f.py:80-82`)
- Lowercases for matching but returns original-cased sentence text (`agents/quantitative_assessor_f.py:85-86`)
- Per PHQ-8 item, fills up to `per_item_cap` evidence quotes using substring matches (`any(kw in s ...)`) (`agents/quantitative_assessor_f.py:89-97`)
- Deduplicates against existing evidence and caps each list (`agents/quantitative_assessor_f.py:99-101`)

**Normal execution path (few-shot server mode):**
- `server.py:23` instantiates `QuantitativeAssessorF()` with defaults
- `server.py:46` calls `.assess(...)`
- `agents/quantitative_assessor_f.py:534` calls `.extract_evidence(...)`
- `agents/quantitative_assessor_f.py:478` calls `_keyword_backfill(...)` (no feature flag / config gate)

This differs from the paper description of evidence retrieval being model-only with "no output" when no evidence is found.

### Git History Evidence

The keyword backfill was introduced on October 3, 2025:

| Commit | Message | Date | Changes |
|--------|---------|------|---------|
| `37747eb` | "few shot code done" | Oct 3, 2025 00:34:42 +0330 | First introduces `DOMAIN_KEYWORDS` and `_keyword_backfill()` (343 lines added) |
| `7414754` | "few shot code done" | Oct 3, 2025 01:00:37 +0330 | Refactors the code (304 lines changed) |
| `dddbf24` | "done" | Oct 3, 2025 01:30:52 +0330 | Renames `quantitative_assessor.py` → `quantitative_assessor_f.py` |

**Note:** Git blame attributes lines 29-38 and 84-102 to commit `37747eb`, not `7414754` as previously stated.

### Cross-Verification
- Example repo output contains August 2025 timestamps: `analysis_output/quan_gemma_few_shot/TEST_analysis_output/chunk_8_step_2_examples_2_embedding_results_analysis.jsonl:1`
- The notebooks/scripts in `quantitative_assessment/` contain no matches for `DOMAIN_KEYWORDS`/`_keyword_backfill`/`backfill` (e.g., `rg -n "DOMAIN_KEYWORDS|_keyword_backfill|keyword_backfill|backfill" quantitative_assessment -S` returns no matches)

### Verdict: Was Keyword Backfill Used for MAE 0.619?

| Evidence | Conclusion |
|----------|------------|
| `agents/quantitative_assessor_f.py` applies keyword backfill | Keyword heuristic is active in the released few-shot agent |
| Repo analysis outputs include August 2025 timestamps | Outputs *as stored* predate the Oct 2025 commit that introduced backfill into the agent file |
| Paper method description does not mention any keyword-based evidence augmentation | Keyword backfill is undocumented in the paper method section (`_literature/ai_psychiatrist.md:73`) |

**CONCLUSION: UNCERTAIN (leaning NO)** — The repository suggests the reported MAE 0.619 is tied to the notebook/analysis-output pipeline, while keyword backfill appears later in the agent code path. The repo does not explicitly record which script+commit produced the paper's MAE, so author confirmation is required.

---

## Critical Finding 2: Server Defaults Mismatch

### Severity: CRITICAL

### Paper Claim
> "We utilized a state-of-the-art open-weight language model, Gemma 3 with 27 billion parameters (Gemma 3 27B) [...]" (`_literature/ai_psychiatrist.md:57`)

### Code Reality

**Location:** `agents/quantitative_assessor_f.py:415-425`, `server.py:23`

```python
class QuantitativeAssessor:
    def __init__(
        self,
        ollama_host: str = "127.0.0.1",
        chat_model: str = "llama3",
        emb_model: str  = "dengcao/Qwen3-Embedding-8B:Q4_K_M",
        pickle_path: str = "agents/chunk_8_step_2_participant_embedded_transcripts.pkl",
        gt_train_csv: str = "agents/train_split_Depression_AVEC2017.csv",
        gt_dev_csv: str   = "agents/dev_split_Depression_AVEC2017.csv",
        top_k: int = 3,
        dim: Optional[int] = None
    ):
```

The FastAPI server instantiates this class with defaults (no override):

```python
quantitative_assessor_F = QuantitativeAssessorF()  # few-shot
```

**Additional default-model mismatches in the server pipeline:**
- `agents/qualitative_assessor_f.py:5` defaults to `model="llama3"` and `server.py:18` uses it with defaults.
- `agents/meta_reviewer.py:31` hardcodes `model="llama3"` and `server.py:20` uses it.
- `agents/qualitive_evaluator.py:31` defaults to `model="llama3"` and `server.py:19` uses it.
- `agents/quantitative_assessor_z.py:5` defaults to `model="llama3"` and `server.py:24` uses it.

### Impact
Running the repository **server** "as-is" uses **llama3** by default for the quantitative assessor (and other agents), not Gemma 3 27B as described in the paper. Additionally, the server defaults `top_k=3` and `dim=None` differ from the paper's reported optimized values `Nexample=2` and `Ndimension=4096` (`_literature/ai_psychiatrist.md:127`).

---

## Critical Finding 3: Hardware Discrepancy

### Severity: HIGH

### Paper Claim
> "Running the whole pipeline [...] on a MacBook Pro with an Apple M3 Pro processor took approximately one minute to compile a report. This efficiency, achieved without the need for dedicated GPUs or specialized hardware, demonstrates the practicality [...]" (`_literature/ai_psychiatrist.md:185`)

### Code Reality

**Location:** `slurm/job_ollama.sh:1-4`, `slurm/job_ollama.sh:14-27`

```bash
#!/bin/bash
#SBATCH -J ollama
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:2
#SBATCH -A trends53c17
```

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OLLAMA_MODELS=/data/users4/splis/ollama/models/
export OLLAMA_BACKEND=gpu
```

**Location:** `README.md:31-44`
```markdown
## Ollama on TReNDS Cluster
1. Start Ollama by submitting the SLURM job script `job_ollama.sh`
```

### Cross-Reference
Multiple scripts reference cluster nodes:
- Hardcoded node names appear in code/notebooks (examples): `quantitative_assessment/quantitative_analysis.py:11`, `qualitative_assessment/qual_assessment.py:11`, `qualitative_assessment/feedback_loop.py:31`, `quantitative_assessment/embedding_quantitative_analysis.ipynb:47`
- Hardcoded cluster paths like `/data/users*/...` appear throughout scripts and notebooks (e.g., `quantitative_assessment/quantitative_analysis.py:20-21`)

### Impact
The repository includes SLURM scripts and environment variables configured for **A100 GPU-backed Ollama serving**, which conflicts with (or at minimum substantially complicates) the paper's "no dedicated GPUs" deployment claim. The paper should clarify what hardware/environment was used to generate the reported results and runtime.

---

## Critical Finding 4: Model Variant Ambiguity

### Severity: MEDIUM

### Paper Claim
> "Gemma 3 with 27 billion parameters (Gemma 3 27B) [...]" (`_literature/ai_psychiatrist.md:57`)

### Code Reality
Multiple model references across files:

| File | Line | Model String |
|------|------|--------------|
| `quantitative_assessment/quantitative_analysis.py` | 14 | `gemma3-optimized:27b` |
| `quantitative_assessment/basic_quantitative_analysis.ipynb` | 40 | `gemma3-optimized:27b` |
| `quantitative_assessment/embedding_quantitative_analysis.ipynb` | 50 | `gemma3-optimized:27b` |
| `qualitative_assessment/qual_assessment.py` | 13 | `gemma3:27b` |
| `qualitative_assessment/feedback_loop.py` | 33 | `alibayram/medgemma:27b` |
| `agents/quantitative_assessor_f.py` | 419 | `llama3` (default!) |
| `agents/meta_reviewer.py` | 31 | `llama3` (hard-coded) |
| `agents/qualitative_assessor_f.py` | 5 | `llama3` (default) |
| `meta_review/meta_review.py` | 43 | `gemma3-optimized:27b` |
| `assets/ollama_example.py` | 7 | `gemma3-optimized:27b` |

### Open Questions
1. What is "gemma3-optimized:27b"? A quantized GGUF variant?
2. Which specific model checkpoints were used for reported results?
3. Were different models used for different experiments?

---

## Critical Finding 5: Undocumented Hyperparameters

### Severity: HIGH

### Paper Claim
Paper specifies optimized hyperparameters `Nchunk = 8`, `Nexample = 2`, `Ndimension = 4096`, but does not specify sampling parameters such as `temperature`, `top_k`, or `top_p` (`_literature/ai_psychiatrist.md:127`).

### Code Reality

**Few-shot** (`agents/quantitative_assessor_f.py:200`):
```python
"options": {"temperature": 0.2, "top_k": 20, "top_p": 0.8}
```

**Few-shot (batch script)** (`quantitative_assessment/embedding_batch_script.py:374-379`, `quantitative_assessment/embedding_batch_script.py:625-630`):
```python
                "options": {
                    # Fairly deterministic parameters
                    "temperature": 0.2,
                    "top_k": 20,
                    "top_p": 0.8
                }
```

**Zero-shot baseline** (`quantitative_assessment/quantitative_analysis.py:137-139`):
```python
"options": {
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0
}
```

### Impact
The baseline uses fully deterministic settings while few-shot uses stochastic settings. This asymmetry could affect comparison validity and makes reproduction without exact parameters impossible.

---

## Critical Finding 6: MAE Computation Methodology

### Severity: HIGH

### Paper Claim
The paper states that "Model performance was evaluated by computing the mean absolute error (MAE) between predicted and groundtruth PHQ-8 scores." (`_literature/ai_psychiatrist.md:73`) and notes that "Subjects without sufficient evidence [...] were excluded from the assessment." (`_literature/ai_psychiatrist.md:127`), but does not spell out the exact MAE aggregation procedure (per-item vs per-subject, and how `"N/A"`/missing predictions are handled across items).

### Code Reality

**Location:** `visualization/quan_visualization.ipynb:573-585`, `visualization/quan_visualization.ipynb:600-620`, `visualization/quan_visualization.ipynb:660-663`, `visualization/quan_visualization.ipynb:398-401`

The visualization notebook implements MAE as:
1. Exclude `"N/A"` predictions before error calculation
2. Compute per-question MAE as `np.mean(abs(pred - gt))` over available predictions
3. Compute the final aggregate as `np.nanmean(<per-question MAEs>)`

Evidence (as stored in the ipynb JSON):

```json
    "            score = entry[question].get('score', 'N/A')\n",
    "            if score != \"N/A\":\n",
    "                try:\n",
    "                    zero_shot_dict[participant_id] = int(score)\n",
```

```json
    "                zero_shot_errors.append(abs(pred_score - ground_truth_score))\n",
    "    zero_shot_mae = np.mean(zero_shot_errors) if zero_shot_errors else np.nan\n",
    "    few_shot_mae = np.mean(few_shot_errors) if few_shot_errors else np.nan\n",
```

```json
    "print(f\"Gemma 3 - Average Few-shot MAE: {np.nanmean(gemma3_few_shot_maes):.8f}\")\n",
```

The notebook output stored in-repo includes:

```json
      "Gemma 3 - Average Few-shot MAE: 0.61929388\n",
```

### Impact
If your reproduction computed MAE differently (e.g., treating N/A as 0, or using different aggregation), you will get materially different results even with identical predictions.

---

## Critical Finding 7: Notebooks vs. Agent Files Divergence

### Severity: HIGH

### Summary

| Aspect | Notebooks (`quantitative_assessment/`) | Agents (`agents/`) |
|--------|----------------------------------------|-------------------|
| Keyword backfill | NO | YES |
| Default model | `gemma3-optimized:27b` | `llama3` |
| Where used in repo | Offline experiments/analysis scripts | FastAPI server pipeline components |
| Used for paper results? | UNKNOWN | UNKNOWN |
| File rename | N/A | `quantitative_assessor.py` → `quantitative_assessor_f.py` (Oct 3, 2025) |

### Evidence
- Notebooks/scripts in `quantitative_assessment/` contain no matches for `DOMAIN_KEYWORDS`/`_keyword_backfill` (e.g., `rg -n "DOMAIN_KEYWORDS|_keyword_backfill|keyword_backfill|backfill" quantitative_assessment -S` returns no matches)
- The few-shot agent adds keyword backfill inside evidence extraction (`agents/quantitative_assessor_f.py:478`)
- The FastAPI server imports and instantiates the agent implementations from `agents/` (e.g., `server.py:7`, `server.py:23`)

---

## Critical Finding 8: Missing Artifacts

### Severity: MEDIUM

### Required Files Not in Repository

**Location:** `agents/quantitative_assessor_f.py:421-423`
```python
        pickle_path: str = "agents/chunk_8_step_2_participant_embedded_transcripts.pkl",
        gt_train_csv: str = "agents/train_split_Depression_AVEC2017.csv",
        gt_dev_csv: str   = "agents/dev_split_Depression_AVEC2017.csv",
```

**Location:** `agents/interview_simulator.py:11-18`
```python
            or os.getenv("TRANSCRIPT_PATH")
            or "agents/transcript.txt"
```

**Location:** `meta_review/meta_review.py:45-55`
```python
    rootdir = "/data/users4/user/ai-psychiatrist"

    # Load qualitative assessment
    qual_df = pd.read_csv(os.path.join(rootdir, "analysis_output/qual/qual_assessment_GEMMA.csv"))

    # Load quantitative assessment
    quan_list = load_jsonl(os.path.join(rootdir, "analysis_output/quan/chunk_8_step_2_examples_2_embedding_results_analysis_2.jsonl"))
```

**Location:** `quantitative_assessment/quantitative_analysis.py:20-21`
```python
dev_split_phq8 = pd.read_csv(r"/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")
train_split_phq8 = pd.read_csv(r"/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")
```

**Note:** Absolute `/data/users*/...` paths appear widely across scripts and notebooks (e.g., `rg -n "/data/users" -S .` returns many matches), and several required artifacts referenced by default paths (e.g., `agents/transcript.txt`) are not included in this repository snapshot.

---

## Additional Issues Found

### 1) Undocumented model-assisted JSON repair (extra LLM call)

**Location:** `agents/quantitative_assessor_f.py:58-79`, `agents/quantitative_assessor_f.py:501-529`

The few-shot agent may perform an additional LLM call to "repair" malformed JSON outputs during scoring:

```python
def _llm_json_repair(ollama_host: str, model: str, broken: str, timeout: int = 120) -> Optional[Dict[str, Any]]:
    """Ask the model to repair malformed JSON to EXACT keys; return dict or None."""
    repair_system = ""
    repair_user = (
        "You will be given malformed JSON for a PHQ-8 result. "
        "Output ONLY a valid JSON object with these EXACT keys:\n"
        f"{', '.join(PHQ8_KEYS)}\n"
        'Each value must be an object: {"evidence": <string>, "reason": <string>, "score": <int 0-3 or "N/A">}.\n'
        "If something is missing or unclear, fill with "
        '{"evidence":"No relevant evidence found","reason":"Auto-repaired","score":"N/A"}.\n\n'
        "Malformed JSON:\n"
        f"{broken}\n\n"
        "Return only the fixed JSON. No prose, no markdown, no tags."
    )
    try:
        fixed = ollama_chat(ollama_host, model, repair_system, repair_user, timeout=timeout)
        fixed = _strip_json_block(fixed)  # in case it adds fences
        fixed = _tolerant_fixups(fixed)
        return json.loads(fixed)
    except Exception:
        return None
```

```python
                    repaired = _llm_json_repair(self.ollama_host, self.chat_model, between)
                    if repaired is not None:
                        return _validate_and_normalize(repaired)
```

### 2) Inconsistent Ollama endpoints and output types across agents

- Few-shot quantitative uses `/api/chat` and returns a parsed dict (`agents/quantitative_assessor_f.py:192-208`, `agents/quantitative_assessor_f.py:501-529`).
- Zero-shot quantitative uses `/api/generate` streaming and returns raw text (`agents/quantitative_assessor_z.py:58-71`).
- Qualitative assessor uses `/api/generate` streaming and returns raw text (`agents/qualitative_assessor_f.py:117-134`).
- Meta-review uses `ollama.Client.chat(...)` and hard-codes `model="llama3"` (`agents/meta_reviewer.py:30-35`).

### 3) Qualitative prompt XML appears malformed

**Location:** `agents/qualitative_assessor_f.py:51-55`

```text
               <little_interest_or_pleasure>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
               </little_interest or pleasure>
```

### 4) Debug prints / potential data leakage

**Location:** `agents/qualitive_evaluator.py:139-167`

The qualitative evaluator prints full prompts, request payloads, and model outputs to stdout (e.g., `print(prompts[label])`, `print(request)`, `print(content)`).

### 5) Hard-coded personal/cluster details in SLURM scripts

- `slurm/job_assess.sh:14` contains `#SBATCH --mail-user=xli77@gsu.edu`.
- `slurm/job_ollama.sh:25` sets `export OLLAMA_MODELS=/data/users4/splis/ollama/models/`.

### 6) Visualization notebooks are not portable (absolute paths)

- `visualization/quan_visualization.ipynb:164-167` loads inputs from `/data/users2/...` and `/data/users4/...`.
- `visualization/quan_visualization.ipynb:365` writes to `/data/users2/agreene46/heatmap_output.pdf`.
- `visualization/qual_boxplot.ipynb:51-55` reads `/data/users2/nblair7/...`.
- `visualization/meta_review_heatmap.ipynb:127-129` hardcodes `rootdir = "/data/users4/user/ai-psychiatrist"`.

### 7) Manual participant exclusion in baseline script

**Location:** `quantitative_assessment/quantitative_analysis.py:31-35`

The baseline script contains a hard-coded `values_already_done = {...}` list and removes those IDs from processing with a comment indicating it was done because the execution loop "kept breaking".

### 8) Repository hygiene

- `agents/.DS_Store` is committed (macOS metadata file).

## Unanswered Questions for Authors

### Priority 1: Methodology Clarification

1. **"Which exact script/notebook produced the MAE 0.619 results?"**
   - Was it `quantitative_assessment/embedding_batch_script.py` or `agents/quantitative_assessor_f.py`?
   - What commit hash corresponds to the paper results?

2. **"Was the `_keyword_backfill()` function in `agents/quantitative_assessor_f.py` used for any reported results?"**
   - If not, why is it enabled unconditionally in the released server agent?
   - If yes, why wasn't it documented in the paper?

3. **"Was model-assisted JSON repair used during experiments?"**
   - The released agent includes a model call to repair malformed JSON during scoring (`agents/quantitative_assessor_f.py:58-79`, `agents/quantitative_assessor_f.py:501-529`).
   - If this was used for reported results, it should be documented; if not, why is it enabled in the released pipeline?

4. **"How were N/A predictions handled in the MAE calculation?"**
   - Were they excluded (as the visualization notebook does)?
   - What percentage of predictions were N/A?

### Priority 2: Model & Hardware

5. **"What exactly is `gemma3-optimized:27b`?"**
   - Is it a quantized GGUF version?
   - What are the specific model files/hashes?

6. **"Were experiments run on MacBook M3 Pro or on A100 cluster?"**
   - The paper claims M3 Pro for clinical accessibility
   - The repo uses SLURM + A100 for Ollama serving

7. **"Why does the server agent default to `llama3` instead of Gemma 3 27B?"**

### Priority 3: Reproducibility

8. **"Can you provide the exact hyperparameters for all experiments?"**
   - temperature, top_k, top_p for both zero-shot and few-shot
   - Random seeds used

9. **"Can you provide the exact participant ID splits?"**
   - Train/validation/test splits
   - Random seed for splitting

10. **"Why do notebooks and agent files use different temperature settings?"**
   - Few-shot uses `temperature=0.2, top_k=20, top_p=0.8` in multiple code paths (`agents/quantitative_assessor_f.py:200`, `quantitative_assessment/embedding_batch_script.py:374-379`)
   - One zero-shot baseline uses `temperature=0, top_k=1, top_p=1.0` (`quantitative_assessment/quantitative_analysis.py:136-140`)
   - The server's zero-shot agent uses `/api/generate` without explicit sampling options (`agents/quantitative_assessor_z.py:58-71`)

---

## Reproduction Recommendations

Based on this analysis, to reproduce MAE 0.619:

1. **Start from the stored MAE logic** in `visualization/quan_visualization.ipynb:573-585` and `visualization/quan_visualization.ipynb:660-663` (exclude `"N/A"`, compute per-question MAEs, then `np.nanmean` across questions).
2. **Be aware notebooks hard-code absolute paths** (e.g., `visualization/quan_visualization.ipynb:164-167`), so they may require path edits to run locally.
3. **Use the same model tag used by the producing scripts**, then confirm with authors whether it corresponds to "Gemma 3 27B" (e.g., several scripts use `gemma3-optimized:27b`; see table in Finding 4).
4. **Match sampling hyperparameters** where applicable (e.g., few-shot uses `temperature=0.2, top_k=20, top_p=0.8` in `agents/quantitative_assessor_f.py:200` and `quantitative_assessment/embedding_batch_script.py:374-379`).
5. **Avoid the FastAPI server path for reproduction** unless you explicitly configure models/artifacts, since defaults differ (Finding 2) and keyword backfill is active (Finding 1).

Your reproduction gap (0.778 vs 0.619) may be explained by:
- Using wrong model (server default is llama3)
- Using agent code with keyword backfill (which may hurt rather than help)
- Different hyperparameters
- Different MAE calculation methodology

---

## Document Verification Checklist

This document has been verified against the codebase. Key verification points:

1. [x] Confirm `agents/quantitative_assessor_f.py:419` defaults to `"llama3"` - **VERIFIED**
2. [x] Confirm `server.py:23` instantiates `QuantitativeAssessorF()` with defaults - **VERIFIED**
3. [x] Confirm `agents/quantitative_assessor_f.py:478` calls `_keyword_backfill()` - **VERIFIED**
4. [x] Confirm `quantitative_assessment/` notebooks have NO keyword backfill - **VERIFIED**
5. [x] Confirm `slurm/job_ollama.sh:4` sets #SBATCH --gres=gpu:A100:2 - **VERIFIED**
6. [x] Confirm `visualization/quan_visualization.ipynb:398-401` contains the printed MAE line `0.61929388` - **VERIFIED**
7. [x] Confirm `visualization/quan_visualization.ipynb:660-663` aggregates MAEs with `np.nanmean` - **VERIFIED**
8. [x] Git blame shows `37747eb` introduced keyword backfill - **VERIFIED**

---

## Conclusion

The repository contains multiple, divergent implementations of the described pipeline (offline scripts/notebooks vs. FastAPI "agents"), with mismatched defaults (e.g., `llama3`), undocumented heuristics (e.g., keyword backfill, JSON repair), and many hard-coded cluster paths. While the paper-reported MAE values match outputs stored in the visualization notebook, the repo does not clearly tie those results to a single runnable script+commit and environment.

Before raising concerns with authors, request a precise pointer to: (1) the exact script/notebook, (2) the exact commit hash, (3) the exact Ollama model tags, and (4) the exact evaluation procedure used for the reported MAEs.

---

*Last updated: December 24, 2025*
