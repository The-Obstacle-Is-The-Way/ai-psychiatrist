# Spec 23: Nested Try Block Cleanup

> **STATUS: OPEN**
>
> **Priority**: Low - Code works correctly. This is maintainability/readability debt.
>
> **GitHub Issue**: #60 (parent tech-debt audit)
>
> **Created**: 2025-12-26
>
> **Last Verified Against Code**: 2025-12-27

---

## Problem Statement

`scripts/reproduce_results.py` contains nested try/finally blocks for client lifecycle management inside `main_async()`. This makes the control flow harder to follow and increases cognitive load. As of 2025-12-27, the nesting is at `scripts/reproduce_results.py:767-808`.

---

## Current State: Nested Try Pattern

### Location: `reproduce_results.py` lines 767-810

```python
# Current structure (simplified)
async with OllamaClient(ollama_settings) as ollama_client:
    if not await check_ollama_connectivity(ollama_client):
        return 1

    transcript_service = TranscriptService(data_settings)

    embedding_client = create_embedding_client(settings)  # Not a context manager
    try:
        try:
            embedding_service = init_embedding_service(...)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            return 1

        experiments = await run_requested_experiments(...)
        persist_experiment_outputs(...)

    finally:
        await embedding_client.close()

    return 0
```

**Issues**:
1. Three nesting levels (async with > try > try)
2. Manual `finally` cleanup for `embedding_client`
3. Early return in inner try requires careful reasoning about finally execution
4. Mix of context manager (`OllamaClient`) and manual cleanup (`embedding_client`)

---

## Refactoring Options

### Option A: AsyncExitStack (Recommended)

Use `contextlib.AsyncExitStack` to manage multiple async resources uniformly:

```python
from contextlib import AsyncExitStack

async def main_async(args: argparse.Namespace) -> int:
    async with AsyncExitStack() as stack:
        # Enter all async contexts
        ollama_client = await stack.enter_async_context(
            OllamaClient(ollama_settings)
        )
        if not await check_ollama_connectivity(ollama_client):
            return 1

        embedding_client = create_embedding_client(settings)
        stack.push_async_callback(embedding_client.close)

        try:
            embedding_service = init_embedding_service(...)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            return 1

        experiments = await run_requested_experiments(...)
        persist_experiment_outputs(...)

    return 0
```

**Benefits**:
- Single nesting level for resource management
- Cleanup order is explicit and guaranteed
- Easy to add more resources
- Standard library pattern

### Option B: Make Embedding Client a Context Manager

Modify the embedding client factory to return a context manager:

```python
# In infrastructure/llm/factory.py
@asynccontextmanager
async def create_embedding_client_context(settings: Settings) -> AsyncIterator[EmbeddingClient]:
    client = create_embedding_client(settings)
    try:
        yield client
    finally:
        await client.close()

# In reproduce_results.py
async with OllamaClient(ollama_settings) as ollama_client:
    if not await check_ollama_connectivity(ollama_client):
        return 1

    async with create_embedding_client_context(settings) as embedding_client:
        try:
            embedding_service = init_embedding_service(...)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            return 1

        experiments = await run_requested_experiments(...)
        persist_experiment_outputs(...)

    return 0
```

**Benefits**:
- Consistent pattern with OllamaClient
- Resource cleanup is automatic
- Still has nested contexts but cleaner

### Option C: Extract Service Initialization (Partial)

Extract initialization into a separate function that handles errors:

```python
async def initialize_services(
    settings: Settings,
    args: argparse.Namespace,
    ollama_client: OllamaClient,
) -> tuple[TranscriptService, EmbeddingService | None, EmbeddingClient] | None:
    """Initialize all services, returning None on failure."""
    transcript_service = TranscriptService(settings.data)

    embedding_client = create_embedding_client(settings)
    try:
        embedding_service = init_embedding_service(...)
        return transcript_service, embedding_service, embedding_client
    except FileNotFoundError as e:
        await embedding_client.close()
        print(f"\nERROR: {e}")
        return None

# In main
async with OllamaClient(ollama_settings) as ollama_client:
    if not await check_ollama_connectivity(ollama_client):
        return 1

    services = await initialize_services(settings, args, ollama_client)
    if services is None:
        return 1

    transcript_service, embedding_service, embedding_client = services
    try:
        experiments = await run_requested_experiments(...)
        persist_experiment_outputs(...)
    finally:
        await embedding_client.close()

    return 0
```

**Benefits**:
- Separates initialization from execution
- Error handling is localized

**Drawbacks**:
- Still has try/finally for cleanup
- Returns tuple (less clean than context manager)

---

## Recommendation

**Use Option A (AsyncExitStack)** for the following reasons:

1. It's the standard Python pattern for managing multiple async resources
2. It handles cleanup in reverse order automatically
3. It works with both context managers and async callbacks
4. It's explicit about resource lifetime
5. No need to modify the embedding client interface

---

## Implementation Plan

### Step 1: Import AsyncExitStack

```python
from contextlib import AsyncExitStack
```

### Step 2: Refactor `main_async()`

Replace the nested try blocks with AsyncExitStack:

```python
async def main_async(args: argparse.Namespace) -> int:
    # ... setup code ...

    if args.dry_run:
        print("\n[DRY RUN] Would run evaluation with above settings.")
        return 0

    try:
        ground_truth = load_ground_truth_for_split(...)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    async with AsyncExitStack() as stack:
        # Ollama client (async context manager)
        ollama_client = await stack.enter_async_context(
            OllamaClient(ollama_settings)
        )
        if not await check_ollama_connectivity(ollama_client):
            return 1

        transcript_service = TranscriptService(data_settings)

        # Embedding client (manual cleanup via callback)
        embedding_client = create_embedding_client(settings)
        stack.push_async_callback(embedding_client.close)

        try:
            embedding_service = init_embedding_service(...)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            return 1

        experiments = await run_requested_experiments(...)
        persist_experiment_outputs(...)

    return 0
```

### Step 3: Update Tests

Verify existing integration tests still pass.

---

## Acceptance Criteria

- [ ] No triple-nested try blocks
- [ ] All async resources cleaned up via AsyncExitStack or context managers
- [ ] Existing tests pass
- [ ] Error handling behavior unchanged (FileNotFoundError still returns 1)
- [ ] Resource cleanup order is correct (embedding_client before ollama_client)

---

## Files To Modify

```
scripts/reproduce_results.py    (lines 767-810)
```

---

## Before/After Comparison

### Before (Current)

```
async with OllamaClient:
    try:
        try:
            init_embedding_service()
        except FileNotFoundError:
            return 1
        # ... work ...
    finally:
        await embedding_client.close()
```

**Nesting depth**: 4 (async with > try > try > except)

### After (AsyncExitStack)

```
async with AsyncExitStack as stack:
    enter_async_context(OllamaClient)
    push_async_callback(embedding_client.close)
    try:
        init_embedding_service()
    except FileNotFoundError:
        return 1
    # ... work ...
```

**Nesting depth**: 2 (async with > try)

---

## References

- GitHub Issue: #60 (tech-debt: Code readability audit)
- CodeRabbit: PR #57 review
- Python docs: [AsyncExitStack](https://docs.python.org/3/library/contextlib.html#contextlib.AsyncExitStack)
- PEP 343: The "with" Statement
