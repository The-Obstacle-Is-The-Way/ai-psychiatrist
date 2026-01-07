# Bug Audit Checklist for AI/ML Research Codebases

**Purpose**: Continuous audit reference for identifying common bugs, anti-patterns, and silent failures in AI/ML research codebases—especially those using AI-assisted code generation.

**Last Updated**: 2026-01-07
**Sources**: Compiled from 2025-2026 research on AI code generation bugs, ML reproducibility failures, and Python anti-patterns.

---

## Table of Contents

- [Bug Audit Checklist for AI/ML Research Codebases](#bug-audit-checklist-for-aiml-research-codebases)
  - [Table of Contents](#table-of-contents)
  - [1. AI Code Generation Bugs](#1-ai-code-generation-bugs)
    - [Checklist](#checklist)
    - [Red Flags](#red-flags)
  - [2. Python Truthiness Traps](#2-python-truthiness-traps)
    - [Checklist](#checklist-1)
    - [Bad vs Good](#bad-vs-good)
  - [3. Silent Exception Swallowing](#3-silent-exception-swallowing)
    - [Checklist](#checklist-2)
    - [Bad vs Good](#bad-vs-good-1)
  - [4. Type Safety \& Validation Bypass](#4-type-safety--validation-bypass)
    - [Checklist](#checklist-3)
    - [Bad vs Good](#bad-vs-good-2)
  - [5. ML-Specific Reproducibility Bugs](#5-ml-specific-reproducibility-bugs)
    - [Checklist](#checklist-4)
    - [Best Practice](#best-practice)
  - [6. Data Leakage \& Train-Test Contamination](#6-data-leakage--train-test-contamination)
    - [Checklist](#checklist-5)
    - [Bad vs Good](#bad-vs-good-3)
  - [7. Async \& Concurrency Bugs](#7-async--concurrency-bugs)
    - [Checklist](#checklist-6)
    - [Bad vs Good](#bad-vs-good-4)
  - [8. Floating Point \& Numerical Bugs](#8-floating-point--numerical-bugs)
    - [Checklist](#checklist-7)
    - [Bad vs Good](#bad-vs-good-5)
  - [9. NumPy/Pandas Silent Failures](#9-numpypandas-silent-failures)
    - [Checklist](#checklist-8)
    - [Bad vs Good](#bad-vs-good-6)
  - [10. Off-by-One \& Fencepost Errors](#10-off-by-one--fencepost-errors)
    - [Checklist](#checklist-9)
    - [Bad vs Good](#bad-vs-good-7)
  - [11. Mutable Default Arguments](#11-mutable-default-arguments)
    - [Checklist](#checklist-10)
    - [Bad vs Good](#bad-vs-good-8)
  - [12. Security Vulnerabilities](#12-security-vulnerabilities)
    - [Checklist](#checklist-11)
    - [Bad vs Good](#bad-vs-good-9)
  - [13. Hardcoded Secrets](#13-hardcoded-secrets)
    - [Checklist](#checklist-12)
    - [Detection](#detection)
  - [14. Error Handling Anti-Patterns](#14-error-handling-anti-patterns)
    - [Checklist](#checklist-13)
    - [Bad vs Good](#bad-vs-good-10)
  - [15. Resource \& Connection Leaks](#15-resource--connection-leaks)
    - [Checklist](#checklist-14)
    - [Bad vs Good](#bad-vs-good-11)
    - [Detection](#detection-1)
  - [16. N+1 Query Problems](#16-n1-query-problems)
    - [Checklist](#checklist-15)
    - [Bad vs Good](#bad-vs-good-12)
    - [Detection](#detection-2)
  - [17. Circular Imports \& TYPE\_CHECKING](#17-circular-imports--type_checking)
    - [Checklist](#checklist-16)
    - [Bad vs Good](#bad-vs-good-13)
    - [Detection](#detection-3)
  - [18. TODO/FIXME Incomplete Implementations](#18-todofixme-incomplete-implementations)
    - [Checklist](#checklist-17)
    - [Detection](#detection-4)
  - [19. Mock Overuse \& Test False Positives](#19-mock-overuse--test-false-positives)
    - [Checklist](#checklist-18)
    - [Bad vs Good](#bad-vs-good-14)
    - [Detection](#detection-5)
  - [20. Schema Drift \& Data Pipeline Corruption](#20-schema-drift--data-pipeline-corruption)
    - [Checklist](#checklist-19)
    - [Bad vs Good](#bad-vs-good-15)
  - [21. JSON Serialization Edge Cases](#21-json-serialization-edge-cases)
    - [Checklist](#checklist-20)
    - [Bad vs Good](#bad-vs-good-16)
  - [22. Static vs Runtime Type Mismatch](#22-static-vs-runtime-type-mismatch)
    - [Checklist](#checklist-21)
    - [Bad vs Good](#bad-vs-good-17)
    - [Detection](#detection-6)
  - [Quick Audit Commands](#quick-audit-commands)
  - [References](#references)
    - [AI Code Generation](#ai-code-generation)
    - [Python Anti-Patterns](#python-anti-patterns)
    - [ML Reproducibility](#ml-reproducibility)
    - [Type Safety](#type-safety)
    - [Concurrency](#concurrency)
    - [Testing](#testing)
    - [Database \& ORM](#database--orm)
    - [Data Pipelines](#data-pipelines)
    - [Security](#security)
  - [Statistics (2025-2026 Research)](#statistics-2025-2026-research)

---

## 1. AI Code Generation Bugs

AI-generated code creates **1.75x more logic errors**, **1.64x more maintainability issues**, **1.57x more security findings**, and **1.42x more performance problems** than human-written code ([The Register, 2025](https://www.theregister.com/2025/12/17/ai_code_bugs/)).

### Checklist

- [ ] **Skipped deep code review**: 90% of AI-introduced production bugs stem from skipping review when AI output "looks right"
- [ ] **Hallucinated dependencies**: AI may suggest libraries that don't exist or have known CVEs patched after training cutoff
- [ ] **Jumbled monolithic outputs**: Large AI-generated blocks often contain inconsistency and duplication
- [ ] **Missing input sanitization**: 40%+ of AI-generated code has input validation flaws
- [ ] **Hardcoded secrets**: AI trained on code containing credentials may reproduce this anti-pattern
- [ ] **Prompt injection vectors**: Malicious comments/READMEs can trick AI into writing backdoors (OWASP #1 LLM risk)

### Red Flags
```python
# AI often generates overly trusting code like:
def process_input(data):
    return eval(data)  # No sanitization!

# Or hardcoded values:
API_KEY = "sk-abc123..."  # Never do this
```

---

## 2. Python Truthiness Traps

"The mistake here is coalescing two distinct conditions (None and 0) and treating them as though they were one. This is a very common anti-pattern." ([Inspired Python](https://www.inspiredpython.com/article/truthy-and-falsy-gotchas))

### Checklist

- [ ] **`if value:` when checking for None**: Conflates `None`, `0`, `""`, `[]`, `{}`, `False`
- [ ] **`if not value:` instead of `if value is None:`**: 0 and empty collections evaluate as falsy
- [ ] **CLI argument handling with truthiness**: `--limit 0` treated as "not set"
- [ ] **Optional return values checked with `if result:`**: May miss valid falsy returns

### Bad vs Good
```python
# BAD: Truthiness trap
if limit:  # 0 is treated as "not set"
    participants = participants[:limit]

# GOOD: Explicit None check
if limit is not None:
    participants = participants[:limit]

# BAD: XML element check
element = dom.find("tag")
if element:  # Empty element is falsy but exists!
    process(element)

# GOOD: Explicit None check
if element is not None:
    process(element)
```

---

## 3. Silent Exception Swallowing

"Letting an exception that occurred pass silently is bad practice." ([Pybites](https://pybit.es/articles/error_handling/))

### Checklist

- [ ] **Bare `except:` blocks**: Catches everything including `SystemExit`, `KeyboardInterrupt`
- [ ] **`except Exception: pass`**: Silently swallows all errors
- [ ] **Missing logging in except blocks**: Errors occur but are never recorded
- [ ] **Async task exceptions never retrieved**: Background task errors lost
- [ ] **`except Exception as e:` without re-raise or logging**: Error captured but ignored
- [ ] **Try/except as control flow**: Using exceptions for normal logic

### Bad vs Good
```python
# BAD: Silent swallowing
try:
    risky_operation()
except Exception:
    pass  # What went wrong? We'll never know.

# BAD: Bare except
try:
    process()
except:  # Catches KeyboardInterrupt too!
    handle_error()

# GOOD: Specific, logged, optionally re-raised
try:
    risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise  # Or handle meaningfully
```

---

## 4. Type Safety & Validation Bypass

"CLI overrides can bypass Pydantic constraints." (BUG-036 in this repo)

### Checklist

- [ ] **Pydantic validates `.env` but not CLI overrides**: Direct assignment bypasses validators
- [ ] **Missing range validation for numeric arguments**: Out-of-range values silently accepted
- [ ] **Type coercion surprises**: `"123"` silently becomes `123` (may or may not be wanted)
- [ ] **Strict mode not enabled when needed**: Pydantic coerces by default
- [ ] **`# type: ignore` overuse**: Masks real type issues
- [ ] **Dynamic Django/ORM fields confusing static analysis**: Runtime magic breaks type checks

### Bad vs Good
```python
# BAD: CLI override bypasses Pydantic validation
settings.some_field = args.override  # No range check!

# GOOD: Explicit validation
def validate_cli_args(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    # ... validate other overrides
```

---

## 5. ML-Specific Reproducibility Bugs

Random seed choice can cause **44-45% accuracy variation** on the same dataset ([CMU SEI](https://www.sei.cmu.edu/blog/the-myth-of-machine-learning-reproducibility-and-randomness-for-acquisitions-and-testing-evaluation-verification-and-validation/)).

### Checklist

- [ ] **Random seed not set or not logged**: Results vary between runs
- [ ] **Multiple RNG sources not all seeded**: Python, NumPy, PyTorch, CUDA each have separate RNGs
- [ ] **GPU non-determinism not addressed**: cuDNN benchmarking selects different algorithms
- [ ] **Parallel execution breaks reproducibility**: Threading/multiprocessing introduces variance
- [ ] **Weight initialization not controlled**: Different starting weights, different outcomes
- [ ] **Library version not pinned**: Different versions produce different results
- [ ] **Hardware differences not documented**: CPU vs GPU, different GPU models

### Best Practice
```python
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## 6. Data Leakage & Train-Test Contamination

Princeton researchers found **648 papers affected by data leakage** across 30 scientific fields ([Princeton Reproducibility](https://reproducible.cs.princeton.edu/)).

### Checklist

- [ ] **Preprocessing before split**: Scaling/normalization fit on full dataset
- [ ] **Feature engineering uses test data**: Statistics computed across all data
- [ ] **Time series future leakage**: Using future data to predict past
- [ ] **Duplicate/near-duplicate samples across splits**: Same data in train and test
- [ ] **External data join without timestamp filtering**: Inadvertent overlap
- [ ] **Target leakage**: Features that encode the target variable
- [ ] **Participant-level scores assigned to chunks**: Our BUG-081—scores must match data granularity

### Bad vs Good
```python
# BAD: Preprocessing before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses all data!
X_train, X_test = train_test_split(X_scaled)

# GOOD: Split first, then fit only on train
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train only
X_test = scaler.transform(X_test)  # Transform test with train params
```

---

## 7. Async & Concurrency Bugs

"Race conditions are a real problem in Python when using threads, even in the presence of the GIL." ([Medium, 2025](https://medium.com/pythoneers/avoiding-race-conditions-in-python-in-2025-best-practices-for-async-and-threads-4e006579a622))

### Checklist

- [ ] **Coroutine never awaited**: `coro()` instead of `await coro()` creates orphaned coroutine
- [ ] **Task exception never retrieved**: `asyncio.create_task()` exceptions silently lost
- [ ] **Race condition on shared state**: Multiple threads/coroutines mutating same data
- [ ] **Missing locks on shared resources**: Concurrent access without synchronization
- [ ] **GIL false security**: GIL doesn't prevent all race conditions
- [ ] **Background task failures unnoticed**: Errors in spawned tasks not propagated

### Bad vs Good
```python
# BAD: Coroutine never awaited
async def main():
    fetch_data()  # Creates coroutine object, never runs!

# GOOD: Await the coroutine
async def main():
    await fetch_data()

# BAD: Unmonitored background task
async def main():
    asyncio.create_task(risky_operation())  # Exceptions lost!

# GOOD: Track and handle task exceptions
async def main():
    task = asyncio.create_task(risky_operation())
    try:
        await task
    except Exception as e:
        logger.error(f"Background task failed: {e}")
```

---

## 8. Floating Point & Numerical Bugs

`0.1 + 0.2 != 0.3` due to IEEE 754 representation limits ([Floating-Point Guide](https://floating-point-gui.de/errors/comparison/)).

### Checklist

- [ ] **Direct `==` comparison of floats**: Almost always wrong
- [ ] **Fixed epsilon that doesn't scale**: "Looks small" epsilon may be huge or tiny relative to values
- [ ] **NaN comparisons**: `NaN != NaN` by definition; use `math.isnan()`
- [ ] **Division producing infinity**: Unchecked division by near-zero
- [ ] **Accumulating rounding errors**: Summing many floats compounds error
- [ ] **Financial calculations with float**: Use `Decimal` instead

### Bad vs Good
```python
# BAD: Direct comparison
if result == 0.3:  # May never be true!
    process()

# GOOD: Use math.isclose() or numpy.isclose()
if math.isclose(result, 0.3, rel_tol=1e-9):
    process()

# BAD: Unchecked NaN
if value > threshold:  # NaN comparisons are always False
    process()

# GOOD: Check for NaN first
if not math.isnan(value) and value > threshold:
    process()
```

---

## 9. NumPy/Pandas Silent Failures

Broadcasting can silently produce wrong results without raising errors ([NumPy Docs](https://numpy.org/doc/stable/user/basics.broadcasting.html)).

### Checklist

- [ ] **Shape mismatch silently broadcast**: Operations on incompatible shapes
- [ ] **DataFrame vs ndarray broadcasting differences**: Pandas and NumPy behave differently
- [ ] **Chained indexing assignment**: `df[col][row] = val` may not work as expected
- [ ] **Silent downcasting deprecated (Pandas 3.0)**: Check for deprecation warnings
- [ ] **Object dtype masking specific types**: String columns as object lose type info
- [ ] **Integer overflow in NumPy**: Wraps around silently with fixed-width ints

### Bad vs Good
```python
# BAD: Silent broadcasting may produce unexpected shapes
result = array_a * array_b  # Are shapes compatible?

# GOOD: Explicit shape assertions
assert array_a.shape == array_b.shape, f"Shape mismatch: {array_a.shape} vs {array_b.shape}"
result = array_a * array_b

# BAD: Chained indexing
df["col"]["row"] = value  # May be a copy!

# GOOD: Use .loc or .iloc
df.loc["row", "col"] = value
```

---

## 10. Off-by-One & Fencepost Errors

"If you build a fence 100 feet long with posts 10 feet apart, how many posts do you need?" ([Incus Data](https://incusdata.com/blog/off-by-one-errors))

### Checklist

- [ ] **Range end confusion**: `range(n)` is 0 to n-1, not 0 to n
- [ ] **Loop boundary wrong**: `<= n` vs `< n`
- [ ] **Array index miscalculation**: Zero-based indexing errors
- [ ] **User-facing 1-based vs internal 0-based**: Conversion errors
- [ ] **Length vs index confusion**: n elements means indices 0 to n-1
- [ ] **Slice endpoint**: `items[start:end]` excludes `end`

### Bad vs Good
```python
# BAD: Off-by-one in user input handling
choice = int(input("Enter choice (1-3): "))
items[choice]  # Should be items[choice - 1]!

# GOOD: Convert 1-based user input to 0-based index
choice = int(input("Enter choice (1-3): "))
items[choice - 1]

# BAD: Fencepost in loop
for i in range(len(items) + 1):  # One too many!
    process(items[i])  # IndexError on last iteration

# GOOD: Correct range
for i in range(len(items)):
    process(items[i])
```

---

## 11. Mutable Default Arguments

"Python's default arguments are evaluated once when the function is defined, not each time the function is called." ([Hitchhiker's Guide](https://docs.python-guide.org/writing/gotchas/))

### Checklist

- [ ] **List/dict/set as default argument**: Shared across all calls
- [ ] **Class attribute that's a mutable object**: Shared across all instances
- [ ] **Default `datetime.now()`**: Evaluated at definition time, not call time

### Bad vs Good
```python
# BAD: Mutable default
def add_item(item, items=[]):  # Shared list!
    items.append(item)
    return items

add_item("a")  # ["a"]
add_item("b")  # ["a", "b"] - Oops!

# GOOD: Use None sentinel
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# BAD: Default datetime
def log_event(msg, timestamp=datetime.now()):  # Fixed at import!
    ...

# GOOD: Compute at call time
def log_event(msg, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    ...
```

---

## 12. Security Vulnerabilities

AI-generated code is **2.74x more likely to have XSS vulnerabilities**, **1.91x more likely to have insecure object references** ([Dark Reading](https://www.darkreading.com/application-security/llms-ai-generated-code-wildly-insecure)).

### Checklist

- [ ] **SQL injection**: String formatting in queries instead of parameterized
- [ ] **Command injection**: `os.system()` or `subprocess` with unsanitized input
- [ ] **XSS vulnerabilities**: Unescaped user input in HTML output
- [ ] **Insecure deserialization**: `pickle.loads()` on untrusted data
- [ ] **Path traversal**: User input in file paths without sanitization
- [ ] **Weak cryptography**: MD5/SHA1 for security, hardcoded IVs/keys
- [ ] **Improper password handling**: Plaintext storage, weak hashing

### Bad vs Good
```python
# BAD: SQL injection
cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")

# GOOD: Parameterized query
cursor.execute("SELECT * FROM users WHERE name = ?", (name,))

# BAD: Command injection
os.system(f"echo {user_input}")

# GOOD: Use subprocess with list args
subprocess.run(["echo", user_input], check=True)
```

---

## 13. Hardcoded Secrets

Security researchers found **11,908 live API keys** in publicly accessible web pages; 63% were reused across multiple sites ([Medium, 2025](https://medium.com/@instatunnel/hardcoded-api-keys-the-rookie-mistake-that-costs-millions-fa6da9dcc494)).

### Checklist

- [ ] **API keys in source code**: Check all strings for key patterns
- [ ] **Passwords in config files**: Even "example" files may contain real creds
- [ ] **Secrets in environment dumps**: Logs may capture env vars
- [ ] **Secrets in git history**: Even if deleted, still in history
- [ ] **Secrets in Docker images**: Layered filesystem preserves deleted files
- [ ] **Test credentials that are actually real**: "Test" keys that work in prod

### Detection
```bash
# Use secret scanning tools
trufflehog git file://./
gitleaks detect
```

---

## 14. Error Handling Anti-Patterns

"Prefer loud failures over best-effort fallbacks." (This repo's policy)

### Checklist

- [ ] **Silent fallbacks**: Substituting default values without warning
- [ ] **Catch-all exception handlers**: `except Exception` hiding specific errors
- [ ] **Error codes instead of exceptions**: Returning `-1` or `None` on failure
- [ ] **Logging but not failing**: Recording error but continuing anyway
- [ ] **Retry without backoff**: Hammering a failing service
- [ ] **Ignoring return values**: Not checking if operation succeeded

### Bad vs Good
```python
# BAD: Silent fallback
def get_config(key):
    try:
        return config[key]
    except KeyError:
        return None  # Caller doesn't know config was missing!

# GOOD: Explicit failure
def get_config(key):
    if key not in config:
        raise ConfigurationError(f"Missing required config: {key}")
    return config[key]

# BAD: Best-effort that masks failures
def process_all(items):
    for item in items:
        try:
            process(item)
        except Exception:
            pass  # How many failed? Why?

# GOOD: Collect and report failures
def process_all(items):
    failures = []
    for item in items:
        try:
            process(item)
        except ProcessingError as e:
            failures.append((item, e))
    if failures:
        raise BatchProcessingError(f"{len(failures)} items failed", failures)
```

---

## 15. Resource & Connection Leaks

Files, database connections, and HTTP clients that aren't properly closed lead to resource exhaustion.

### Checklist

- [ ] **Files opened without `with` statement**: Not guaranteed to close on exception
- [ ] **Database connections not returned to pool**: Pool exhaustion under load
- [ ] **HTTP clients not closed**: Socket leaks (`requests.Session`, `httpx.Client`)
- [ ] **`multiprocessing.Pool` without context manager**: Zombie processes
- [ ] **Async context managers not awaited properly**: `async with` missing
- [ ] **Missing `finally` blocks for cleanup**: Resources leaked on exception
- [ ] **Thread/executor pools not shut down**: Hangs on exit

### Bad vs Good
```python
# BAD: File not guaranteed to close
f = open("data.txt")
data = f.read()
f.close()  # Never reached if exception above!

# GOOD: Context manager guarantees close
with open("data.txt") as f:
    data = f.read()

# BAD: Connection pool exhaustion
def get_data():
    conn = pool.getconn()
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()
    # Connection never returned!

# GOOD: Context manager returns connection
async with pool.connection() as conn:
    async with conn.cursor() as cursor:
        await cursor.execute(query)
        return await cursor.fetchall()
```

### Detection
```bash
# Find file opens without context manager
grep -rn "open(" --include="*.py" . | grep -v "with "

# Find connection creation without context
grep -rn "\.connect(" --include="*.py" . | grep -v "with "
grep -rn "Session()" --include="*.py" . | grep -v "with "
```

---

## 16. N+1 Query Problems

"The silent performance killer" — 100 items = 101 database queries.

### Checklist

- [ ] **Lazy-loaded relationships accessed in loops**: Classic N+1
- [ ] **Missing `joinedload`/`selectinload` in SQLAlchemy**
- [ ] **Serializers triggering relationship loads**: Marshmallow/Pydantic accessing relations
- [ ] **Nested API responses without eager loading**
- [ ] **No query logging in development**: Can't see the problem

### Bad vs Good
```python
# BAD: N+1 queries
books = session.query(Book).all()
for book in books:
    print(book.author.name)  # Triggers query per book!

# GOOD: Eager loading
from sqlalchemy.orm import selectinload

books = session.query(Book).options(selectinload(Book.author)).all()
for book in books:
    print(book.author.name)  # No additional queries
```

### Detection
```bash
# Enable SQLAlchemy query logging
# In config: echo=True or logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

# Check for eager loading usage
grep -rn "joinedload\|selectinload\|subqueryload" --include="*.py" .
```

---

## 17. Circular Imports & TYPE_CHECKING

Type hints increase circular import likelihood; breaks at runtime even when MyPy passes.

### Checklist

- [ ] **Runtime `ImportError: cannot import name`**: Circular dependency
- [ ] **`TYPE_CHECKING` imports used at runtime**: Should only be for hints
- [ ] **Missing `from __future__ import annotations`**: Forward refs not deferred
- [ ] **Import at module level that should be local**: Move import inside function
- [ ] **Tightly coupled modules**: Design smell indicating need for refactor

### Bad vs Good
```python
# BAD: Runtime circular import
from other_module import OtherClass  # Circular!

class MyClass:
    def method(self) -> OtherClass: ...

# GOOD: TYPE_CHECKING guard
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from other_module import OtherClass

class MyClass:
    def method(self) -> OtherClass: ...  # String forward ref
```

### Detection
```bash
# Test import
python -c "import src.module"  # Will fail on circular imports

# Find TYPE_CHECKING usage (verify it's used correctly)
grep -rn "TYPE_CHECKING" --include="*.py" .
```

---

## 18. TODO/FIXME Incomplete Implementations

AI often leaves placeholder code that looks complete but isn't.

### Checklist

- [ ] **`TODO` comments in production code**: Unfinished work
- [ ] **`FIXME` markers**: Known bugs not addressed
- [ ] **`pass` in function bodies**: Stub implementations
- [ ] **`raise NotImplementedError`**: Intentionally incomplete
- [ ] **Hardcoded return values**: `return 0`, `return []`, `return {}`
- [ ] **Commented-out code**: Dead code that may confuse
- [ ] **`...` (ellipsis) in function bodies**: Placeholder

### Detection
```bash
# Find TODO/FIXME markers
grep -rn "TODO\|FIXME\|XXX\|HACK\|BUG" --include="*.py" .

# Find stub implementations
grep -rn "pass$" --include="*.py" .
grep -rn "NotImplementedError" --include="*.py" .
grep -rn "^\s*\.\.\.$" --include="*.py" .

# Find suspicious hardcoded returns
grep -rn "return 0$\|return \[\]$\|return {}$\|return None$" --include="*.py" .
```

---

## 19. Mock Overuse & Test False Positives

Tests pass but production fails because mocks don't match real behavior.

### Checklist

- [ ] **Mocking internal logic instead of external boundaries**: Over-mocking
- [ ] **Missing `autospec=True`**: Mocks accept any signature silently
- [ ] **Global module patches**: `@patch('requests.post')` affects unrelated tests
- [ ] **Tests that verify mock was called, not behavior**: Testing the mock, not code
- [ ] **Mocked return values that don't match real API**: Stale mock data
- [ ] **Test-only code paths in production code**: `if testing:` branches

### Bad vs Good
```python
# BAD: Mock accepts wrong signature silently
@patch('module.api_call')
def test_function(mock_call):
    mock_call.return_value = {"data": []}
    result = function_under_test("arg1", "arg2", "EXTRA_ARG")  # No error!

# GOOD: autospec enforces real signature
@patch('module.api_call', autospec=True)
def test_function(mock_call):
    mock_call.return_value = {"data": []}
    result = function_under_test("arg1", "arg2")  # Extra arg would raise TypeError
```

### Detection
```bash
# Check mock density (many mocks = potentially weak tests)
grep -rn "@mock\|@patch\|Mock()\|MagicMock" tests/ | wc -l

# Check for autospec usage
grep -rn "@patch" tests/ | grep -v "autospec"
```

---

## 20. Schema Drift & Data Pipeline Corruption

"Silent pipeline breaker" — partial data loads treated as success.

### Checklist

- [ ] **No schema validation on ingestion**: Accepting any shape
- [ ] **Partial data loads treated as success**: 60% of records loaded, no error
- [ ] **Column renames in source not detected**: Schema changed upstream
- [ ] **New enum values not handled**: `default: pass` on unknown values
- [ ] **Nullable fields becoming non-nullable**: Or vice versa
- [ ] **No data quality assertions**: Row counts, null checks, range checks
- [ ] **ETL "success" with empty results**: Zero rows returned, no warning

### Bad vs Good
```python
# BAD: No validation, silent corruption
def process_data(raw: dict) -> ProcessedData:
    return ProcessedData(
        user_id=raw.get("user_id"),  # Could be None
        score=raw.get("score", 0),   # 0 masks missing data
    )

# GOOD: Explicit validation with assertions
def process_data(raw: dict) -> ProcessedData:
    if "user_id" not in raw or raw["user_id"] is None:
        raise ValueError("Missing required field: user_id")
    if "score" not in raw:
        raise ValueError("Missing required field: score")
    return ProcessedData(user_id=raw["user_id"], score=raw["score"])
```

---

## 21. JSON Serialization Edge Cases

Crashes in production on edge case data that worked in dev.

### Checklist

- [ ] **`datetime` objects not serializable**: Works until one record has datetime
- [ ] **`Decimal` precision loss via float**: Financial calculations corrupted
- [ ] **`Enum` members not serializable**: `MyEnum.VALUE` fails
- [ ] **`UUID` objects not serializable**: Need `.hex` or `str()`
- [ ] **`dataclass` instances not handled**: Need custom encoder
- [ ] **`None` vs `"null"` string confusion**: Different semantics
- [ ] **Bytes objects in JSON payloads**: Need base64 encoding
- [ ] **Circular references causing infinite loops**: Object A references B references A

### Bad vs Good
```python
# BAD: Crashes on datetime
import json
data = {"created_at": datetime.now()}
json.dumps(data)  # TypeError: Object of type datetime is not JSON serializable

# GOOD: Custom encoder
def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json.dumps(data, default=json_serializer)

# BETTER: Use orjson which handles these natively
import orjson
orjson.dumps(data)
```

---

## 22. Static vs Runtime Type Mismatch

MyPy passes, runtime crashes — external data doesn't match type hints.

### Checklist

- [ ] **External data not validated**: JSON from API assumed to match types
- [ ] **`Any` type hiding errors**: `data: Any` bypasses all checks
- [ ] **Stub file divergence**: Type stubs don't match runtime behavior
- [ ] **Optional without None check**: `x: Optional[str]` then `x.upper()`
- [ ] **Type narrowing not preserved**: Check in one function, use in another
- [ ] **`cast()` without validation**: `cast(MyType, data)` trusts blindly
- [ ] **Forward references not resolved**: `"ClassName"` as string not evaluated

### Bad vs Good
```python
# BAD: MyPy trusts, runtime crashes
def process_response(data: UserData) -> str:
    return data.name.upper()

# Caller:
response = api.get("/user")  # Returns dict, not UserData!
process_response(response)   # MyPy: OK, Runtime: AttributeError

# GOOD: Validate at boundary
from pydantic import ValidationError

def process_response(raw: dict) -> str:
    try:
        data = UserData.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid API response: {e}")
    return data.name.upper()
```

### Detection
```bash
# Find cast() usage without validation
grep -rn "cast(" --include="*.py" .

# Find Any type usage
grep -rn ": Any\|-> Any" --include="*.py" .
```

---

## Quick Audit Commands

```bash
# === Core Quality Gates ===
uv run ruff check .           # Lint
uv run ruff format --check .  # Format
uv run mypy src/ --strict     # Type check
uv run pytest --cov           # Tests + coverage
uv run bandit -r src/         # Security scan

# === Exception Handling ===
grep -rn "except:" --include="*.py" src/
grep -rn "except Exception:" --include="*.py" src/ | grep -v "as"
grep -rn -A1 "except.*:" --include="*.py" src/ | grep "pass"

# === Truthiness Traps ===
grep -rn "if limit:" --include="*.py" src/
grep -rn "if not.*:" --include="*.py" src/ | grep -v "is not None"

# === Mutable Defaults ===
grep -rn "def.*=\[\]" --include="*.py" src/
grep -rn "def.*={}" --include="*.py" src/

# === Secrets Detection ===
grep -rn "api_key\|API_KEY\|password\|secret\|token" --include="*.py" src/ | grep -v "env\|getenv\|environ"
grep -rn "sk-\|pk_\|ghp_\|AKIA" --include="*.py" src/

# === Float Comparison ===
grep -rn "== 0\.\|!= 0\." --include="*.py" src/

# === SQL Injection Risk ===
grep -rn "execute.*f\"" --include="*.py" src/
grep -rn "execute.*%" --include="*.py" src/

# === Resource Leaks ===
grep -rn "open(" --include="*.py" src/ | grep -v "with "
grep -rn "\.connect(" --include="*.py" src/ | grep -v "with "

# === Incomplete Implementations ===
grep -rn "TODO\|FIXME\|XXX\|HACK" --include="*.py" src/
grep -rn "pass$" --include="*.py" src/
grep -rn "NotImplementedError" --include="*.py" src/

# === Mock Quality ===
grep -rn "@patch" tests/ | grep -v "autospec"

# === Type Safety ===
grep -rn "type: ignore" --include="*.py" src/
grep -rn ": Any\|-> Any" --include="*.py" src/
grep -rn "cast(" --include="*.py" src/

# === JSON Serialization Risk ===
grep -rn "json.dumps" --include="*.py" src/ | grep -v "default="
```

---

## References

### AI Code Generation
- [The Register: AI-authored code needs more attention](https://www.theregister.com/2025/12/17/ai_code_bugs/)
- [Dark Reading: Security Pitfalls of AI Code](https://www.darkreading.com/application-security/coders-adopt-ai-agents-security-pitfalls-lurk-2026)
- [arXiv: What's Wrong with LLM-Generated Code](https://arxiv.org/html/2407.06153v1)
- [Veracode: 2025 GenAI Code Security Report](https://www.veracode.com/blog/genai-code-security-report/)
- [arXiv: Importing Phantoms - LLM Package Hallucination](https://arxiv.org/html/2501.19012v1)

### Python Anti-Patterns
- [Hitchhiker's Guide: Common Gotchas](https://docs.python-guide.org/writing/gotchas/)
- [Inspired Python: Truthy and Falsy Gotchas](https://www.inspiredpython.com/article/truthy-and-falsy-gotchas)
- [Little Book of Python Anti-Patterns](https://docs.quantifiedcode.com/python-anti-patterns/)
- [charlax: Error Handling Anti-Patterns](https://github.com/charlax/professional-programming/blob/master/antipatterns/error-handling-antipatterns.md)
- [Real Python: Most Diabolical Python Antipattern](https://realpython.com/the-most-diabolical-python-antipattern/)

### ML Reproducibility
- [Princeton: Leakage and the Reproducibility Crisis](https://reproducible.cs.princeton.edu/)
- [Wiley: Reproducibility in ML-based Research](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)
- [CMU SEI: ML Reproducibility Myth](https://www.sei.cmu.edu/blog/the-myth-of-machine-learning-reproducibility-and-randomness-for-acquisitions-and-testing-evaluation-verification-and-validation/)
- [Ben Kuhn: Avoiding Bugs in ML Code](https://www.benkuhn.net/ml-bugs-2/)

### Type Safety
- [ToolShelf: Mastering Type-Safe Python 2025](https://toolshelf.tech/blog/mastering-type-safe-python-pydantic-mypy-2025/)
- [Pydantic: Strict Mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)
- [MyPy: Common Issues and Solutions](https://mypy.readthedocs.io/en/stable/common_issues.html)
- [Adam Johnson: Fix Circular Imports with TYPE_CHECKING](https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/)

### Concurrency
- [Medium: Avoiding Race Conditions in Python 2025](https://medium.com/pythoneers/avoiding-race-conditions-in-python-in-2025-best-practices-for-async-and-threads-4e006579a622)
- [Super Fast Python: RuntimeWarning Coroutine Never Awaited](https://superfastpython.com/asyncio-coroutine-was-never-awaited/)
- [Super Fast Python: Asyncio Race Conditions](https://superfastpython.com/asyncio-race-conditions/)

### Testing
- [Pytest with Eric: Common Mocking Problems](https://pytest-with-eric.com/mocking/pytest-common-mocking-problems/)
- [David Adamojr: AI-Generated Tests are Lying to You](https://davidadamojr.com/ai-generated-tests-are-lying-to-you/)

### Database & ORM
- [Medium: SQLAlchemy N+1 Hell Patterns](https://medium.com/@Modexa/10-sqlalchemy-relationship-patterns-that-dont-become-n-1-hell-9643dbc68712)
- [psycopg3: Connection Pool Docs](https://www.psycopg.org/psycopg3/docs/advanced/pool.html)

### Data Pipelines
- [Medium: When Successful Pipelines Quietly Corrupt Your Data](https://medium.com/towards-data-engineering/when-successful-pipelines-quietly-corrupt-your-data-4a134544bb73)
- [Medium: Schema Drift - The Silent Pipeline Breaker](https://medium.com/@adilshk047/schema-drift-the-silent-data-pipeline-breaker-how-it-happens-why-it-hurts-and-how-to-fix-it-3bf838662d3d)

### Security
- [OWASP: Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Endor Labs: Common Vulnerabilities in AI-Generated Code](https://www.endorlabs.com/learn/the-common-security-vulnerabilities-in-ai-generated-code)
- [Security Boulevard: Are Environment Variables Still Safe for Secrets?](https://securityboulevard.com/2025/12/are-environment-variables-still-safe-for-secrets-in-2026/)
- [OWASP: Top 10 LLM & Gen AI Vulnerabilities 2025](https://www.brightdefense.com/resources/owasp-top-10-llm/)

---

## Statistics (2025-2026 Research)

| Metric | Value | Source |
|--------|-------|--------|
| AI code logic errors vs human | 1.75x more | CodeRabbit 2025 |
| AI code security issues vs human | 1.57x more | CodeRabbit 2025 |
| AI code XSS vulnerabilities | 2.74x more | CodeRabbit 2025 |
| AI code insecure object refs | 1.91x more | CodeRabbit 2025 |
| AI code improper password handling | 1.88x more | CodeRabbit 2025 |
| Silent failures causing bugs | 40% of investigations | PSF Survey 2025 |
| Papers affected by data leakage | 648 papers | Princeton 2023 |
| Random seed accuracy variance | 44-45% range | CMU SEI |
| AI-generated code with vulns | 30-50% | IEEE/Academic 2025 |
| Live API keys found in public web | 11,908 | Security Research 2025 |
