# RL-Unit-Test

An intelligent framework for automated Python unit test generation and coverage optimization using Abstract Syntax Trees (AST), Control Flow Graphs (CFG), data-flow and path-feasibility analysis, a Decision Quality Matrix (DQM), and Reinforcement Learning (RL).

---

# Project Overview

RL-Unit-Test is a research-oriented software engineering project that automatically analyzes Python source code, constructs execution paths, evaluates path feasibility, generates concrete test inputs and pytest scenarios, executes those scenarios on the real target function, measures line and branch coverage, and applies Reinforcement Learning to optimize test selection.

The main optimization objectives are:

1. Reach the maximum achievable coverage over reachable program behavior.
2. Reach that coverage using the minimum possible number of test scenarios.

Coverage is the primary objective. Test count is minimized among suites that preserve the same coverage target.

The project is being developed as a modular and extensible architecture suitable for academic research and future industrial applications, including externally supplied Python files and Git repositories.

---

# Current Features

## Static Code Analysis

- Python AST Analysis
- Function Extraction
- Cyclomatic Complexity Analysis
- Branch Detection
- Type Hint Analysis
- Docstring Analysis
- Risk Level Detection

---

## Control Flow Analysis

- Control Flow Graph (CFG)
- Bounded Execution Path Discovery
- Configurable Per-Node Visit Limit
- Loop, `continue`, and Loop-Scoped `break` Flow Detection
- Nested-Loop Exit and Loop-`else` Semantics
- Exception Flow Analysis
- Stable Path Metadata

---

## Data-Flow and Path Feasibility

- Variable Definition and Update Tracking
- Local Symbolic State Propagation
- Numeric Range Inference
- Boolean and Collection-State Constraints
- Aggregate Collection-State Propagation
- Relational Constraint Analysis
- Variable-to-Variable Constraint Support
- Loop-State Consistency Checks
- Iteration-Scoped Loop Activation Analysis
- `FEASIBLE`, `INFEASIBLE`, and `UNKNOWN` Path Classification
- Contradictory Path Elimination

---

## Path Prioritization

- Decision Quality Matrix (DQM)
- Path Ranking
- Risk-Based Prioritization
- Normalized DQM Scores

---

## Scenario and Input Generation

- Automatic Scenario Generation
- Constraint-Aware Input Candidate Generation
- Stable Scenario IDs
- Scenario Metadata
- Priority Levels
- Execution Path Mapping
- Scenario-Action Mapping
- Typed Default Value Generation
- Primitive and Collection Type Support
- `list`, `tuple`, `set`, `Optional`, `Union`, and Nested Type Support
- Subscript Alias Propagation
- Loop-Variable Constraint Propagation
- Aggregate-Aware Derived-Value Input Synthesis
- Affine `while` Control-Variable Initialization
- Safe Optional Dictionary Lookup Synthesis
- Relational Witness Forwarding
- Dynamic Return-Value Evaluation
- Safe Built-in `round` Return Replay
- Safe Built-in `isinstance` Predicate Replay
- Safe f-String and Format-Spec Evaluation
- Stage- and Category-Aware Scenario Rejection Reporting
- Concrete Result and Exception Validation

---

## Reinforcement Learning

- Coverage State Representation
- State Encoder
- Extended State Key Encoding
- Executed Test Count Representation
- Action Model
- Q-Table
- Epsilon-Greedy Policy
- Dynamic Epsilon Control
- Q-Learning Agent
- Q-Learning Trainer
- Multi-Episode Training Sessions
- Training Statistics
- Best Episode Selection
- Configurable Target Coverage

---

## Real Coverage Environment

- Real pytest Execution
- Real Coverage Feedback
- Scenario Suite Coverage Transition
- Cumulative Scenario Execution
- Episode Reset Support
- Target Coverage Based Termination
- Reachable Coverage Handling
- Function-Level Coverage Evaluation
- File-Level Coverage Evaluation
- Line Coverage Tracking
- Branch Coverage Tracking

---

## Reward System

The RL reward system currently considers:

- Coverage improvement
- No-improvement penalties
- Test execution cost
- Full/target coverage reward
- Efficient test selection

This allows the agent to optimize not only coverage growth but also the number of tests required to reach the target.

---

## Test Generation

- Automatic pytest Generation
- Stable Test Naming
- Metadata Generation
- Cumulative Test Suite Generation
- Minimized Suite Export Support

---

## Test Automation

- Automatic Test Execution
- Concrete Scenario Validation
- Execution Summary
- Timeout Handling
- Error Reporting
- Real Coverage Measurement

---

## Coverage Analysis

- Line Coverage
- Branch Coverage
- Function-Level Coverage
- File-Level Coverage
- Missing Line Detection
- Uncovered Branch Detection
- Coverage Reports
- Coverage Workflow
- Exact Combined Project Coverage over the Analyzed Scope
- Module-Qualified Line and Branch Identities
- Project-Level Greedy Suite Minimization with Exact Replay Verification

---

## Training and Reporting

- Multi-Episode Training
- Episode Statistics
- Total and Average Reward
- Best Episode Detection
- Best Coverage Detection
- Executed Test Count Tracking
- Q-Table State Statistics
- Function Coverage Summary
- File Coverage Summary
- Human-Readable Training Reports
- JSON Reports
- DQM Reports
- Persistent Pipeline Diagnostic Results
- Funnel Snapshots and Structured Rejection Summaries
- Explicit `COMPLETED`, `PARTIAL`, `FAILED`, and `TIMED_OUT` States
- Unavailable Coverage Metrics Reported as Unmeasured Instead of `0%`

---

## Pipeline Diagnostics and Global Timeout

- Atomic JSON Diagnostic Checkpoints
- Partial Results Preserved Across Global Timeout
- Global Orchestration Deadline at the Service API Boundary
- Windows `spawn` Worker Isolation
- Run-Scoped Worker Process-Tree Cleanup
- Deterministic Worker Process-Handle Closure
- Separate Per-Test and Pipeline-Wide Timeout Configuration

`timeout_seconds` remains the pytest/coverage execution timeout. `pipeline_timeout_seconds` is the independent deadline for the complete service pipeline. Checkpoints contain aggregate stage, funnel, rejection, and coverage metadata; they do not persist generated inputs, expected/actual values, source code, or tracebacks.

```text
Pipeline worker
      |
      v
Diagnostic checkpoint
      |
      +--> Completed result
      |
      +--> Timeout watchdog
                |
                v
         Worker tree cleanup
                |
                v
        TIMED_OUT partial result
```

Global timeout is integrated into both the service API and the interactive project-analysis workflow. Multi-function source orchestration applies this limit independently to each supported function.

---

## Interactive Developer Tool

Running `python main.py` opens a production-oriented menu:

```text
RL-UNIT-TEST DEVELOPER TOOL

1. Kaynak Kod / Proje Analizi
2. Hızlı Statik Ön İnceleme
3. Dış Kaynak Analizi
0. Çıkış
```

Option 1 asks for a real Python source file, module path, single/all-functions selection, output directory, and optionally advanced production settings. There is no implicit `datasets/sample_code.py` or `calculate_score` fallback. Example datasets remain valid only when the user selects them explicitly.

Option 2 asks for a real Python source file and reports AST function, complexity, and production-support metadata without starting test generation, coverage, or RL training.

Option 3 opens a dedicated external-source submenu. Each entry remains a separate request model; the terminal never guesses whether one ambiguous string is source code, a path, or a URL:

```text
DIŞ KAYNAK ANALİZİ

1. Python kodu yapıştır
2. Python dosyası seç
3. Yerel proje klasörü seç
4. Public GitHub URL gir
0. Ana menüye dön
```

Pasted Python is collected as multiline input until a line containing only `__END__`; the marker is not included in the source. Empty input is rejected. File selection accepts an explicit existing `.py` path, while local projects require an existing directory. GitHub URLs use the existing acquisition-service validation and are never combined with paste or upload input.

External analysis defaults to `STATIC_DISCOVERY_ONLY`. `TRUSTED_DYNAMIC_ANALYSIS` requires a separate explicit `EVET` confirmation after warning that source code will execute and that timeout is not a sandbox. Dynamic settings expose output root, module-selection mode, module/function limits, per-function pipeline timeout, episode count, random seed, greedy minimization, and RL–greedy comparison. Enabling comparison also enables its production greedy baseline.

The research and diagnostic operations remain available through the advanced CLI: `analyze`, `cfg`, `dqm`, `dqm-json`, `test`, `coverage`, `demo`, and `rl`. Non-interactive `rl` requires explicit source, module, and single/all-functions selection arguments.

---

## Multi-Function Production Orchestration

- Deterministic AST-Based Function Discovery in Source Order
- Explicit Support Metadata for Top-Level, Async, Nested, Method, Conditional, Dunder, and Ambiguous Targets
- Single-Function and All-Functions Selection Modes
- Function-Isolated `COMPLETED`, `PARTIAL`, `FAILED`, `TIMED_OUT`, and `UNSUPPORTED` Results
- Separate, Traversal-Safe Output Directories per Function
- Immutable Project Result with Atomic JSON Reporting
- Project-Level Status and Stable CLI Exit Codes
- Function Coverage Reported Individually; Aggregate Project Coverage Marked Unmeasured

```text
Python source
      |
      v
Function discovery
      |
      v
Per-function isolated pipeline
      |
      +--> Completed result
      +--> Partial/failed result
      +--> Timed-out result
      |
      v
Project JSON report
```

Single-function CLI example:

```text
python main.py --operation rl --source-file datasets/sample_robustness_code.py --module-path datasets.sample_robustness_code --function-name calculate_category_usage --pipeline-timeout-seconds 30
```

All-functions CLI example:

```text
python main.py --operation rl --source-file datasets/sample_robustness_code.py --module-path datasets.sample_robustness_code --all-functions --pipeline-timeout-seconds 30
```

`timeout_seconds` remains the pytest/coverage subprocess limit. `pipeline_timeout_seconds` is applied independently to each selected function. A separate project-wide deadline is not implemented; without one, every selected target is attempted in deterministic source order.

---

## Secure Source Acquisition and Python Project Discovery

The service layer can now resolve three source kinds without importing or executing project code:

- A local Python file
- A local project directory
- A public GitHub repository using an exact `https://github.com/<owner>/<repository>` URL

```text
Local Python file ---------+
                           |
Local project directory ---+--> Source acquisition policy
                           |       |
Public GitHub HTTPS URL ---+       +--> Resource and path limits
                                   +--> Symlink/junction containment
                                   +--> Deterministic Python discovery
                                   +--> Module-path candidates
                                   +--> Immutable JSON-safe inventory
```

Local discovery supports top-level modules, regular packages, normalized package `__init__.py` paths, and conventional `src/` layouts. Ambiguous or non-importable layouts are reported instead of guessed. PEP 263 encoding detection and per-file AST parsing classify unsupported encodings and syntax errors without allowing one broken file to hide valid modules. Test discovery is controlled by the explicit `include_tests` request option.

Ignored directories come from one general policy covering Git metadata, virtual environments, caches, build outputs, installed dependencies, `node_modules`, and generated output. Discovery does not follow symlinks, junctions, or reparse points, and every candidate is checked against the resolved project root. Configurable limits cover clone time, repository bytes, Python file count, single-file bytes, total Python bytes, and path depth.

Public GitHub acquisition uses a unique tool-owned system-temp workspace and a shallow, single-branch, no-tags clone. Git runs with `shell=False`, credential prompting and helpers disabled, process-local Git configuration, disabled LFS smudge, no submodule recursion, and a disabled hook path. Clone output is discarded; only the validated 40-character commit SHA is captured. Failed or timed-out clones clean their partial workspace, while local user-owned sources are never cleanup targets. No dependency manifest, setup script, package manager, or repository code is executed.

The real local-project acceptance discovered top-level, package, and `src/` modules, excluded tests by default, included them on request, ignored `.venv`, and preserved the user-owned workspace. A real temporary local Git fixture verified commit-SHA resolution. Anonymous `ls-remote` access succeeded for both the current origin and GitHub's official small public connectivity fixture. The single official resolver acceptance completed the clone in 1.133 seconds and resolved commit `7fd1a60b01f91b314f59955a4e4d4e80d8edf11d`. The fixture contained no Python files; this is reported as a non-fatal `NO_PYTHON_FILES` partial inventory rather than `CLONE_FAILED`. That run exposed Windows read-only Git pack cleanup behavior, which now has a regression-tested writable-retry path; the tool-owned workspace was removed and no Git process remained.

---

## External Source-to-Analysis Integration

The backend keeps four input modes distinct: `INLINE_PYTHON_SOURCE`, `UPLOADED_PYTHON_FILE`, `LOCAL_PROJECT_DIRECTORY`, and `PUBLIC_GITHUB_REPOSITORY`. Inline text, uploaded bytes, local paths, and GitHub URLs are not multiplexed through one ambiguous field. Module selection supports deterministic `ALL_ELIGIBLE_WITH_LIMIT`, `EXPLICIT_RELATIVE_PATHS`, and `EXPLICIT_MODULE_NAMES` policies, with configurable module and per-module function limits. Discovered modules and functions beyond those limits remain visible in reports as `SKIPPED_LIMIT` instead of disappearing from the inventory.

```text
Inline Python -----------+                         +--> Static metadata only
Uploaded .py ------------+--> Secure acquisition -+
Local project directory -+          |              +--> Trusted dynamic analysis
Public GitHub URL -------+          v                        |
                               Module selection              v
                                      |             Multi-function pipeline
                                      |               +--> pytest/coverage
                                      |               +--> greedy minimization
                                      +-------------->+--> RL vs greedy report
```

`STATIC_DISCOVERY_ONLY` is the default and never starts pytest, coverage, greedy minimization, or RL. Real execution requires the caller to select `TRUSTED_DYNAMIC_ANALYSIS` explicitly. A timeout bounds work; it is not a security sandbox. Trusted external code can still access host resources, so arbitrary untrusted execution remains unsupported.

Inline and upload payloads are byte-limited, encoding/syntax checked, written to unique tool-owned system-temp files, retained through analysis, and cleaned afterward. Their source text and bytes are never serialized. Local directories remain user-owned. Public GitHub sources retain the secure anonymous clone and commit-SHA policy, and their tool-owned clone survives until persistent artifacts and the atomic external report are written.

Dynamic analysis reuses the production `SourceAnalysisOrchestrator` and creates fresh per-module/per-function service state. The validated project root or `src` root is passed only to the isolated worker and coverage subprocess. Coverage uses that root as `cwd` and as the complete run-specific `PYTHONPATH`; the parent process path and import cache are restored. Dependency installation is never attempted. Missing dependencies and import failures become safe per-module results without raw tracebacks, environment data, credentials, kwargs, or expected/actual values in the external JSON.

The real inline acceptance produced a two-scenario pool, exact 100% line and branch coverage, a two-test greedy suite, and a two-test RL suite; exact coverage equality was verified and the comparison result was `TIE`. Real uploaded-file and local multi-module/package-relative-import acceptances also completed, persisted artifacts outside tool temp, preserved the local project, cleaned tool-owned workspaces, and left the parent `sys.path` unchanged. The function-limit acceptance discovered three eligible functions, executed the first two in source order, retained the third as `SKIPPED_LIMIT`, and reported the project as `PARTIAL`. The 38.3 interactive adapter acceptance additionally verified separate terminal source kinds, static-by-default requests, explicit trusted confirmation, multiline paste termination, state isolation, interrupt cleanup, and internal `ValueError` propagation. The sprint did not repeat a public network clone because the prior verified public fixture contained no eligible Python module for a safe dynamic acceptance. Current regression: `1963 passed, 1 skipped in 271.13s`.

---

## FastAPI Asynchronous Analysis Jobs

The `api.app.create_app(...)` application factory exposes an import-safe asynchronous job backend. Importing the API does not start Uvicorn, a worker, or an analysis pipeline. Uvicorn starts only through the guarded `api.server` entry point.

Source submission remains explicit and separate:

- `POST /api/v1/jobs/inline` accepts inline Python JSON.
- `POST /api/v1/jobs/upload` accepts one multipart `.py` upload.
- `POST /api/v1/jobs/github` accepts a public GitHub repository URL.
- No API endpoint accepts an arbitrary local server filesystem path. Local project selection remains terminal-only.

```text
Inline JSON -----+
Multipart .py ---+--> Bounded job queue --> Worker thread
GitHub HTTPS ----+          |                    |
                            |                    +--> Existing external analysis service
                            |                           +--> Static discovery by default
                            |                           +--> Trusted dynamic pipeline
                            v
                  QUEUED -> RUNNING -> COMPLETED / PARTIAL / FAILED / TIMED_OUT
                            |
                            +--> Safe polling result
                            +--> Opaque artifact IDs
                            +--> Contained JSON / generated-pytest downloads
```

`GET /api/v1/jobs/{job_id}` returns safe status and progress metadata. The `/result` endpoint returns `409` until a result exists, `/artifacts` lists allowlisted persistent artifacts, and `/artifacts/{artifact_id}` downloads by server-generated opaque ID without accepting client paths. `GET /api/v1/health` reports bounded queue capacity without running a pipeline.

The backend limits concurrent and queued jobs, inline/upload bytes, and terminal-job retention. Expired terminal records and their server-owned output directories are removed lazily and idempotently; running jobs and user-owned sources are not retention cleanup targets. Raw source, upload bytes, kwargs, expected/actual values, tracebacks, credentials, environment values, and tool-temp paths are not stored in public job models.

`STATIC_DISCOVERY_ONLY` remains the default. `TRUSTED_DYNAMIC_ANALYSIS` requires both `trusted_execution_acknowledged: true` and a positive per-function pipeline timeout. Strategy comparison enables its production greedy baseline. Timeout is not a security sandbox; dynamic execution is only for trusted sources.

Queued jobs can be cancelled before execution. Running jobs are not falsely reported as cancelled because the production pipeline does not expose a safe job-level worker termination handle; a cancellation request is recorded and the API returns `409` while existing per-function timeout protection remains active. Terminal jobs cannot be cancelled again.

CORS is disabled unless an explicit origin allowlist is supplied. Anonymous public GitHub acquisition uses the existing resolver; private credentials and dependency installation are unsupported. Authentication and rate limiting are not implemented yet.

## Production Web UI

The FastAPI backend now serves a responsive Turkish developer interface from local HTML, CSS, and framework-independent JavaScript. No Node build chain, CDN, remote font, analytics, or browser runtime dependency is required. Start the localhost server with:

```text
python -m api.server
```

Then open `http://127.0.0.1:8000/`. The server remains bound to localhost by default. `/docs`, `/redoc`, `/openapi.json`, and all `/api/v1/...` endpoints remain available.

```text
Kod Yapıştır --------+
Python .py upload ---+--> Aynı-origin FastAPI job API
Public GitHub URL ---+          |
                                +--> QUEUED / RUNNING polling
                                +--> Terminal result metadata
                                +--> Scenario-pool target coverage
                                +--> Greedy verified coverage
                                +--> Best-RL verified coverage
                                +--> Backend strategy comparison
                                +--> Opaque-ID artifact download
```

The three browser inputs are separate tabs and map directly to the inline, multipart upload, and public GitHub endpoints. The browser does not guess whether one value is code, a path, or a URL. Server filesystem and local project-directory input is intentionally absent from the Web UI; local project analysis remains available through the terminal workflow.

`STATIC_DISCOVERY_ONLY` is selected by default. `TRUSTED_DYNAMIC_ANALYSIS` reveals a visible execution warning and requires a separate acknowledgement before submission. Timeout limits work but is not a sandbox; dynamic mode must only be used for trusted code. Advanced settings expose the existing API module-selection modes, module/function limits, episode count, deterministic random seed `42`, pytest/coverage timeout, per-function pipeline timeout, greedy minimization, and RL–greedy comparison without duplicating analysis logic in the browser. Greedy minimization is enabled by default in the Web UI because it is the primary minimum-suite baseline; the more expensive RL–greedy comparison remains optional. The backward-compatible API and service defaults are unchanged.

The page polls one opaque job ID at a time with bounded retry/backoff and stops at `COMPLETED`, `PARTIAL`, `FAILED`, `TIMED_OUT`, or `CANCELLED`. Queued cancellation is distinguished from the `409` response for a running job. Dynamic function cards keep three measured results distinct: the concrete-valid full scenario pool is the attainable target, greedy is the independently verified minimized suite, and best RL is the actual coverage snapshot of the highest-ranked episode. A lower RL result never replaces the attainable target. Greedy and RL preservation are reported against exact line and branch identities, and a non-preserving greedy candidate is not presented as a successful minimum. `null` remains `Ölçülmedi`, and the UI does not calculate its own strategy winner. A separate Project Coverage section renders only the backend's combined project result; it never derives project coverage from function percentages. Generated pytest and JSON reports are listed and downloaded only through server-issued artifact IDs.

The UI uses DOM text APIs for external values and does not place source code in the URL, browser storage, logs, or result surface. Static assets are contained under the fixed `/static` mount. Responses add a self-only Content Security Policy, `nosniff`, no-referrer, frame denial, and a restrictive permissions policy. CORS stays disabled by default. Authentication and rate limiting remain unimplemented and are part of production hardening rather than completed UI functionality.

The automated Web UI contract and real localhost HTTP acceptance cover static and trusted-dynamic submission, upload, separate GitHub routing, polling, measured coverage, backend-authoritative `TIE` comparison, JSON/generated-pytest downloads, source non-reflection, temp cleanup, and orphan-server cleanup. The project-level result is exposed through the same asynchronous job result and artifact endpoints; static discovery keeps it `null`.

A controlled trusted-dynamic acceptance of `sample_complex_code.py` used the Web production configuration (`max_visits_per_node=3`, three episodes, epsilon `0.0`, learning rate `0.5`, discount factor `0.9`, seed `42`, 30-second coverage timeout, 120-second function timeout, greedy and comparison enabled):

| Function | Pool | Pool line / branch | Greedy tests | Greedy line / branch | Preserved | Best RL tests | Best RL line / branch | Preserved | Winner |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `calculate_score` | 9 | 95.00% / 94.44% | 9 | 95.00% / 94.44% | Yes | 9 | 95.00% / 94.44% | Yes | `TIE` |
| `calculate_letter_grade` | 8 | 100% / 100% | 8 | 100% / 100% | Yes | 8 | 100% / 100% | Yes | `TIE` |
| `calculate_bonus` | 10 | 100% / 100% | 4 | 100% / 100% | Yes | 6 | 100% / 100% | Yes | `GREEDY` |
| `calculate_average` | 7 | 100% / 100% | 3 | 100% / 100% | Yes | 4 | 100% / 100% | Yes | `GREEDY` |
| `evaluate_student` | 5 | 91.67% / 90.00% | 5 | 91.67% / 90.00% | Yes | 5 | 91.67% / 90.00% | Yes | `TIE` |

The pool itself does not reach 100% for `calculate_score` (missing line `22`, branch arc `19→22`) or `evaluate_student` (missing line `141`, branch arc `140→141`). These are reported as current generation/reachability limits; neither RL nor greedy is allowed to inflate the attainable target.

## Exact Combined Project Coverage and Minimization

Trusted dynamic analysis now performs a real combined pytest/coverage execution after the selected functions have produced concrete-valid scenarios. Function-level percentages are never averaged. The primary project metric is explicitly named `ANALYZED_PROJECT_SCOPE_COVERAGE`; whole-repository line and branch coverage remain `null` unless a separate whole-repository measurement exists.

Exact identities are module-qualified and serialized with normalized `/` paths:

- line: `(relative_module_path, line_number)`
- branch: `(relative_module_path, from_line, to_line)`

Coverage.py negative branch targets are retained because they represent real function-exit arcs. Absolute source and tool-temporary paths are not included in the public result. The full combined suite's actually covered exact identity set is the minimization target, so an attainable target below 100% is reported honestly rather than inflated.

```text
Concrete-valid scenarios from selected functions
                    |
                    v
       Full combined project pytest
                    |
                    v
   Exact module-qualified coverage target
                    |
                    v
  Per-scenario project contribution measurement
                    |
                    v
 Deterministic greedy set cover
   1. new total identities
   2. new branch identities
   3. new line identities
   4. original project order
   5. stable project test ID
                    |
                    v
 Backward redundancy elimination
                    |
                    v
 Minimized combined pytest + exact replay verification
```

The result records discovered and selected modules; discovered, eligible, analyzed, completed, partial, failed, timed-out, unsupported, and `SKIPPED_LIMIT` functions; and an explicit `scope_complete` flag. Completed subsets remain measurable, but missing, unsupported, failed, timed-out, or limited targets make the result `PARTIAL`. Interaction-dependent coverage that isolated contributions cannot explain is reported as `NON_ADDITIVE_COVERAGE`; the full suite remains the safe verified fallback. The deterministic greedy result always declares `globally_minimal = false`.

Artifacts are persisted under `project_combined/` as the full generated pytest, minimized generated pytest, and machine-readable project coverage/minimization JSON. They are included in external analysis output, asynchronous API result/artifact contracts, terminal reporting, and the Web UI's separate Project Coverage section. Public JSON contains exact identities, aggregate counts, controlled status/failure metadata, and relative artifact paths, but no raw source, kwargs, expected/actual values, tracebacks, credentials, environment values, or absolute tool paths.

The real acceptance set includes a two-module, three-function temporary package with a relative import and redundant loop-path candidates, plus all five functions in `datasets/sample_complex_code.py`. The multi-module full suite contained 6 tests and was reduced to 5 (16.67%) while preserving 100% line and branch coverage with `scope_complete=true`. The five-function sample full suite contained 39 tests and was reduced to 29 (25.64%); both suites reported 97.30% line and 96.67% branch coverage and identical 72-line/58-branch exact targets. All full and minimized pytest runs exited `0`. A function-limit acceptance preserves the completed subset while reporting incomplete scope instead of whole-project success. Day 40.2 remains responsible for final real-browser E2E, broader production hardening, and release reporting.

---

# Architecture

```text
Python Source
      │
      ▼
Python AST Analyzer
      │
      ▼
Control Flow Graph Builder
      │
      ▼
Execution Path Analyzer
      │
      ▼
Data-Flow Analyzer
      │
      ▼
Path-State Analyzer
      │
      ▼
Path-Feasibility Analyzer
      │
      ▼
Input Candidate Generator
      │
      ▼
Decision Quality Matrix
      │
      ▼
Scenario Generator
      │
      ▼
Concrete Scenario Validation
      │
      ▼
Scenario Action Mapper
      │
      ▼
Coverage Environment
      │
      ▼
State Encoder
      │
      ▼
Q-Learning Agent
      │
      ├── Q-Table
      └── Epsilon-Greedy Policy
      │
      ▼
Scenario Selection
      │
      ▼
Pytest Generation
      │
      ▼
Automatic Test Execution
      │
      ▼
Real Coverage Measurement
      │
      ▼
Reward Calculation
      │
      ▼
Q-Value Update
      │
      ▼
Next Training Step / Episode
```

The feasibility and concrete-validation layers prevent contradictory or behaviorally incorrect scenarios from entering the RL action pool. The RL agent therefore learns from scenarios that are both statically supported and executable on the real target function.

---

# Technologies

- Python 3.14
- Python AST
- Pytest
- coverage.py
- JSON
- FastAPI
- Local HTML, CSS, and Framework-Independent JavaScript
- Control Flow Graph (CFG)
- Decision Quality Matrix (DQM)
- Data-Flow Analysis
- Symbolic Path-State Analysis
- Reinforcement Learning
- Q-Learning

### Future Integrations

- Tree-sitter
- Isolated Project Execution

---

# Current Development Status

## Static Analysis

- ✅ AST Analyzer
- ✅ Function Analyzer
- ✅ Type Hint Analyzer
- ✅ Complexity Metrics

## CFG

- ✅ CFG Builder
- ✅ Execution Path Analyzer
- ✅ Configurable Path Visit Limit
- ✅ Loop, Loop-Scoped `break`, and Exception Flow Analysis
- ✅ Nested-Loop Exit and Loop-`else` Semantics

## Data Flow and Feasibility

- ✅ Data Flow Analyzer
- ✅ Path State Analyzer
- ✅ Path Feasibility Analyzer
- ✅ Relational Constraint Support
- ✅ Aggregate Collection-State Feasibility
- ✅ Iteration-Scoped Loop Activation Analysis
- ✅ Contradictory Path Filtering

## DQM

- ✅ Decision Quality Matrix
- ✅ Path Prioritization

## Scenario Generation

- ✅ Input Candidate Generator
- ✅ Typed Path Input Generator
- ✅ Loop and Alias-Aware Input Generation
- ✅ Aggregate-Aware Derived-Value Input Synthesis
- ✅ Affine `while` Control-Variable Initialization
- ✅ Safe Optional Dictionary Lookup Synthesis
- ✅ Dynamic Expected Result Generation
- ✅ Safe Built-in `round` Return Replay
- ✅ Safe Built-in `isinstance` Predicate Replay
- ✅ Scenario Generator
- ✅ Unsupported Path Failure Isolation and Rejection Reporting
- ✅ Scenario Action Mapper
- ✅ Concrete Scenario Validation

## Reinforcement Learning

- ✅ Coverage Environment
- ✅ Reward Calculator
- ✅ State Encoder
- ✅ Extended State Representation
- ✅ Q-Table
- ✅ Epsilon-Greedy Policy
- ✅ Dynamic Epsilon Support
- ✅ Q-Learning Agent
- ✅ Q-Learning Trainer
- ✅ Multi-Episode Training
- ✅ Training Statistics
- ✅ Best Episode Selection

## Real Coverage Integration

- ✅ Scenario Suite Coverage Transition
- ✅ Real pytest Execution
- ✅ Real Coverage Feedback
- ✅ Target Coverage Support
- ✅ Reachable Coverage Handling
- ✅ Function-Level Coverage Evaluation
- ✅ File-Level Coverage Evaluation
- ✅ Exact Combined Analyzed-Project-Scope Coverage
- ✅ Module-Qualified Exact Line and Branch Identities
- ✅ Project-Level Greedy Suite Minimization and Replay Verification

## Test Generation

- ✅ Pytest Generator
- ✅ Test File Writer
- ✅ Cumulative Scenario Suite Generation

## Automation

- ✅ Test Execution
- ✅ Coverage Workflow
- ✅ End-to-End RL Training Service
- ✅ Configurable Path Expansion in the Production Service
- ✅ Persistent Partial Pipeline Diagnostics
- ✅ Service-Level Global Orchestration Timeout
- ✅ Windows Worker-Tree and Process-Handle Cleanup
- ✅ Multi-Function Source Orchestration
- ✅ `main.py` Option 1 Production Integration
- ✅ Interactive Static Source Preview
- ✅ Atomic Project JSON Report
- ✅ Four-Mode External Source Analysis API
- ✅ Static-Only Default and Explicit Trusted Dynamic Policy
- ✅ Module/Function Selection Limits and Import-Root Isolation
- ✅ External Coverage, Greedy, and RL Comparison Reporting
- ✅ Combined Project Coverage in Terminal, JSON, API, Artifacts, and Web UI

---

# Experimental Evaluation

The RL pipeline is evaluated on increasingly complex datasets rather than relying only on the initial sample program. Results from different target functions are reported separately.

## `evaluate_application` Benchmark

An earlier experiment on the `evaluate_application` target produced:

| Metric | Result |
| --- | ---: |
| Generated scenarios | 36 |
| Reachable maximum line coverage | 98.65% |
| Reachable maximum branch coverage | 98.61% |
| Highest single-scenario coverage | 24.32% |
| Lowest single-scenario coverage | 4.05% |
| Average single-scenario coverage | 19.40% |

The experiment demonstrated that no individual scenario could approach the reachable maximum. Multiple scenarios were required to cover different execution paths. Leave-one-out and greedy checks also showed that all 36 scenarios in the available pool were required to preserve the measured maximum.

## `process_order` Ultracomplex Benchmark

The `datasets/sample_ultracomplex_code.py::process_order` target was used to expose limitations in loop expansion, feasibility analysis, relational input generation, expected-result evaluation, and concrete scenario validation.

Before the latest revisions:

| Metric | Result |
| --- | ---: |
| Execution paths | 619 |
| FEASIBLE paths | 25 |
| INFEASIBLE paths | 594 |
| UNKNOWN paths | 0 |
| Concrete scenario pool | 14 |
| Function line coverage | 52.24% |
| Function branch coverage | 48.08% |

The diagnostic showed that paths after a local two-iteration `while` loop required a third visit to the loop condition to model the final exit check.

After integrating a configurable `max_visits_per_node` value with a production default of `3`, improving generic input generation, and validating generated scenarios against the real target function, an earlier checkpoint produced:

| Metric | Result |
| --- | ---: |
| Execution paths | 2,128 |
| FEASIBLE paths | 340 |
| INFEASIBLE paths | 1,788 |
| UNKNOWN paths | 0 |
| Concrete scenario pool | 60 |
| Scenarios executed in the measured episode | 51 |
| Function line coverage | 83.58% |
| Function branch coverage | 84.62% |
| File line coverage | 68.60% |
| File branch coverage | 66.67% |

The current checkpoint, after reachability alignment, aggregate-state feasibility, derived-value synthesis, iteration-scoped loop analysis, and safe replay improvements, is:

| Metric | Result |
| --- | ---: |
| Concrete scenario pool | 165 |
| Scenarios executed in the measured RL episode | 144 |
| Function line coverage | 88.06% |
| Function branch coverage | 92.31% |
| Feasible but uncovered source lines | None |
| Unresolved reachability lines | None |
| Infeasible-only lines | Present; bounded-analysis-specific |

The improvement was achieved without modifying the benchmark source code and without adding `process_order`-specific rules to the framework. `INFEASIBLE_ONLY` remains a bounded-analysis classification: it describes the explored path space under the configured visit limit rather than proving global semantic unreachability.

## Robustness Benchmarks

Recent robustness work extended the generic pipeline with aggregate-state feasibility, derived-value input synthesis, affine `while` initialization, loop-scoped `break` control flow, safe built-in replay, optional dictionary lookup synthesis, and isolated handling of unsupported expected-result paths.

| Target | Pipeline result | Bounded paths | Concrete validation / suite progress | Function line coverage | Function branch coverage |
| --- | --- | ---: | --- | ---: | ---: |
| `calculate_category_usage` | Completed | 3 | 3 | 100% | 100% |
| `determine_transaction_risk` | Completed | 5 | 4 | 90% | 87.5% |
| `analyze_transactions` | Timeout / partial | 24,560 | At least 96 reached the temporary concrete suite | Not finalized | Not finalized |

For `determine_transaction_risk`, all five bounded paths were classified as feasible. Four became concrete-valid scenarios, while one was isolated as an `UNREACHABLE_INPUT` rejection instead of aborting scenario generation. `FEASIBLE_UNCOVERED 215` refers to source line 215, not to a count of uncovered paths.

The `analyze_transactions` production run reached the externally enforced 180-second orchestration timeout. The observed 96 tests are an intermediate concrete-suite count, not a final scenario-pool size and not an RL executed-test count. Because orchestration did not complete, no final line or branch coverage estimate is reported. No new expression exception was observed before timeout. The new service-level global timeout has not yet been exercised on this heavy target, so no such result is claimed. The remaining blockers are bounded-path explosion and final CLI/orchestration integration; the two smaller robustness targets complete successfully.

The current small production acceptance run for `calculate_category_usage` completed with 3 bounded paths, a 3-scenario pool, 3 RL-executed tests, 3 Q-table states, 100% function line coverage, 100% function branch coverage, and a `COMPLETED` pipeline diagnostic. The optional greedy baseline measured all three scenarios independently and preserved the exact six-line/four-branch-arc target in a separately generated pytest suite. Every scenario had a unique contribution, so the honest result was `NO_REDUCTION` with 0% reduction.

The 36.3 production CLI acceptance also completed `determine_transaction_risk` with 4 scenarios, 4 RL-executed tests, 4 Q-table states, 90% function line coverage, and 87.50% function branch coverage. A separate real three-function fixture preserved two `COMPLETED` results around one controlled `PARTIAL` result and produced an atomic project JSON report. The heavy `analyze_transactions` all-functions run was intentionally not repeated; its final coverage remains unmeasured.

---

# Current Optimization Problem

The framework can now generate a substantially larger validated scenario pool for the ultracomplex target. The current optimization problem is test-suite minimization.

Current situation:

```text
165 validated scenarios
        │
        ▼
RL scenario selection
        │
        ▼
144 executed scenarios
        │
        ▼
88.06% line coverage / 92.31% branch coverage
```

The deterministic baseline now determines whether the measured validated-pool coverage can be preserved by a smaller subset:

```text
165 available scenarios
        │
        ▼
Scenario contribution analysis
        │
        ├── Deterministic greedy set cover
        └── Raw RL episode selection
        │
        ▼
Independent real pytest/coverage verification
        │
        ▼
Exact line/branch equality and winner policy
```

Each scenario is executed independently to obtain exact function-scoped line identities and coverage.py branch arcs. The selector uses deterministic greedy set cover, then applies backward redundancy elimination. The selected scenarios are exported under `greedy_minimized/` and executed together in a final real pytest/coverage verification; equal percentages with different identities are not accepted as preservation.

The implementation explicitly reports:

- `algorithm = GREEDY_SET_COVER_WITH_BACKWARD_ELIMINATION`
- `globally_minimal = false`
- `coverage_preserved = true/false`
- controlled timeout, isolated-execution failure, and non-additive coverage states

A real temporary multi-branch minimization acceptance reduced a three-scenario pool to two scenarios while preserving the exact line and branch-arc sets. The exact strategy-comparison acceptance used a separate branch-heavy four-scenario pool: greedy and the best raw RL episode each used three tests, both reproduced the full-pool exact identities, and the measured winner was `TIE`.

Strategy comparison keeps the existing training-session best-episode policy unchanged. It reconstructs immutable ordered scenario traces, verifies candidate raw RL suites with the same function-scoped coverage service and target used by greedy, and selects the comparison episode by exact preservation, action count, unique count, reward, then episode number. It does not apply greedy elimination to the raw RL result. A strategy cannot win on test count while missing any target line or branch arc, and every result keeps `globally_minimal = false`.

The real `determine_transaction_risk` acceptance used four validated scenarios and three deterministic RL episodes. All three episodes reached the exact nine-line/seven-branch-arc target. Greedy selected four tests, raw RL selected four tests, both preserved 90% line and 87.50% branch coverage with identical identities, and the result was `TIE` with 0% reduction for both strategies. The 165-scenario ultracomplex comparison was not run because contribution measurement still requires a coverage subprocess per scenario; performance and caching work is required before that experiment.

---

# Test Status

The project contains an extensive automated test suite covering individual analysis and RL components, regression cases, integration flows, and end-to-end real coverage training.

Latest full regression run:

| Test result | Status |
| --- | ---: |
| Passed | 1,963 |
| Failed | 0 |
| Skipped | 1 (Windows symlink creation unavailable) |
| Duration | 271.13s |

---

# Project Structure

```text
RL-Unit-Test
│
├── analyzer
├── cfg
├── evaluator
├── generator
├── rl
├── services
├── api
├── web
├── tests
├── output
├── datasets
└── main.py
```

---

# Development Roadmap

## Completed

- Python AST Analysis
- CFG Construction
- Bounded Execution Path Analysis
- Decision Quality Matrix
- Data-Flow Analysis
- Path-State Analysis
- Path-Feasibility Classification
- Contradictory Path Filtering
- Typed and Constraint-Aware Input Generation
- Collection, Alias, and Loop-Variable Input Generation
- Aggregate Collection-State Feasibility
- Aggregate-Aware Derived-Value Input Synthesis
- Affine `while` Control-Variable Initialization
- Safe Optional Dictionary Lookup Synthesis
- Iteration-Scoped Loop Activation Analysis
- Relational Witness Integration
- Dynamic Return and f-String Evaluation
- Safe Built-in `round` Return Replay
- Safe Built-in `isinstance` Predicate Replay
- Loop-Scoped `break` Control-Flow Semantics
- Unsupported Scenario Path Failure Isolation
- Scenario Generation
- Scenario Action Mapping
- Concrete Scenario Validation
- Pytest Generation
- Automatic Test Execution
- Coverage Analysis
- Coverage Workflow
- Reinforcement Learning Infrastructure
- Q-Learning Implementation
- Multi-Episode RL Training
- Real Coverage Feedback Integration
- Scenario Suite Coverage Transition
- Target Coverage Support
- Extended RL State Representation
- Training Statistics
- Best Episode Selection
- End-to-End Real RL Training
- Complex Dataset Evaluation
- Ultracomplex Dataset Evaluation
- Configurable Production Path Expansion
- Persistent Pipeline Diagnostic and Funnel Snapshots
- Structured Scenario and Concrete Rejection Summaries
- Atomic JSON Diagnostic Checkpoints
- Service-Level Global Orchestration Timeout
- Windows Worker Process-Tree and Process-Handle Cleanup
- Deterministic Multi-Function Discovery and Eligibility Reporting
- Function-Isolated Production Orchestration
- Atomic Project JSON Reporting
- `main.py` Option 1 Single/All-Functions Production Integration
- Interactive Static Source Preview
- Exact Function Line/Branch Scenario Contribution Signatures
- Deterministic Greedy Set-Cover Baseline
- Backward Scenario Redundancy Elimination
- Real Minimized-Suite Coverage Verification
- Immutable Ordered RL Episode Selection Traces
- Exact-Coverage RL vs Greedy Strategy Comparison
- Real Raw-RL Suite Coverage Verification
- Optional Per-Function Project Comparison Serialization
- Immutable Local/GitHub Source Acquisition Models
- Secure Public GitHub HTTPS URL Validation and Shallow Clone Policy
- Deterministic Python Project Discovery and Module-Path Inference
- Tool-Owned Temporary Workspace Cleanup
- Source Discovery Resource and Path-Containment Limits
- Separate Inline, Upload, Local Directory, and Public GitHub Source Models
- Static-Only External Discovery Default
- Explicit Trusted External Dynamic Analysis
- Deterministic External Module and Function Limits
- Subprocess/Worker Import-Root Isolation
- External Source to Coverage, Greedy, and RL Comparison Integration
- Atomic JSON-Safe External Analysis Reporting
- Interactive External Source Menu Integration
- FastAPI Asynchronous Analysis Job Backend
- Bounded Job Queue, Safe Polling, and Artifact Downloads
- Responsive Local Production Web UI
- Separate Inline, Upload, and Public GitHub Browser Flows
- Job Lifecycle, Coverage, Strategy Comparison, and Artifact Views
- CSP and Baseline Browser Security Headers
- Exact Combined Analyzed-Project-Scope Coverage
- Module-Qualified Project Line and Branch Identities
- Project-Level Greedy Set Cover and Backward Redundancy Elimination
- Full/Minimized Combined Pytest Exact Replay Verification
- Project Coverage JSON, API, Artifact, Terminal, and Web UI Integration

---

## In Progress

- Raw vs Reachable Coverage Separation
- Remaining Coverage-Gap Classification
- Duplicate and Equivalent Scenario Detection
- Large-Suite Minimization Performance
- Robustness Path-Explosion Reduction
- Comparison Coverage-Execution Caching and Performance
- Project-Wide Orchestration Deadline
- Final `analyze_transactions` Coverage Measurement
- External Analysis Performance and Project-Wide Deadline
- Day 40.2 Final Browser E2E, Production Hardening, and Release Reporting

---

## Planned

- Reward Function Optimization
- State Representation Experiments
- Hyperparameter Evaluation
- Additional Complex Dataset Experiments
- Dependency Discovery
- Isolated and Resource-Limited Execution
- Tree-sitter Integration

---

# Current Limitations

- Bounded path exploration can grow rapidly for functions with nested loops, multiple branches, and exception paths.
- `INFEASIBLE_ONLY` is relative to the configured bounded path space and is not a proof of global semantic infeasibility.
- The `analyze_transactions` benchmark has only a partial concrete-suite observation because the 180-second orchestration limit expired.
- The observed 96 temporary-suite tests are neither the final scenario-pool size nor the number of RL-executed tests.
- Final line and branch coverage for `analyze_transactions` have not been measured.
- Per-function global timeout is available through the service API and `main.py` option 1; a separate total project deadline is not implemented.
- Exact analyzed-project-scope coverage is measured with a real combined suite; whole-repository coverage remains unmeasured unless it is independently measured over every repository target.
- Arbitrary Python expression replay and unrestricted external-project execution are intentionally unsupported; replay is limited to explicitly safe constructs.
- The available deterministic greedy baseline is 1-minimal after backward elimination, not globally optimal; exact global minimum-suite guarantees are not available.
- External dynamic analysis is opt-in trusted execution, not a sandbox. Per-function timeout does not prevent source code from accessing host files, processes, or networks.
- A separate total external-project deadline is not implemented; deterministic module/function limits and per-function deadlines bound individual work units.
- Public acquisition supports anonymous HTTPS repositories only. Private repositories, tokens, automatic dependency installation, and arbitrary untrusted project execution are intentionally unsupported.
- The FastAPI job backend and local Web UI are complete, but authentication, rate limiting, and final browser hardening are not. The terminal workflow remains the only interface for user-owned local project directories.

---

# Final Acceptance Flow

```text
Bounded path discovery
        │
        ▼
Feasibility and reachability classification
        │
        ▼
Safe input and expected-result synthesis
        │
        ▼
Concrete validation
        │
        ├── Accepted scenarios enter the RL action pool
        └── Structured rejections remain observable
        │
        ▼
RL execution and real coverage measurement
        │
        ▼
Report completed metrics or explicit timeout / partial status
```

---

# Research Goal

The long-term research objective of RL-Unit-Test is not simply to maximize raw code coverage.

The optimization problem is:

> **Achieve the maximum reachable code coverage using the minimum number of automatically generated test cases.**

The RL agent will therefore be evaluated against deterministic baseline methods to determine whether learned scenario-selection strategies can reduce test-suite size while preserving verified coverage.

---

# License

MIT License

---

# Author

**Furkan Çinçik**

Software Engineering

Manisa Celal Bayar University

Research Project

**RL-Unit-Test**
