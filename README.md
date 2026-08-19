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
0. Çıkış
```

Option 1 asks for a real Python source file, module path, single/all-functions selection, output directory, and optionally advanced production settings. There is no implicit `datasets/sample_code.py` or `calculate_score` fallback. Example datasets remain valid only when the user selects them explicitly.

Option 2 asks for a real Python source file and reports AST function, complexity, and production-support metadata without starting test generation, coverage, or RL training.

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

The real local-project acceptance discovered top-level, package, and `src/` modules, excluded tests by default, included them on request, ignored `.venv`, and preserved the user-owned workspace. A real temporary local Git fixture verified commit-SHA resolution. Anonymous `ls-remote` access succeeded for both the current origin and GitHub's official small public connectivity fixture. The single official resolver acceptance completed the clone in 1.133 seconds and resolved commit `7fd1a60b01f91b314f59955a4e4d4e80d8edf11d`. The fixture contained no Python files; this is reported as a non-fatal `NO_PYTHON_FILES` partial inventory rather than `CLONE_FAILED`. That run exposed Windows read-only Git pack cleanup behavior, which now has a regression-tested writable-retry path; the tool-owned workspace was removed and no Git process remained. Repository inventory handoff into multi-file RL analysis remains a 38.2 task.

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
- Control Flow Graph (CFG)
- Decision Quality Matrix (DQM)
- Data-Flow Analysis
- Symbolic Path-State Analysis
- Reinforcement Learning
- Q-Learning

### Future Integrations

- Tree-sitter
- FastAPI
- Web Dashboard
- Git Repository Input
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
| Passed | 1,807 |
| Failed | 0 |
| Skipped | 1 (Windows symlink creation unavailable) |
| Duration | 109.57s |

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

---

## In Progress

- Raw vs Reachable Coverage Separation
- Remaining Coverage-Gap Classification
- Duplicate and Equivalent Scenario Detection
- Test Suite Minimization
- Robustness Path-Explosion Reduction
- Comparison Coverage-Execution Caching and Performance
- Project-Wide Orchestration Deadline
- Final `analyze_transactions` Coverage Measurement
- Repository Inventory to Multi-Function Analysis Integration (38.2)

---

## Planned

- Reward Function Optimization
- State Representation Experiments
- Hyperparameter Evaluation
- Additional Complex Dataset Experiments
- Dependency Discovery
- Isolated and Resource-Limited Execution
- Tree-sitter Integration
- FastAPI Backend
- Web Interface

---

# Current Limitations

- Bounded path exploration can grow rapidly for functions with nested loops, multiple branches, and exception paths.
- `INFEASIBLE_ONLY` is relative to the configured bounded path space and is not a proof of global semantic infeasibility.
- The `analyze_transactions` benchmark has only a partial concrete-suite observation because the 180-second orchestration limit expired.
- The observed 96 temporary-suite tests are neither the final scenario-pool size nor the number of RL-executed tests.
- Final line and branch coverage for `analyze_transactions` have not been measured.
- Per-function global timeout is available through the service API and `main.py` option 1; a separate total project deadline is not implemented.
- Aggregate project coverage is reported as unmeasured because function percentages are not arithmetically averaged and a combined-suite measurement is not yet performed.
- Arbitrary Python expression replay and unrestricted external-project execution are intentionally unsupported; replay is limited to explicitly safe constructs.
- The available deterministic greedy baseline is 1-minimal after backward elimination, not globally optimal; exact global minimum-suite guarantees are not available.
- Source acquisition currently produces a safe project inventory only; repository-wide RL/coverage orchestration is not connected until 38.2.
- Public acquisition supports anonymous HTTPS repositories only. Private repositories, tokens, dependency installation, and untrusted project execution are intentionally unsupported.

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
