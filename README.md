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

Global timeout is integrated into the service API. `main.py` option 8 and multi-function project orchestration remain work for a later sprint.

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

The current small production acceptance run for `calculate_category_usage` completed with 3 bounded paths, a 3-scenario pool, 3 RL-executed tests, 3 Q-table states, 100% function line coverage, 100% function branch coverage, and a `COMPLETED` pipeline diagnostic.

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

The next objective is to determine whether the same or higher reachable coverage can be achieved with a smaller subset:

```text
165 available scenarios
        │
        ▼
Scenario contribution analysis
        │
        ├── Deterministic greedy baseline
        └── RL-based scenario selection
        │
        ▼
Minimum verified scenario subset
        │
        ▼
Maximum reachable coverage
```

To evaluate this objectively, scenario contribution analysis and a greedy minimum-scenario baseline will be used.

The baseline measures the marginal coverage contribution of each scenario:

$$
\text{Marginal Contribution} =
\text{Coverage}(\text{Current Suite} + \text{Scenario})
-
\text{Coverage}(\text{Current Suite})
$$

This provides a deterministic baseline against which the RL agent's test-selection efficiency can be compared.

---

# Test Status

The project contains an extensive automated test suite covering individual analysis and RL components, regression cases, integration flows, and end-to-end real coverage training.

Latest full regression run:

| Test result | Status |
| --- | ---: |
| Passed | 1,628 |
| Failed | 0 |
| Duration | 96.67s |

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

---

## In Progress

- Raw vs Reachable Coverage Separation
- Remaining Coverage-Gap Classification
- Duplicate and Equivalent Scenario Detection
- Scenario Marginal Coverage Analysis
- Greedy Minimum Scenario Baseline
- Test Suite Minimization
- RL Test Selection Efficiency Evaluation
- Robustness Path-Explosion Reduction
- `main.py` Option 8 and Multi-Function Timeout Orchestration
- Final `analyze_transactions` Coverage Measurement

---

## Planned

- RL vs Greedy Baseline Comparison
- Reward Function Optimization
- State Representation Experiments
- Hyperparameter Evaluation
- Configurable `main.py` Command-Line Interface
- Additional Complex Dataset Experiments
- External Python File and Project Input
- Git Repository Input
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
- Global timeout is available through the service API, but `main.py` option 8 and multi-function project orchestration are not yet integrated.
- Arbitrary Python expression replay and unrestricted external-project execution are intentionally unsupported; replay is limited to explicitly safe constructs.
- Exact global minimum-suite guarantees are not yet available; minimization remains an evaluation objective.

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
