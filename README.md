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
- Loop and `continue` Flow Detection
- Exception Flow Analysis
- Stable Path Metadata

---

## Data-Flow and Path Feasibility

- Variable Definition and Update Tracking
- Local Symbolic State Propagation
- Numeric Range Inference
- Boolean and Collection-State Constraints
- Relational Constraint Analysis
- Variable-to-Variable Constraint Support
- Loop-State Consistency Checks
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
- Relational Witness Forwarding
- Dynamic Return-Value Evaluation
- Safe f-String and Format-Spec Evaluation
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
- ✅ Loop and Exception Flow Analysis

## Data Flow and Feasibility

- ✅ Data Flow Analyzer
- ✅ Path State Analyzer
- ✅ Path Feasibility Analyzer
- ✅ Relational Constraint Support
- ✅ Contradictory Path Filtering

## DQM

- ✅ Decision Quality Matrix
- ✅ Path Prioritization

## Scenario Generation

- ✅ Input Candidate Generator
- ✅ Typed Path Input Generator
- ✅ Loop and Alias-Aware Input Generation
- ✅ Dynamic Expected Result Generation
- ✅ Scenario Generator
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

After integrating a configurable `max_visits_per_node` value with a production default of `3`, improving generic input generation, and validating generated scenarios against the real target function:

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

The improvement was achieved without modifying the benchmark source code and without adding `process_order`-specific rules to the framework.

Some terminal branches are intentionally or structurally unreachable. The next evaluation step is therefore to separate raw coverage from reachable coverage and determine the minimum scenario subset that preserves the reachable maximum.

---

# Current Optimization Problem

The framework can now generate a substantially larger validated scenario pool for the ultracomplex target. The current optimization problem is test-suite minimization.

Current situation:

```text
60 validated scenarios
        │
        ▼
RL scenario selection
        │
        ▼
51 executed scenarios
        │
        ▼
83.58% line coverage / 84.62% branch coverage
```

The next objective is to determine whether the same or higher reachable coverage can be achieved with a smaller subset:

```text
60 available scenarios
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
| Passed | 1,249 |
| Failed | 0 |

Latest integrated development commit:

`e1fbe53 feat: enhance path analysis and RL test generation`

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
- Relational Witness Integration
- Dynamic Return and f-String Evaluation
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

---

## In Progress

- Raw vs Reachable Coverage Separation
- Remaining Coverage-Gap Classification
- Duplicate and Equivalent Scenario Detection
- Scenario Marginal Coverage Analysis
- Greedy Minimum Scenario Baseline
- Test Suite Minimization
- RL Test Selection Efficiency Evaluation

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