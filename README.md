# RL-Unit-Test

An intelligent framework for automated Python unit test generation and coverage optimization using Abstract Syntax Trees (AST), Control Flow Graphs (CFG), Decision Quality Matrix (DQM), and Reinforcement Learning (RL).

---

# Project Overview

RL-Unit-Test is a research-oriented software engineering project that automatically analyzes Python source code, constructs execution paths, prioritizes them using a Decision Quality Matrix (DQM), generates unit test scenarios, creates pytest test cases, executes them automatically, measures real code coverage, and applies Reinforcement Learning to optimize test selection.

The main optimization objectives are:

1. Reach the maximum achievable code coverage.
2. Reach that coverage using the minimum possible number of test scenarios.

The project is being developed as a modular and extensible architecture suitable for academic research and future industrial applications.

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
- Execution Path Discovery
- Loop Detection
- Exception Flow Analysis

---

## Path Prioritization

- Decision Quality Matrix (DQM)
- Path Ranking
- Risk-Based Prioritization
- Normalized DQM Scores

---

## Scenario Generation

- Automatic Scenario Generation
- Stable Scenario IDs
- Scenario Metadata
- Priority Levels
- Execution Path Mapping
- Scenario-Action Mapping

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

---

## Test Automation

- Automatic Test Execution
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
Decision Quality Matrix
      │
      ▼
Scenario Generator
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
      │
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

---

# Technologies

- Python 3.14
- Python AST
- Pytest
- coverage.py
- JSON
- Control Flow Graph (CFG)
- Decision Quality Matrix (DQM)
- Reinforcement Learning
- Q-Learning

### Future Integrations

- Tree-sitter
- FastAPI
- Web Dashboard

---

# Current Development Status

## Static Analysis

- ✅ AST Analyzer
- ✅ Function Analyzer
- ✅ Complexity Metrics

## CFG

- ✅ CFG Builder
- ✅ Execution Path Analyzer

## DQM

- ✅ Decision Quality Matrix
- ✅ Path Prioritization

## Scenario Generation

- ✅ Scenario Generator
- ✅ Scenario Action Mapper

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

---

# Experimental Evaluation

The RL pipeline is now being evaluated on more complex datasets instead of relying only on the initial sample program.

A recent experiment on the `evaluate_application` target produced:

```text
Generated scenarios             : 36
Reachable maximum coverage      : 98.65%
Highest single-scenario coverage: 24.32%
Lowest single-scenario coverage : 4.05%
Average single-scenario coverage: 19.40%
```

The experiment demonstrates that no individual scenario can approach the reachable maximum coverage. Multiple scenarios must therefore be combined to cover different execution paths.

The current research focus is determining how many of the generated scenarios are actually required to reach the same maximum coverage.

---

# Current Optimization Problem

The framework can successfully generate scenarios and reach high real coverage. The next optimization problem is test-suite minimization.

Current situation:

```text
36 generated scenarios
        │
        ▼
Scenario selection
        │
        ▼
98.65% reachable coverage
```

The next objective is to determine whether the same coverage can be achieved with a smaller subset:

```text
36 available scenarios
        │
        ▼
Minimum useful scenario subset
        │
        ▼
98.65% reachable coverage
```

To evaluate this objectively, scenario contribution analysis and a greedy minimum-scenario baseline will be used.

This baseline will measure the marginal coverage contribution of each scenario:

```text
Marginal Contribution =
Coverage(Current Suite + Scenario)
-
Coverage(Current Suite)
```

This will provide a deterministic baseline against which the RL agent's test-selection efficiency can be compared.

---

# Test Status

The project currently contains an extensive automated test suite covering individual RL components, integration flows, and end-to-end real coverage training.

Latest full regression run:

```text
992 tests passed
```

Additional focused test suites for recently modified RL components have also completed successfully.

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
- Execution Path Analysis
- Decision Quality Matrix
- Scenario Generation
- Scenario Action Mapping
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
- Individual Scenario Coverage Analysis

---

## In Progress

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
- Intelligent Test Input Generation
- Additional Complex Dataset Experiments
- Tree-sitter Integration
- FastAPI Backend
- Web Interface

---

# Research Goal

The long-term research objective of RL-Unit-Test is not simply to maximize code coverage.

The optimization problem is:

> **Achieve the maximum reachable code coverage using the minimum number of automatically generated test cases.**

The RL agent will therefore be evaluated against deterministic baseline methods to determine whether learned scenario-selection strategies can reduce test-suite size while preserving coverage.

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