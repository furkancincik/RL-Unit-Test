# RL-Unit-Test

An intelligent framework for automated Python unit test generation and coverage optimization using Abstract Syntax Trees (AST), Control Flow Graphs (CFG), Decision Quality Matrix (DQM), and Reinforcement Learning (RL).

---

# Project Overview

RL-Unit-Test is a research-oriented software engineering project that automatically analyzes Python source code, constructs execution paths, prioritizes them using a Decision Quality Matrix (DQM), generates unit test scenarios, creates pytest test cases, executes them automatically, measures code coverage, and applies Reinforcement Learning to optimize test selection for maximum coverage with minimum test cases.

The project is being developed as a modular and extensible architecture suitable for both academic research and industrial applications.

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

- Scenario Generation
- Stable Scenario IDs
- Scenario Metadata
- Priority Levels
- Execution Path Mapping

---

## Reinforcement Learning Infrastructure

- Coverage State Representation
- State Encoder
- State Key Encoding
- Action Model
- Reward Calculation
- Coverage Environment
- Q-Table
- Epsilon-Greedy Policy
- Q-Learning Agent
- Q-Learning Trainer
- Scenario-Action Mapping

---

## Test Generation

- Automatic pytest Generation
- Stable Test Naming
- Metadata Generation

---

## Test Automation

- Automatic Test Execution
- Execution Summary
- Timeout Handling
- Error Reporting

---

## Coverage Analysis

- Line Coverage
- Branch Coverage
- Coverage Reports
- Coverage Workflow

---

## Reporting

- JSON Reports
- DQM Reports
- Coverage Reports

---

# Architecture

```
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
Q-Learning Trainer
      │
      ▼
Q-Learning Agent
      │
      ▼
Pytest Generator
      │
      ▼
Automatic Test Execution
      │
      ▼
Coverage Measurement
      │
      ▼
Coverage Feedback
      │
      ▼
Next Training Episode
```

---

# Technologies

- Python 3.14
- AST
- Pytest
- coverage.py
- JSON
- Control Flow Graph (CFG)
- Decision Quality Matrix (DQM)
- Reinforcement Learning
- Q-Learning

Future Integrations

- Tree-sitter
- FastAPI
- Web Dashboard

---

# Project Statistics

Current Implementation

### Static Analysis

- ✅ AST Analyzer
- ✅ Function Analyzer
- ✅ Complexity Metrics

### CFG

- ✅ CFG Builder
- ✅ Execution Path Analyzer

### DQM

- ✅ Decision Quality Matrix
- ✅ Path Prioritization

### Scenario Generation

- ✅ Scenario Generator
- ✅ Scenario Action Mapper

### Reinforcement Learning

- ✅ Coverage Environment
- ✅ Reward Calculator
- ✅ State Encoder
- ✅ Q-Table
- ✅ Epsilon-Greedy Policy
- ✅ Q-Learning Agent
- ✅ Q-Learning Trainer

### Test Generation

- ✅ Pytest Generator
- ✅ Test File Writer

### Automation

- ✅ Test Execution
- ✅ Coverage Workflow

---

# Test Status

```
520+ tests passed
0 failures
0 warnings
```

---

# Project Structure

```
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

---

## In Progress

- RL Integration with Scenario Pipeline
- Coverage Feedback Integration

---

## Planned

- Intelligent Test Input Generation
- Multi-Episode RL Training
- Coverage Optimization
- Test Suite Minimization
- Tree-sitter Integration
- FastAPI Backend
- Web Interface

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