# RL-Unit-Test

An intelligent framework for automated unit test generation and coverage optimization using Control Flow Graphs (CFG), Decision Quality Matrix (DQM), and Reinforcement Learning.

---

## Project Overview

RL-Unit-Test is a research-oriented software engineering project that automatically analyzes Python source code, generates execution paths, prioritizes them using a Decision Quality Matrix (DQM), creates pytest test cases, executes them automatically, measures code coverage, and prepares the infrastructure for Reinforcement Learning-based optimization.

---

## Current Features

### Static Code Analysis
- Python AST Analysis
- Function Metrics
- Cyclomatic Complexity
- Branch Analysis
- Risk Level Detection
- Type Hint Detection
- Docstring Analysis

### Control Flow Analysis
- Control Flow Graph (CFG)
- Execution Path Discovery
- Loop Detection
- Exception Flow Analysis

### Path Prioritization
- Decision Quality Matrix (DQM)
- Path Ranking
- Risk-Based Prioritization
- Normalized DQM Scores

### Test Generation
- Scenario Generation
- Automatic pytest Generation
- Stable Test Naming
- Metadata Generation

### Test Automation
- Automatic Test Execution
- Execution Summary
- Timeout Handling
- Error Reporting

### Coverage Analysis
- Line Coverage
- Branch Coverage
- Coverage Reports
- Coverage Workflow

### Reporting
- JSON Reports
- DQM Reports
- Coverage Reports

---

## Architecture

```
Python Source
      │
      ▼
 AST Analyzer
      │
      ▼
 CFG Builder
      │
      ▼
 Path Analyzer
      │
      ▼
 DQM Prioritization
      │
      ▼
 Scenario Generator
      │
      ▼
 Pytest Generator
      │
      ▼
 Test Execution
      │
      ▼
 Coverage Measurement
      │
      ▼
 Reinforcement Learning (Next Phase)
```

---

## Technologies

- Python 3.14
- AST
- Pytest
- coverage.py
- JSON
- Decision Quality Matrix (DQM)
- Control Flow Graph (CFG)
- Reinforcement Learning (planned)
- Tree-sitter (planned)
- FastAPI (planned)

---

## Project Statistics

Current Status

- ✅ Static Analyzer
- ✅ CFG Builder
- ✅ Path Analyzer
- ✅ DQM Engine
- ✅ Scenario Generator
- ✅ Pytest Generator
- ✅ Test File Writer
- ✅ Automatic Test Execution
- ✅ Coverage Measurement
- ✅ Coverage Workflow
- ✅ CLI Interface

---

## Test Status

```
130 tests passed
0 failures
0 warnings
```

---

## Roadmap

### Completed

- AST Analysis
- CFG Construction
- Execution Path Analysis
- DQM
- Scenario Generation
- pytest Generation
- Test Automation
- Coverage Measurement

### In Progress

- Intelligent Test Input Generation

### Planned

- Reinforcement Learning Agent
- Coverage Optimization
- Test Minimization
- Tree-sitter Integration
- FastAPI API
- Web Interface

---

## License

MIT License

---

## Author

**Furkan Çinçik**

Software Engineering

Manisa Celal Bayar University

RL Unit Test Research Project