RL-Unit-Test

An intelligent and research-oriented framework for automated Python unit test generation and test-suite optimization using Abstract Syntax Trees (AST), Control Flow Graphs (CFG), path feasibility analysis, a Decision Quality Matrix (DQM), real coverage feedback, and Reinforcement Learning (RL).

Project Overview

RL-Unit-Test analyzes Python source code, discovers execution paths, evaluates whether those paths are feasible, generates concrete test inputs, creates and executes pytest scenarios, measures real line and branch coverage, and trains a Q-learning agent to select efficient test suites.

The project has two ordered optimization objectives:

Reach the maximum achievable coverage over reachable program behavior.

Preserve that coverage using the minimum possible number of test scenarios.

Coverage is therefore the primary objective. Test count is minimized only among suites that achieve the same coverage target.

The framework is developed as a modular architecture suitable for academic experimentation and future industrial use with externally supplied Python files or repositories.

Analysis and Generation Pipeline

Python Source
      │
      ▼
AST and Function Analysis
      │
      ▼
Control Flow Graph Construction
      │
      ▼
Bounded Execution Path Discovery
      │
      ├── Data-Flow Analysis
      ├── Path-State Analysis
      └── Path-Feasibility Analysis
      │
      ▼
Concrete Input Candidate Generation
      │
      ▼
DQM Path Prioritization
      │
      ▼
Scenario Generation
      │
      ▼
Concrete Execution Validation
      │
      ▼
Scenario-Action Mapping
      │
      ▼
Q-Learning Training
      │
      ▼
Pytest Suite Generation and Execution
      │
      ▼
Real Line and Branch Coverage
      │
      ▼
Reward and Q-Value Update

Concrete execution validation is a safety boundary between static analysis and RL training. A generated scenario is admitted to the RL action pool only when its observed result or exception matches its expected behavior on the target function.

Current Features

Static Code Analysis

Python AST analysis

Function and parameter extraction

Type-hint analysis

Cyclomatic complexity analysis

Condition, loop, return, and exception detection

Docstring and structural metadata extraction

Risk-level detection

Control-Flow and Path Analysis

Control Flow Graph construction

Bounded execution path discovery

Configurable per-node visit limit

Loop and continue flow modeling

Exception-flow analysis

Stable execution-path metadata

Data-Flow, State, and Feasibility Analysis

Variable definition and update tracking

Local symbolic state propagation

Numeric range inference

Relational constraint analysis

Variable-to-variable constraint support

Boolean and collection-state constraints

Loop-state consistency checks

Path classification as FEASIBLE, INFEASIBLE, or UNKNOWN

Early elimination of contradictory execution paths

Input and Scenario Generation

Constraint-aware input candidate generation

Typed default value generation

Support for primitive and collection types

Support for list, tuple, set, Optional, Union, and nested types

Subscript alias propagation such as first_item = items[0]

Loop-variable constraint propagation

Relational witness forwarding

Dynamic return-value evaluation

Safe f-string and format-spec evaluation

Stable scenario IDs and metadata

Execution-path-to-scenario mapping

Concrete result and exception validation

Decision Quality Matrix

Path scoring and ranking

Risk-based prioritization

Normalized DQM scores

Decision, loop, and exception metadata

Reinforcement Learning

Coverage-state representation

Extended state-key encoding

Executed-test-count representation

Scenario action model

Q-table

Epsilon-greedy policy

Dynamic epsilon control

Q-learning agent and trainer

Multi-episode training sessions

Best-episode selection

Configurable target coverage

Training statistics

Real Coverage Environment

Real pytest execution

Real coverage.py feedback

Cumulative scenario-suite transitions

Episode reset support

Target-coverage termination

Function-level line and branch coverage

File-level line and branch coverage

Missing-line and uncovered-branch detection

Reachable-coverage support

Reward System

The reward model currently considers:

Coverage improvement

No-improvement penalties

Test execution cost

Full or target coverage reward

Efficient scenario selection

This supports the project's lexicographic objective: maximize coverage first, then reduce test-suite size without losing coverage.

Test Generation and Reporting

Automatic pytest generation

Stable test naming

Cumulative test-suite generation

Test-file writing

Timeout and execution-error handling

Human-readable RL training reports

JSON reports

DQM reports

Function and file coverage summaries

Experimental Evaluation

The framework is evaluated on multiple datasets with different control-flow complexity levels. Results from different targets are reported separately and should not be interpreted as the same experiment.

evaluate_application Benchmark

An earlier complex-dataset experiment produced:

Generated scenarios              : 36
Reachable maximum line coverage  : 98.65%
Reachable maximum branch coverage: 98.61%
Highest single-scenario coverage : 24.32%
Lowest single-scenario coverage  : 4.05%
Average single-scenario coverage : 19.40%

No individual scenario could approach the reachable maximum. The experiment demonstrated that multiple scenarios were necessary to cover distinct execution paths. Leave-one-out and greedy checks also showed that all 36 scenarios were required by the available scenario pool to preserve that result.

process_order Ultracomplex Benchmark

The datasets/sample_ultracomplex_code.py::process_order target was used to expose limitations in loop expansion, path feasibility, relational input generation, dynamic expected-result evaluation, and concrete scenario validation.

With the original path expansion limit, the pipeline produced:

Execution paths             : 619
FEASIBLE paths              : 25
INFEASIBLE paths            : 594
Concrete scenario pool      : 14
Function line coverage      : 52.24%
Function branch coverage    : 48.08%

The diagnostic showed that paths after a two-iteration local while loop required a third visit to the loop condition in order to model the exit check. After integrating a configurable max_visits_per_node value with a default of 3, and improving general-purpose input generation, the updated production pipeline produced:

Execution paths             : 2128
FEASIBLE paths              : 340
INFEASIBLE paths            : 1788
UNKNOWN paths               : 0
Concrete scenario pool      : 60
Executed scenarios          : 51
Function line coverage      : 83.58%
Function branch coverage    : 84.62%
File line coverage          : 68.60%
File branch coverage        : 66.67%

This improvement was achieved without modifying the benchmark source code and without adding target-specific rules to the framework.

Some terminal branches in the benchmark are intentionally or structurally unreachable. Consequently, the remaining work is not to force raw coverage to 100%, but to identify the reachable coverage ceiling and minimize the scenario subset that preserves it.

Current Optimization Problem

The input-generation bottleneck has been substantially reduced: the ultracomplex scenario pool increased from 14 to 60 and function coverage increased from 52.24% to 83.58%.

The current research problem is now test-suite minimization:

60 validated scenarios
        │
        ▼
Scenario contribution analysis
        │
        ├── Deterministic greedy baseline
        └── RL-based scenario selection
        │
        ▼
Smallest suite preserving maximum reachable coverage

Marginal scenario contribution is defined as:

Marginal Contribution =
Coverage(Current Suite + Scenario)
- Coverage(Current Suite)

The greedy baseline provides a deterministic reference against which the Q-learning agent's selection efficiency can be evaluated.

Current Development Status

Completed

AST and function analysis

Type-hint extraction

CFG construction

Bounded execution path analysis

Loop and exception-flow analysis

Data-flow analysis

Path-state analysis

Path-feasibility classification

DQM scoring and path prioritization

Constraint-aware input candidate generation

Typed and collection input generation

Relational witness integration

Dynamic return and f-string evaluation

Scenario generation and action mapping

Concrete scenario execution validation

Pytest generation and file writing

Automated test execution

Real line and branch coverage measurement

Function-level and file-level coverage evaluation

Coverage-state and reward models

Q-learning infrastructure

Multi-episode real RL training

Training statistics and reporting

Configurable execution-path visit limit in the production service

Complex and ultracomplex benchmark evaluation

In Progress

Raw versus reachable coverage separation

Remaining coverage-gap classification

Duplicate and equivalent scenario detection

Scenario marginal-contribution analysis

Greedy minimum-scenario baseline

RL test-selection efficiency evaluation

Test-suite minimization

Planned

RL versus greedy baseline comparison

Reward-function optimization

State-representation experiments

Hyperparameter evaluation

Configurable main.py command-line interface

Additional complex dataset experiments

External Python file and project ingestion

Git repository input support

Dependency discovery and isolated execution

Tree-sitter integration

FastAPI backend

Web dashboard

Next Sprint

The next sprint focuses on measurable optimization rather than simply generating more paths:

Make the target source file, module, function, episode count, and path-visit limit configurable from main.py.

Report raw coverage and reachable coverage separately.

Classify each remaining uncovered line and branch as reachable, unreachable, unsupported, or unresolved.

Normalize and deduplicate concrete scenarios before RL training.

Measure each scenario's marginal line and branch contribution.

Build a deterministic greedy minimization baseline.

Compare RL and greedy suites at equal coverage.

Export the smallest verified pytest suite found.

The sprint succeeds when the current process_order coverage is preserved or improved using fewer than 51 executed scenarios, without introducing benchmark-specific logic.

Test Status

The project includes unit, integration, regression, and end-to-end tests for the analysis, generation, validation, coverage, and RL layers.

Latest full regression run:

1249 tests passed

Latest integrated change:

e1fbe53 feat: enhance path analysis and RL test generation

Project Structure

RL-Unit-Test
│
├── analyzer
├── cfg
│   ├── control_flow_graph.py
│   ├── data_flow_analyzer.py
│   ├── path_analyzer.py
│   ├── path_feasibility_analyzer.py
│   └── path_state_analyzer.py
├── evaluator
├── generator
│   ├── input_candidate_generator.py
│   ├── path_input_generator.py
│   └── scenario_generator.py
├── rl
├── services
│   └── real_rl_training_service.py
├── datasets
├── tests
├── output
└── main.py

Technologies

Python 3.14

Python AST

pytest

coverage.py

JSON

Control Flow Graphs

Decision Quality Matrix

Q-learning

Planned integrations include Tree-sitter, FastAPI, and a web dashboard.

Research Goal

The long-term goal of RL-Unit-Test is not merely to maximize raw code coverage.

Achieve the maximum reachable code coverage using the minimum number of automatically generated test cases.

The RL agent is evaluated against deterministic baselines to determine whether learned scenario-selection strategies can reduce test-suite size while preserving the same verified coverage.

License

MIT License

Author

Furkan Çinçik

Software Engineering

Manisa Celal Bayar University

Research Project