# Final Strategy Evaluation Results

This document preserves the frozen D6/D7 strategy comparison used for final delivery. All strategies receive the same DQM-ordered, safe, concrete-valid candidate pool and the same execution-derived exact line and branch target. Rounded percentages are display-only values. `EXACT_GREEDY` is deterministic but is not claimed to be globally optimal (`globally_minimal=false`).

## Frozen configuration

- Strategies: `DQM_PREFIX`, `EXACT_GREEDY`, `RL`
- RL seeds: `7, 19, 31, 43, 59, 71, 83, 97, 109, 127`
- Episodes per seed: 20
- Epsilon / decay / minimum: `0.30 / 0.95 / 0.05`
- Learning rate / discount factor: `0.5 / 0.9`
- DQM order: `(-normalized_score, path_index)`
- Winner order: exact target reached, selected tests, executed tests, zero-contribution selections, duration
- Combined coverage: `(covered line identities + covered branch identities) / (total line identities + total branch identities)`

## Exact results

| Dataset | Exact target | Pool | DQM | Exact greedy | RL selections by seed | RL mean / median / min / max / SD | Target rate | RL/Greedy | Winner |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| Shopping Cart | 65/92 lines, 33/42 branches, 98/134 combined | 25 | 25 | 13 | 19, 14, 14, 17, 19, 19, 18, 17, 18, 17 | 17.2 / 17.5 / 14 / 19 / 1.7776 | 10/10 | 1.3231 | `BASELINE_WINS` |
| `sample_code.py` | 6/6 lines, 4/4 branches, 10/10 combined | 3 | 3 | 3 | 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 | 3.0 / 3.0 / 3 / 3 / 0 | 10/10 | 1.0000 | `BASELINE_WINS` by duration; selection count tied |
| `sample_complex_code.py` | 72/74 lines, 58/60 branches, 130/134 combined | 39 | 39 | 29 | 36, 36, 36, 35, 35, 35, 37, 36, 36, 36 | 35.8 / 36.0 / 35 / 37 / 0.6 | 10/10 | 1.2345 | `BASELINE_WINS` |
| `sample_robustness_code.py` | 15/16 lines, 11/12 branches, 26/28 combined | 7 | 7 | 7 | 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 | 7.0 / 7.0 / 7 / 7 / 0 | 10/10 | 1.0000 | `BASELINE_WINS` by duration; selection count tied |

Total RL target reach is 40/40 (100%). RL reliably reached every attainable exact target and improved on naive DQM ordering for the two nontrivial pools, but it did not outperform exact greedy in test-selection efficiency. The scientific conclusion is therefore: **RL target reliability is strong, while minimization is weaker under the frozen configuration.** No best-seed cherry-picking, reward tuning, candidate filtering, or denominator changes were applied.

## D5 coverage closure

The authoritative Shopping Cart fixture is `datasets/shopping_cart.py`, SHA-256 `0a7b6d151cf707c69b5762186bdd5c6a5e7436bc849a0670304503b994acbbc1`. Correlated witness synthesis reduced the safe-supported-but-not-generated blocker from 5 lines and 6 branch edges to 0/0. The full pool contains 25 tests; the 13-test exact-greedy suite preserves 65/92 line, 33/42 branch, and 98/134 combined exact identities.

## Scope and safety

The evaluated dataset set is intentionally limited, RL hyperparameters were frozen rather than exhaustively tuned, and exact greedy has evaluation-time access to measured marginal contributions. Unsafe or unresolved constructor, custom-object, aliasing, mutation, and dynamic-call behavior remains controlled `PARTIAL`, `UNSUPPORTED`, or `TIMED_OUT` rather than being executed speculatively. Timeout is not a sandbox. Public summaries contain no raw source, synthetic keyword arguments, setup plans, object state, expected/actual values, raw Q-table, credentials, traceback, memory address, or absolute temporary path.
