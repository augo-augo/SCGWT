# Testing Guide

Run the test suite with `pytest` from the repository root:

```bash
pytest
```

The initial tests cover:

- entropy estimation sanity checks (`tests/test_motivation.py`)
- rollout buffer sampling shape and basic training-loop wiring (`tests/test_training.py`)

Add additional scenarios mirroring new modules as they land (e.g., empowerment stability,
workspace heuristics, and optimizer behaviours).
