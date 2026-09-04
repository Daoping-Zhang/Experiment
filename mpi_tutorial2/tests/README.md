# Test files

`scripts/verify.py` is the automated acceptance suite (runs real teacher +
worker processes, asserts correct reduce/allreduce results in both modes,
4 MB payload, no deadlock — every scenario has a timeout).

Quick smoke test without subprocess orchestration:

```bash
python3 tests/test_smoke.py
```
