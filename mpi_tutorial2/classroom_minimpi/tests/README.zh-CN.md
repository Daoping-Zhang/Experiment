# 测试文件（Tests）

> 中文版 — English version: [README.md](./README.md)

`scripts/verify.py` 是自动验收套件（启动真实 teacher + worker 进程，断言两种
Mode 下 reduce/allreduce 结果正确、4 MB payload、无死锁——每个场景都有超时保护）。

不 orchestrate 子进程的快速冒烟测试：

```bash
python3 tests/test_smoke.py
```
