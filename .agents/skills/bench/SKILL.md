---
name: bench
description: Validate returned HPC benchmark evidence and 10x scaling analysis for Assignment 1.
---

Read `SPEC.md`, `config/benchmark.yaml`, and the returned run manifest.

Run `pixi run ire-assn1 -- validate-run <path>`. Verify that workload fractions, seeds,
repetitions, candidate ordering, index parameters, hardware identity, timing units, RSS, CUDA memory,
throughput, and index sizes are present and comparable. Check exact and HNSW measurements and the
inputs to 10x extrapolation. Report missing or incomparable evidence before interpreting trends. Do
not execute full benchmarks locally and do not modify result files.
