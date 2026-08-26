---
name: eval
description: Validate synthetic or returned HPC evaluation evidence for Assignment 1.
---

Read `SPEC.md` and the resolved run manifest before evaluating results.

For synthetic validation, run `pixi run eval-synthetic` and `pixi run test`.

For a returned HPC bundle, run `pixi run ire-assn1 -- validate-run <path>`. Inspect metric
coverage, exclusion counts, bootstrap sample counts, slice definitions, fixed sample predictions,
configuration identity, Git identity, and input checksums. Compare reported intervals against the
stored bootstrap draws. Report failures before observations. Do not modify result files.
