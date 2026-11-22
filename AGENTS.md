# Disassembly research workspace

Read README.md, IMPLEMENTATION_STATUS.md, and PUBLICATION_CHECKS.md first. Documents may describe earlier development stages; do not treat a plan as an implementation or a historical result as a new run.

Preserve official ASAP and previous results. Use a new run directory for any experiment. Do not publish credentials, raw API logs, CAD assets or private experience libraries. E drive must never be accessed or enumerated. Do not follow symlinks/junctions into forbidden paths.

Paid API requests and long physics experiments require explicit scope, model, call/cost/time caps and user authorization. Existing historical budgets do not authorize new runs. Offline tests must not silently call a provider or load stored keys.

The current CLI uses --run and the caseplanner package. Astra ranks valid parts; native ASAP retains DFS, pose/direction/support search and final-pair policy. The default research store is .cases/. Offline provider tests use a separate library. The --run-dir launcher and the prior experience, lesson and Python skill libraries are retained for historical runs. See docs/ASTRA_CASES_V1.md for the current profile and docs/history/README_20261121.md for the earlier one.

The current planner exports the verified part-level chain and a reversed assembly candidate in export_v1.json. Robot grasp, motion, and physical execution remain later work.

This public snapshot excludes most local run data. Report missing data and portability limits instead of silently substituting a mock or changing the original verifier. Record actual changes and verification in IMPLEMENTATION_STATUS.md and CHANGELOG.md.
