# Publication verification — 2026-11-21

The user requested publication to `https://github.com/ramen0519/llm-robot-disassembly`.

## Passed

- Exact reviewed copy of 577 existing files; original and copied SHA-256 values match. Source files were rechecked after preparation.
- Existing Python files: 533 AST syntax checks passed. This does not imply runtime/physics success.
- Two generated skill sources match their registered SHA-256 values; no regeneration or new registration.
- Portable tests: 39 passed with Python site-packages disabled (`-B -S`); no model, CAD or physics execution. [Log](reports/publication/portable_tests.txt).
- Secret/privacy scan: no genuine API token, private key, personal user path, email or phone found. The `sk-proj-fake...` literal in the redaction test is a deliberate fake and was reviewed.
- Original C-drive code/results unchanged. All publication files/test temporary files were written on F. E was not accessed.

## Failed / limited

A broader test command ran 59 tests: 49 passed, 10 errors. Those errors are C-only ledger/memory path restrictions on this F-drive checkout. They were not bypassed. [Full log](reports/publication/extended_tests_known_path_errors.txt).

The portable 39 tests are a subset/recheck, not an additional 39 independent successful tests. No fresh installation, complete test suite, live API or full RedMax run was performed for this publication.

## Actual offline commands

```text
python -B -m unittest tests.test_relational_knowledge tests.test_relational_retrieval tests.test_relational_migration tests.test_relational_runtime tests.test_scored_order_skill -v
# 59 tests, 10 path-guard errors

python -B -S -m unittest tests.test_relational_knowledge tests.test_relational_retrieval tests.test_relational_migration tests.test_scored_order_skill.ScoredOrderSkillTests.test_id_independent_numeric_program_and_complete_ties tests.test_scored_order_skill.ScoredOrderSkillTests.test_no_io_no_part_ids_no_shapes_leak -v
# 39 tests, OK
```

The existing orchestrator Python executable was used, with all temporary storage directed to the F-drive publication directory. No package was installed.

## Distribution scope

The full experience library, private CAD, original attachments, credentials, raw provider requests/responses, build products and large trajectories are not included. See PUBLICATION_MANIFEST.json for the exact existing files. New explanatory documents and example source extracts are identified by the initial Git commit.

The repository commit records the submitted snapshot. Remote upload verification is recorded separately in the local publication receipt after push.
