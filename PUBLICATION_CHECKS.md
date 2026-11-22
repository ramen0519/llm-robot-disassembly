# 공개본과 후속 구현 검증

## 2026-11-22 후속 로컬 브랜치

F 드라이브에서 공개 기준 커밋 `034006a`으로부터 독립 작업본을 만들었다. 보존 점검에서 C 작업본과 일치하는 공개 파일 577개를 대조했다. 실험 산출물·연구실 CAD·로컬 모델과 키는 Git에 포함하지 않는다. 이 후속 브랜치는 로컬에만 있다.

[실측, 모의 응답과 미실행 항목](reports/MASTER_DESIGN_DELIVERY_20261122.md) · [산출물 무결성 검사](evidence/release_verification.json)

## Publication verification — 2026-11-21

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
