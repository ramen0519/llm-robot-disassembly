"""Frozen-library orchestration fixtures; no API or physical simulation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_skill_program import program, rule
from test_skill_validation import fixture_case, save
from orchestrator.skill_library_merge import freeze_skill_union
from orchestrator.skill_program import program_hash
from orchestrator.skill_registry import SkillRegistry, retrieve
from worker.task_memory import digest


class SkillLibraryMergeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="skill_library_union_fixture_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def library(self, name="first", policy=None, *, failure=False):
        policy = policy or program()
        root = self.root / name
        registry = SkillRegistry(root / "registry", True, fixture=True)
        identity = program_hash(policy)
        registry.propose(policy, {"description": "Synthetic unit fixture only"},
                         {"origin": "offline_fixture", "generation_id": name})
        bundles = [fixture_case(root, "debug", "debug", "a", policy=policy),
                   fixture_case(root, "admission1", "admission", "b", policy=policy),
                   fixture_case(root, "admission2", "admission", "c", policy=policy,
                                success=not failure)]
        registry.set_manifests(identity,
            {"phase": "debug", "cases": [bundles[0][0]]},
            {"phase": "admission", "cases": [b[0] for b in bundles[1:]]})
        for case, artifacts, _ in bundles:
            registry.add_validation(identity, case, artifacts)
        registry.admit(identity)
        path = root / "frozen.json"
        return path, registry.freeze(path)

    def merge(self, paths, filename="union.json", fixture=True):
        return freeze_skill_union(paths, self.root / filename,
                                  allow_fixture=fixture, allow_test_root=True)

    def rewrite(self, path, value):
        value["content_sha256"] = digest({k: v for k, v in value.items()
                                         if k != "content_sha256"})
        save(path, value)

    def test_empty_input_is_valid_empty_actual_library(self):
        value = self.merge([], fixture=False)
        self.assertEqual(retrieve(value), [])
        self.assertEqual(value["entries"], [])
        self.assertFalse(value["fixture"])
        self.assertTrue(value["usable_for_actual_planning"])
        self.assertEqual(value["merge_provenance"]["source_entry_count"], 0)
        self.assertEqual(value, json.loads((self.root / "union.json").read_text()))

    def test_fixture_permission_does_not_make_no_input_a_fixture(self):
        self.assertFalse(self.merge([])["fixture"])

    def test_union_preserves_complete_entries_and_input_bytes(self):
        left, left_value = self.library()
        right, right_value = self.library("second", program(rule(direction="xn")))
        before = {p: p.read_bytes() for p in (left, right)}
        value = self.merge([left, right])
        self.assertEqual(len(retrieve(value, allow_fixture=True)), 2)
        expected = left_value["entries"] + right_value["entries"]
        self.assertEqual(value["entries"], sorted(expected,
                         key=lambda item: item["candidate"]["program_hash"]))
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        metadata = value["merge_provenance"]
        self.assertEqual(metadata["source_entry_count"], 2)
        self.assertEqual(metadata["unique_entry_count"], 2)
        self.assertEqual(metadata["duplicate_entry_count"], 0)
        self.assertEqual(metadata["new_api_calls"], 0)
        self.assertEqual(metadata["new_physics_calls"], 0)
        self.assertEqual(metadata["sources"][0]["file_sha256"],
                         hashlib.sha256(before[left]).hexdigest())
        self.assertTrue(value["fixture"])
        self.assertFalse(value["usable_for_actual_planning"])
        self.assertFalse(value["hard_pruning"])

    def test_exact_entry_duplicate_deduplicates_and_keeps_both_sources(self):
        path, original = self.library()
        copy = self.root / "copied_snapshot.json"
        copy.write_bytes(path.read_bytes())
        merged = self.merge([path, copy])
        self.assertEqual(merged["entries"], original["entries"])
        self.assertEqual(merged["merge_provenance"]["duplicate_entry_count"], 1)
        self.assertEqual(len(merged["merge_provenance"]["sources"]), 2)

    def test_same_program_with_different_origin_fails_without_output(self):
        first, _ = self.library()
        second, _ = self.library("different_origin")
        with self.assertRaisesRegex(ValueError, "conflicting_frozen_skill_program"):
            self.merge([first, second])
        self.assertFalse((self.root / "union.json").exists())

    def test_fixture_requires_explicit_permission(self):
        path, _ = self.library()
        with self.assertRaisesRegex(ValueError, "fixture_snapshot"):
            self.merge([path], fixture=False)

    def test_fixture_and_actual_even_empty_cannot_mix(self):
        fixture, _ = self.library()
        self.merge([], "actual_empty.json", fixture=False)
        with self.assertRaisesRegex(ValueError, "cannot_mix"):
            self.merge([fixture, self.root / "actual_empty.json"])
        self.assertFalse((self.root / "union.json").exists())

    def test_empty_failed_fixture_stays_empty_fixture(self):
        path, value = self.library(failure=True)
        self.assertEqual(value["entries"], [])
        merged = self.merge([path])
        self.assertEqual(merged["entries"], [])
        self.assertTrue(merged["fixture"])
        self.assertFalse(merged["usable_for_actual_planning"])

    def test_source_snapshot_tampering_rejected(self):
        path, value = self.library()
        value["entries"][0]["candidate"]["card"]["description"] = "tampered"
        save(path, value)
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            self.merge([path])

    def test_rehashed_outer_cannot_hide_inner_tampering(self):
        path, value = self.library()
        value["entries"][0]["candidate"]["card"]["description"] = "tampered"
        self.rewrite(path, value)
        with self.assertRaisesRegex(ValueError, "invalid_frozen_skill_entry"):
            self.merge([path])

    def test_nonadmitted_entry_cannot_be_promoted(self):
        path, value = self.library()
        admission = value["entries"][0]["admission"]
        admission["admitted"] = False
        admission["lifecycle"] = "provisional"
        admission["decision_sha256"] = digest({k: v for k, v in admission.items()
                                                if k != "decision_sha256"})
        self.rewrite(path, value)
        with self.assertRaisesRegex(ValueError, "invalid_frozen_skill_entry"):
            self.merge([path])

    def test_outer_actual_flag_cannot_upgrade_fixture(self):
        path, value = self.library()
        value["fixture"] = False
        value["usable_for_actual_planning"] = True
        self.rewrite(path, value)
        with self.assertRaisesRegex(ValueError, "invalid_frozen_skill_entry"):
            self.merge([path], fixture=False)

    def test_output_is_exclusive_and_may_not_replace_input(self):
        source, _ = self.library()
        merged = self.merge([source])
        original_output = (self.root / "union.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.merge([source])
        self.assertEqual((self.root / "union.json").read_bytes(), original_output)
        with self.assertRaisesRegex(ValueError, "must_not_replace_input"):
            freeze_skill_union([source], source, allow_fixture=True, allow_test_root=True)
        self.assertEqual(len(retrieve(merged, allow_fixture=True)), 1)
        self.assertEqual(merged["entries"], json.loads(source.read_text())["entries"])

    def test_merged_snapshot_can_be_merged_again_without_losing_entries(self):
        source, original = self.library()
        self.merge([source])
        next_value = self.merge([source, self.root / "union.json"], "next.json")
        self.assertEqual(next_value["entries"], original["entries"])
        self.assertEqual(next_value["merge_provenance"]["duplicate_entry_count"], 1)

    def test_e_drive_and_other_forbidden_paths_fail_before_filesystem_access(self):
        for forbidden in ("E:/never_inspected/library.json", "e:relative.json",
                          "/mnt/e/no.json", "C:/dev/../outside.json"):
            with self.subTest(path=forbidden), patch(
                    "orchestrator.skill_library_merge._safe_path",
                    side_effect=AssertionError("filesystem guard must not run")):
                with self.assertRaisesRegex(ValueError, "exclude non-C"):
                    freeze_skill_union([forbidden], self.root / "unused.json",
                                       allow_test_root=True)

    def test_outside_runs_or_temporary_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "stay within project runs"):
            freeze_skill_union([], "C:/dev/unrelated_library.json")

    def test_source_changed_mid_merge_fails_before_writing(self):
        source, _ = self.library()
        from orchestrator.skill_library_merge import read_artifact as real_read

        def changing_read(path, allow_test_root=False):
            result = real_read(path, allow_test_root)
            with Path(path).open("ab") as stream:
                stream.write(b" ")
            return result

        with patch("orchestrator.skill_library_merge.read_artifact", side_effect=changing_read):
            with self.assertRaisesRegex(ValueError, "source_changed_during_merge"):
                self.merge([source])
        self.assertFalse((self.root / "union.json").exists())

    def test_invalid_sequence_and_flags_rejected(self):
        with self.assertRaisesRegex(ValueError, "path_sequence"):
            freeze_skill_union("C:/ignored.json", self.root / "out.json")
        with self.assertRaisesRegex(ValueError, "must_be_bool"):
            freeze_skill_union([], self.root / "out.json", allow_fixture=1)


if __name__ == "__main__":
    unittest.main()
