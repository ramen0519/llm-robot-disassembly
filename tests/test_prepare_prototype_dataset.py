"""No extraction/API/physics: validate the bounded dataset preparation contract."""
import importlib.util
from pathlib import Path
import stat
import unittest
from unittest.mock import patch
import zipfile


SOURCE = Path(__file__).parents[1] / "tools/prepare_prototype_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_prototype_dataset_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def member(name, size=100):
    info = zipfile.ZipInfo(name)
    info.file_size = size
    return info


def good_catalog():
    return [member(MODULE.MEMBER_PREFIX + name)
            for name in list(MODULE.EXPECTED_SHA256) + ["contact_graph.json"]]


class DatasetPreparationTests(unittest.TestCase):
    def test_only_predeclared_payloads_selected(self):
        catalog = good_catalog() + [member("test_assembly/00561/0.obj")]
        self.assertEqual(set(MODULE.validated_members(catalog)),
                         set(MODULE.EXPECTED_SHA256) | {"contact_graph.json"})

    def test_archive_path_traversal_and_duplicate_rejected(self):
        for invalid in ("../escape.obj", "/escape.obj", "test_assembly/../escape.obj",
                        "test_assembly\\escape.obj", "C:/escape.obj", "./escape.obj",
                        "test_assembly/safe.obj\x00hidden"):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                MODULE.validated_members(good_catalog() + [member(invalid)])
        with self.assertRaises(ValueError):
            MODULE.validated_members(good_catalog() + good_catalog()[:1])

    def test_symlink_member_rejected(self):
        link = member("test_assembly/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            MODULE.validated_members(good_catalog() + [link])

    def test_wrong_task_members_and_oversized_payload_rejected(self):
        for catalog in (good_catalog()[:-1],
                        good_catalog() + [member(MODULE.MEMBER_PREFIX + "run.py")],
                        good_catalog() + [member(MODULE.MEMBER_PREFIX + "sub/0.obj")]):
            with self.assertRaises(ValueError):
                MODULE.validated_members(catalog)
        catalog = good_catalog()
        catalog[0].file_size = 1024 * 1024 + 1
        with self.assertRaises(ValueError):
            MODULE.validated_members(catalog)

    def test_e_drive_rejected_before_filesystem_inspection(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("filesystem touched")):
            for value in ("E:/anything", "e:\\anything", "/mnt/e/anything", "/mnt/e"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    MODULE.checked_path(value, MODULE.RUN)

    def test_unapproved_destinations_rejected_before_filesystem_inspection(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("filesystem touched")):
            for value in ("C:/dev/disassembly_codex_ready_v2/runs/old_run",
                          "C:/dev/disassembly_codex_ready_v2/runs/../old_run",
                          "//host/share/input"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    MODULE.checked_path(value, MODULE.RUN)


if __name__ == "__main__":
    unittest.main()
