"""Offline feature-contract fixtures; not actual ASAP or physical verification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.independent_features import describe_loaded, describe_task, filter_features


class MeshFixture:
    """Explicit fake mesh properties; accessing raw vertices is a test failure."""
    bounds = [[0, 0, 0], [2, 4, 6]]
    center_mass = [1, 2, 3]
    volume = 48
    is_watertight = True
    is_winding_consistent = True

    @property
    def vertices(self):
        raise AssertionError("raw vertices must not be sent")


def assembly():
    return {part: {"mesh": MeshFixture(), "transform": "final"} for part in ("base", "10", "2")}


def description(meshes=None, **changes):
    args = dict(assembly=assembly() if meshes is None else meshes,
                input_sha256={"2.obj": "b" * 64, "config.json": "a" * 64},
                assembly_id="official/fixture", source_commit="c" * 40,
                base_part="base", solver_settings={"seed": 0, "budget": 100})
    args.update(changes)
    return describe_loaded(**args)


class IndependentFeatureTests(unittest.TestCase):
    def test_geometric_fields_and_unknowns_are_explicit(self):
        value = description()
        self.assertEqual(value["schema_version"], 2)
        self.assertEqual(value["kind"], "independent_cad_features")
        self.assertEqual(list(value["parts"]), ["10", "2", "base"])
        self.assertEqual(value["parts"]["2"]["extents"], [2, 4, 6])
        self.assertEqual(value["parts"]["2"]["bounds_center"], [1, 2, 3])
        self.assertEqual(value["parts"]["2"]["validity"]["center_mass_status"], "uniform_density_estimate")
        self.assertIsNone(value["relations"]["contact_edges"])
        self.assertEqual(value["relations"]["contact_source"], "not_measured")
        self.assertIsNone(value["relations"]["known_precedence_constraints"])
        self.assertEqual(value["verifier_calls"], 0)
        self.assertEqual(value["task"]["goal"], {"type": "native_full_disassembly"})
        encoded = json.dumps(value, allow_nan=False)
        for forbidden in ("native_order", "heur_out_score", "raw_vertices", "past_success_order"):
            self.assertNotIn(forbidden, encoded)

    def test_open_mesh_does_not_claim_valid_mass_properties(self):
        meshes = assembly()
        meshes["2"]["mesh"].is_watertight = False
        value = description(meshes)
        validity = value["parts"]["2"]["validity"]
        self.assertEqual(validity["center_mass_status"], "uncertain")
        self.assertIn("mesh_not_watertight", validity["mass_properties_uncertainty"])

    def test_bad_mass_properties_are_null_not_json_nan(self):
        meshes = assembly()
        meshes["2"]["mesh"].center_mass = [float("nan"), 0, 0]
        meshes["2"]["mesh"].volume = float("inf")
        value = description(meshes)
        self.assertIsNone(value["parts"]["2"]["center_mass"])
        self.assertIsNone(value["parts"]["2"]["volume"])
        json.dumps(value, allow_nan=False)

    def test_negative_signed_volume_is_uncertain(self):
        meshes = assembly()
        meshes["2"]["mesh"].volume = -48
        self.assertEqual(description(meshes)["parts"]["2"]["validity"]["center_mass_status"], "uncertain")

    def test_invalid_bounds_fail_closed(self):
        for bounds in ([[0, 0], [1, 2]], [[2, 0, 0], [1, 1, 1]], [[0, 0, 0], [float("nan"), 1, 1]]):
            with self.subTest(bounds=bounds):
                meshes = assembly()
                meshes["2"]["mesh"].bounds = bounds
                with self.assertRaises(ValueError):
                    description(meshes)

    def test_observed_edges_normalized_not_called_blockers(self):
        value = description(contact_graph=[("base", "2"), ("2", "base"), ("10", "2")])
        self.assertEqual(value["relations"]["contact_edges"], [["10", "2"], ["2", "base"]])
        self.assertFalse(value["relations"]["contact_is_blocker_proof"])
        self.assertIsNone(value["relations"]["contact_graph_frame_verified"])

    def test_empty_measured_graph_differs_from_unmeasured(self):
        value = description(contact_graph=[])
        self.assertEqual(value["relations"]["contact_edges"], [])
        self.assertEqual(value["relations"]["contact_source"], "caller_supplied_observed_graph")

    def test_networkx_like_graph_is_accepted_without_mutation(self):
        class GraphFixture:
            def edges(self):
                return [("2", "base")]
        self.assertEqual(description(contact_graph=GraphFixture())["relations"]["contact_edges"], [["2", "base"]])

    def test_unknown_contact_part_and_self_loops_rejected(self):
        for edges in ([('unknown', '2')], [('2', '2')], [(2, 'base')], [('2', 'base', 1)]):
            with self.subTest(edges=edges):
                with self.assertRaises(ValueError):
                    description(contact_graph=edges)

    def test_nonfinal_frame_rejected(self):
        meshes = assembly()
        meshes["2"]["transform"] = "initial"
        with self.assertRaisesRegex(ValueError, "final_transformed"):
            description(meshes)

    def test_invalid_task_metadata_rejected(self):
        for changes in ({"source_commit": "unknown"}, {"base_part": "missing"},
                        {"input_sha256": {}}, {"input_sha256": {"2.obj": "bad"}},
                        {"solver_settings": {"timeout": float("inf")}}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    description(**changes)

    def test_filter_drops_removed_parts_and_edges_without_changing_original(self):
        original = description(contact_graph=[("10", "2"), ("2", "base")])
        saved = copy.deepcopy(original)
        filtered = filter_features(original, ["base", "2"])
        self.assertEqual(list(filtered["parts"]), ["2", "base"])
        self.assertEqual(filtered["relations"]["contact_edges"], [["2", "base"]])
        self.assertEqual(filtered["state"]["assembled_parts"], ["2", "base"])
        self.assertEqual(filtered["task"]["input_sha256"], original["task"]["input_sha256"])
        filtered["parts"]["2"]["bounds"][0][0] = 99
        self.assertEqual(original, saved)

    def test_filter_preserves_unmeasured_contacts(self):
        self.assertIsNone(filter_features(description(), ["base"])["relations"]["contact_edges"])

    def test_filter_rejects_unknown_duplicate_or_bad_schema(self):
        for remaining in (["missing"], ["2", "2"], [2]):
            with self.subTest(remaining=remaining):
                with self.assertRaises(ValueError):
                    filter_features(description(), remaining)
        with self.assertRaises(ValueError):
            filter_features({"schema_version": 1}, ["2"])

    def test_loader_input_integrity_is_measured_before_load(self):
        with tempfile.TemporaryDirectory(prefix="independent_feature_fixture_") as tmp:
            root = Path(tmp)
            path = root / "fixture"
            path.mkdir()
            (path / "2.obj").write_text("fixture-only-not-real-geometry", encoding="utf-8")
            (path / "config.json").write_text("{}", encoding="utf-8")
            with patch("worker.independent_features._source_commit", return_value="c" * 40), patch(
                    "worker.independent_features._load_final_assembly", return_value=assembly()) as loader:
                value = describe_task(root, path, "base", {"seed": 0})
            loader.assert_called_once_with(root, path)
            self.assertTrue(value["input_integrity"]["unchanged"])
            self.assertEqual(value["input_integrity"]["before_sha256"], value["input_integrity"]["after_sha256"])
            self.assertEqual(set(value["task"]["input_sha256"]), {"2.obj", "config.json"})

    def test_changed_or_added_input_during_load_is_rejected(self):
        for added in (False, True):
            with self.subTest(added=added), tempfile.TemporaryDirectory(prefix="independent_feature_fixture_") as tmp:
                root = Path(tmp)
                (root / "2.obj").write_text("fixture", encoding="utf-8")
                def mutating_loader(checkout, directory):
                    (directory / ("new.json" if added else "2.obj")).write_text("changed fixture", encoding="utf-8")
                    return assembly()
                with patch("worker.independent_features._source_commit", return_value="c" * 40), patch(
                        "worker.independent_features._load_final_assembly", side_effect=mutating_loader):
                    with self.assertRaisesRegex(ValueError, "source_inputs_changed"):
                        describe_task(root, root, "base", {})


if __name__ == "__main__":
    unittest.main()
