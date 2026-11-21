"""CAD observations for independent ranking, without a heuristic or verifier call.

The loader uses the same final CAD transforms as ASAP.  These observations are
ranking inputs, not collision, stability, blocking, or removability judgments.
Python 3.8 compatible; importing this module does not import ASAP or NumPy.
"""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Optional


FRAME = "upstream load_assembly(transform='final') CAD frame; not a sampled stable pose"
FEATURE_KIND = "independent_cad_features"


def _json_copy(value: Any) -> Any:
    """Reject non-JSON and nonfinite metadata instead of serializing guesses."""
    return json.loads(json.dumps(value, allow_nan=False))


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _vector(value: Any, length: int) -> Optional[list]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result = [_finite_number(item) for item in value]
    return None if any(item is None for item in result) else result


def _mesh_description(mesh: Any) -> Dict[str, Any]:
    raw_bounds = mesh.bounds.tolist() if hasattr(mesh.bounds, "tolist") else mesh.bounds
    if not isinstance(raw_bounds, (tuple, list)) or len(raw_bounds) != 2:
        raise ValueError("invalid_mesh_bounds")
    bounds = [_vector(row, 3) for row in raw_bounds]
    if any(row is None for row in bounds):
        raise ValueError("invalid_mesh_bounds")
    if any(bounds[0][axis] > bounds[1][axis] for axis in range(3)):
        raise ValueError("reversed_mesh_bounds")
    extents = [bounds[1][axis] - bounds[0][axis] for axis in range(3)]
    if any(not math.isfinite(extent) for extent in extents):
        raise ValueError("nonfinite_mesh_extents")
    center = _vector(mesh.center_mass, 3)
    volume = _finite_number(mesh.volume)
    watertight = bool(mesh.is_watertight)
    winding = getattr(mesh, "is_winding_consistent", None)
    winding = bool(winding) if winding is not None else None
    mass_properties_reliable = bool(
        watertight and winding is True and volume is not None and volume > 0
        and center is not None)
    reasons = []
    if not watertight:
        reasons.append("mesh_not_watertight")
    if winding is not True:
        reasons.append("winding_consistency_unconfirmed")
    if volume is None or volume <= 0:
        reasons.append("nonpositive_or_nonfinite_signed_volume")
    if center is None:
        reasons.append("nonfinite_or_invalid_center_mass")
    return {
        "bounds": bounds,
        "extents": extents,
        "bounds_center": [bounds[0][axis] / 2 + bounds[1][axis] / 2 for axis in range(3)],
        "center_mass": center,
        "volume": volume,
        "watertight": watertight,
        "validity": {
            "bounds_finite": True,
            "winding_consistent": winding,
            "center_mass_finite": center is not None,
            "volume_finite": volume is not None,
            "center_mass_status": "uniform_density_estimate" if mass_properties_reliable else "uncertain",
            "mass_properties_assume_uniform_density": True,
            "mass_properties_uncertainty": reasons,
            "physical_material_density_known": False,
        },
    }


def _contact_edges(contact_graph: Any, part_ids: set) -> Optional[list]:
    if contact_graph is None:
        return None
    # Accept an already observed networkx-like graph or an explicit edge list.
    # Nothing in this function computes contacts or calls the physical solver.
    edges = contact_graph.edges() if callable(getattr(contact_graph, "edges", None)) else contact_graph
    normalized = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("invalid_contact_edge")
        left, right = edge
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("contact_ids_must_be_strings")
        if left not in part_ids or right not in part_ids or left == right:
            raise ValueError("contact_edge_outside_assembly")
        normalized.add(tuple(sorted((left, right))))
    return [list(edge) for edge in sorted(normalized)]


def describe_loaded(
    assembly: Mapping[str, Any],
    input_sha256: Mapping[str, str],
    assembly_id: str,
    source_commit: str,
    base_part: Optional[str],
    solver_settings: Mapping[str, Any],
    contact_graph: Any = None,
) -> Dict[str, Any]:
    """Describe already final-transformed meshes; pure apart from mesh properties.

    Callers supplying fixtures are responsible for marking them as fixtures in
    their run evidence.  This helper never certifies a fixture as real physics.
    All supplied parts are retained.  IDs are sorted only for serialization;
    their order is not a geometric preference or a suggested removal order.
    """
    if not isinstance(assembly, Mapping) or not assembly:
        raise ValueError("empty_or_invalid_assembly")
    if any(not isinstance(part, str) or not part for part in assembly):
        raise ValueError("part_ids_must_be_nonempty_strings")
    if not isinstance(assembly_id, str) or not assembly_id:
        raise ValueError("assembly_id_required")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError("source_commit_must_be_full_sha1")
    if base_part is not None and base_part not in assembly:
        raise ValueError("base_part_outside_assembly")
    if not isinstance(solver_settings, Mapping):
        raise ValueError("solver_settings_must_be_mapping")
    hashes = {}
    for name, digest in sorted(input_sha256.items()):
        if not isinstance(name, str) or not name or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("invalid_input_file_hash")
        hashes[name] = digest.lower()
    if not hashes:
        raise ValueError("input_file_hashes_required")
    parts = {}
    for part in sorted(assembly):
        data = assembly[part]
        if not isinstance(data, Mapping) or "mesh" not in data:
            raise ValueError("assembly_mesh_missing")
        if data.get("transform", "final") != "final":
            raise ValueError("features_require_final_transformed_mesh")
        parts[part] = _mesh_description(data["mesh"])
    edges = _contact_edges(contact_graph, set(parts))
    return {
        "schema_version": 2,
        "kind": FEATURE_KIND,
        "task": {
            "assembly_id": assembly_id,
            "input_sha256": hashes,
            "source_commit": source_commit.lower(),
            "base_part": base_part,
            "solver_settings": _json_copy(dict(solver_settings)),
            "goal": {"type": "native_full_disassembly"},
        },
        "parts": parts,
        "relations": {
            "contact_edges": edges,
            "contact_source": "not_measured" if edges is None else "caller_supplied_observed_graph",
            "contact_graph_recomputed": False,
            "contact_graph_frame_verified": None,
            "contact_is_blocker_proof": False,
            "frame": FRAME,
            "units": "input_CAD_units_not_verified",
            "known_precedence_constraints": None,
            "precedence_source": "not_measured",
        },
        "role": "CAD observations for ranking; no physical feasibility or blocker verdict",
        "verifier_calls": 0,
        "prior_plan_or_verdicts_used": False,
    }


def filter_features(features: Mapping[str, Any], assembled_parts: Iterable[str]) -> Dict[str, Any]:
    """Return a snapshot with removed meshes/relations excluded, never mutate input.

    Input hashes and goal still identify the original task.  Surviving CAD
    coordinates remain in the final assembly frame, not the current sampled
    stable pose.  Removed bodies are excluded from the observation, not from
    ASAP's own task or search state.
    """
    if features.get("schema_version") != 2 or features.get("kind") != FEATURE_KIND:
        raise ValueError("not_independent_cad_features")
    current = list(assembled_parts)
    if any(not isinstance(part, str) for part in current) or len(current) != len(set(current)):
        raise ValueError("invalid_assembled_parts")
    if not set(current).issubset(features["parts"]):
        raise ValueError("assembled_part_missing_from_features")
    result = copy.deepcopy(dict(features))
    present = set(current)
    result["parts"] = {part: result["parts"][part] for part in sorted(current)}
    edges = result["relations"]["contact_edges"]
    if edges is not None:
        result["relations"]["contact_edges"] = [edge for edge in edges if set(edge).issubset(present)]
    # Do not silently retain future, unfiltered state-specific relation types.
    constraints = result["relations"].get("known_precedence_constraints")
    if constraints is not None:
        raise ValueError("precedence_filter_not_implemented")
    result["state"] = {"assembled_parts": sorted(current), "frame": FRAME}
    return result


def _safe_path(path: Path) -> Path:
    # The project explicitly excludes E:.  Reject it before filesystem access.
    if (re.match(r"^[eE]:", str(path)) or str(path).lower() == "/mnt/e"
            or str(path).lower().startswith("/mnt/e/")):
        raise ValueError("excluded_drive")
    absolute = path.absolute()
    for item in reversed((absolute,) + tuple(absolute.parents)):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("input_links_not_supported")
    return absolute


def _source_hashes(directory: Path) -> Dict[str, str]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".obj", ".json"):
            continue
        _safe_path(path)
        if not path.is_file():
            raise ValueError("source_input_not_file")
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    if not any(name.endswith(".obj") for name in result):
        raise ValueError("no_upstream_obj_inputs")
    return result


def _source_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        universal_newlines=True, stderr=subprocess.PIPE,
    ).strip()


def _load_final_assembly(root: Path, directory: Path) -> Mapping[str, Any]:
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        module = importlib.import_module("assets.load")
        expected = root / "assets" / "load.py"
        if Path(module.__file__).absolute() != expected:
            raise ValueError("loaded_assets_module_from_different_checkout")
        return module.load_assembly(str(directory), transform="final")
    finally:
        sys.path[:] = old_path


def describe_task(
    asap_root: Path,
    assembly_dir: Path,
    base_part: Optional[str],
    solver_settings: Mapping[str, Any],
    contact_graph: Any = None,
) -> Dict[str, Any]:
    """Read official ASAP CAD meshes and prove OBJ/JSON inputs stayed unchanged.

    No solver, heuristic, GNN, contact computation or prior plan is invoked.
    A caller may supply an already observed contact graph; its provenance must
    be retained separately in that run.  This function writes no output files.
    """
    root = _safe_path(Path(asap_root))
    directory = _safe_path(Path(assembly_dir))
    commit = _source_commit(root)
    before = _source_hashes(directory)
    loaded = _load_final_assembly(root, directory)
    try:
        assembly_id = directory.relative_to(root / "assets").as_posix()
    except ValueError:
        assembly_id = directory.name
    result = describe_loaded(loaded, before, assembly_id, commit, base_part,
                             solver_settings, contact_graph)
    after = _source_hashes(directory)
    if before != after:
        raise ValueError("source_inputs_changed_during_feature_extraction")
    result["input_integrity"] = {
        "before_sha256": before,
        "after_sha256": after,
        "unchanged": True,
    }
    return result
