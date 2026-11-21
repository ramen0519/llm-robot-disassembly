"""Lab motor, one moving gripper and one fixed-frame jig. Run in the ASAP WSL environment."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PREVIOUS = RUNS / "lab_motor_asap_3h_20261117"
NATIVE = RUNS / "lab_motor_full_asap_20261117"
sys.path.insert(0, str(NATIVE))
import run_native as baseline


def protocol():
    rig = baseline.load(ROOT / "configs/lab_motor_frame_jig.json")
    cfg = baseline.load(PREVIOUS / "protocol.json")
    assert baseline.sha(PREVIOUS / "protocol.json") == "4fb0efaf603d6def0e7f1dad31fc8bf3312f1b10d6c84ad08fc171d3b6a478fa"
    assert rig["product"] == cfg["product"] and rig["fixed_frame"] == cfg["frame"]
    assert rig["moving_grippers"] == 1 and rig["extra_supported_parts"] == 0
    # Native serial stability counts the base in max_fix. The mover uses one more slot.
    assert rig["asap_max_grippers"] == rig["moving_grippers"] + 1 + rig["extra_supported_parts"]
    cfg["max_grippers"] = rig["asap_max_grippers"]
    return cfg


def check(cfg):
    from plan_sequence import feasibility_check
    from plan_sequence.planner import base

    with (PREVIOUS / "heur-out/events.jsonl").open(encoding="utf8") as stream:
        events = [json.loads(line) for line in stream]
    witness = next(e for e in events if e["kind"] == "stability_end")
    attempt = next(e for e in events if e["kind"] == "simulate_end")
    assert witness["fixed"] == [cfg["frame"]] and witness["success"] is False
    assert witness["parts_fall"] == ["3", "30"] and attempt["part_move"] == "64"
    planner = object.__new__(base.SequencePlanner)
    planner.base_part = cfg["frame"]
    planner.asset_folder = planner.assembly_dir = None
    planner.save_sdf = False
    cases = [
        ("recorded_instability", (False, witness["parts_fall"]), False),
        ("one_extra_hold_needed", (False, ["3"]), False),
        ("frame_enough", (True, []), True),
    ]
    for label, reply, accepted in cases:
        fixed_at_call = []

        def recorded_reply(asset, folder, fixed, free, *args):
            # ASAP mutates the fix list after the check returns.
            fixed_at_call.append(tuple(fixed))
            return reply

        with patch.object(base, "check_assemblable", return_value=([1, 0, 0], [])), \
                patch.object(feasibility_check, "check_stable", side_effect=recorded_reply):
            sim = planner._simulate(attempt["part_move"], attempt["parts_rest"], [], None, cfg["max_grippers"])
        assert fixed_at_call == [(cfg["frame"],)]
        assert sim["feasible"] == accepted and sim["parts_fix"] == ([] if accepted else None)
        print(label, "PASS")
    print("max_grippers=2 extra_supported_parts=0; replay/fixtures only, physics and API calls=0")


def execute(cfg):
    # Each new run gets independent outputs and the same source geometry as the interrupted run.
    features = baseline.load(PREVIOUS / "geometry_features.json")
    features["task"]["solver_settings"]["max_grippers"] = cfg["max_grippers"]
    folder = Path(tempfile.mkdtemp(prefix="lab_motor_frame_jig_", dir=str(RUNS)))
    print("Run directory:", folder, flush=True)
    baseline.write_new(folder / "protocol.json", cfg)
    baseline.write_new(folder / "geometry_features.json", features)
    manifest = baseline.load(PREVIOUS / "input_manifest.json")
    manifest["method_scope_sha256"] = baseline.sha(folder / "protocol.json")
    baseline.write_new(folder / "input_manifest.json", manifest)

    source_load = baseline.load
    previous_gnn = NATIVE / "gnn_native/result.json"
    assert baseline.sha(previous_gnn) == "41f55be75ab7ba9909791bf922280989249e1cd85a0d388f864b964d64e9ca2a"

    def load_with_gnn_receipt(path):
        return source_load(previous_gnn if path == folder / "gnn_native/result.json" else path)

    with patch.object(baseline, "HERE", folder), patch.object(baseline, "load", load_with_gnn_receipt):
        return baseline.run("heur-out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "execute"))
    mode = parser.parse_args().mode
    with baseline.network_disabled():
        cfg = protocol()
        if mode == "check":
            check(cfg)
        else:
            sys.exit(execute(cfg))
