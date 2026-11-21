"""Plot saved ASAP paths without running physics (Python 3.8 worker only).

The figure is an XY/XZ projection of recorded simulator coordinates, not an
assembly rendering or a collision/stability test. Existing artifacts are read
only; the output PNG is created exclusively. No provider/API calls are made.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from remote_rank_client import checked_path


MAX_JSON_BYTES = 64 * 1024 * 1024


def _json(path):
    path = checked_path(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("recorded path JSON exceeds visualization input limit")
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("recorded path artifact must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _referenced_file(root, reference):
    if type(reference) is not str or not reference or any(c in reference for c in "\x00\n\r"):
        raise ValueError("invalid path sidecar reference")
    relative = reference.replace("\\", "/")
    # Reject forbidden drive and traversal strings before touching a path.
    if (PureWindowsPath(reference).drive or PurePosixPath(relative).is_absolute()
            or ":" in relative or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("sidecar must be a relative path within the selected run")
    return checked_path(root / relative)


def _path(value, np):
    raw = value.get("path")
    if type(raw) is not list or not raw:
        raise ValueError("nonempty recorded trajectory required")
    if any(type(row) is not list or len(row) != 6 or
           any(type(item) not in (int, float) for item in row) for row in raw):
        raise ValueError("trajectory must contain recorded xyz/euler six-vectors")
    states = np.asarray(raw, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6 or not np.isfinite(states).all():
        raise ValueError("recorded trajectory must be finite")
    return states


def _switch_index(repaired, states, outcome, np):
    controller = repaired.get("controller", {})
    actions = controller.get("actions")
    switch_q = controller.get("switch_q")
    if (controller.get("switch_count") != 1 or type(actions) is not list
            or len(actions) != 2 or type(actions[1].get("step")) is not int
            or actions[1]["step"] < 1 or type(switch_q) is not list or len(switch_q) != 6):
        raise ValueError("the selected trajectory has no single observed force switch")
    switch_q = np.asarray(switch_q, dtype=float)
    if not np.isfinite(switch_q).all():
        raise ValueError("switch state must be finite")
    matches = np.flatnonzero(np.all(np.isclose(states, switch_q, rtol=0, atol=1e-12), axis=1))
    if len(matches) == 0:
        raise ValueError("observed force switch does not occur in recorded trajectory")
    # Pinned check_success adds an extra starting sample at each frame block.
    # The match is authoritative; the expected index disambiguates stationary
    # or repeated samples using the recorded switch step and block size.
    frame_skip = outcome.get("scope", {}).get("physics", {}).get("frame_skip")
    step = actions[1]["step"]
    expected = step + (step - 1) // frame_skip if type(frame_skip) is int and frame_skip > 0 else step
    return int(min(matches, key=lambda value: abs(int(value) - expected)))


def render(run_dir, output):
    # Import after lexical guards; no renderer or simulator initialization.
    root, target = checked_path(run_dir), checked_path(output)
    if target.suffix.lower() != ".png":
        raise ValueError("output must be a new PNG path")
    if target.exists():
        raise FileExistsError("existing figure must not be overwritten")
    if not target.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    result, result_sha = _json(root / "path_outcomes.json")
    outcomes = result.get("outcomes")
    if type(outcomes) is not list or not outcomes or any(type(item) is not dict for item in outcomes):
        raise ValueError("at least one recorded path-skill outcome is required")
    eligible = [item for item in outcomes if item.get("feasible") is True]
    outcome = (eligible or outcomes)[0]
    failed_file = _referenced_file(root, outcome.get("failed_path"))
    repaired_file = _referenced_file(root, outcome.get("trajectory_file"))
    failed, failed_sha = _json(failed_file)
    repaired, repaired_sha = _json(repaired_file)
    if failed.get("raw_success") is not False:
        raise ValueError("the referenced single-force attempt is not recorded as failed")
    if outcome.get("trajectory_sha256") != repaired_sha:
        raise ValueError("repaired trajectory file does not match the recorded hash")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    first, second = _path(failed, np), _path(repaired, np)
    switch = _switch_index(repaired, second, outcome, np)
    if not np.allclose(first[0], second[0], rtol=0, atol=1e-12):
        raise ValueError("failed and repair trajectories do not share the recorded start")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    colors = {"failed": "#838b95", "first": "#dc831f", "second": "#2878b5"}
    for ax, vertical, label in zip(axes, (1, 2), ("Y", "Z")):
        ax.plot(first[:, 0], first[:, vertical], color=colors["failed"], linewidth=2,
                linestyle="--", alpha=.8, label="Single-force attempt (failed)")
        ax.plot(second[:switch + 1, 0], second[:switch + 1, vertical],
                color=colors["first"], linewidth=2.8, label="Repair: first force")
        ax.plot(second[switch:, 0], second[switch:, vertical],
                color=colors["second"], linewidth=2.8, label="Repair: second force")
        ax.scatter([first[-1, 0]], [first[-1, vertical]], color=colors["failed"],
                   marker="x", s=65, zorder=5, label="Failed endpoint")
        ax.scatter([second[0, 0]], [second[0, vertical]], color="#17212b", s=38,
                   zorder=6, label="Start")
        ax.scatter([second[switch, 0]], [second[switch, vertical]], color="#964b9b", s=48,
                   zorder=7, label="Observed force switch")
        ax.scatter([second[-1, 0]], [second[-1, vertical]], color="#2b8b57", marker="*", s=110,
                   zorder=6, label="Repair endpoint")
        ax.set_title("X" + label + " projection")
        ax.set_xlabel("X (ASAP world units)")
        ax.set_ylabel(label + " (ASAP world units)")
        ax.grid(True, alpha=.22)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(.12)
    part = str(outcome.get("scope", {}).get("part_move", "unknown"))
    passed = outcome.get("feasible") is True
    noncollinear = repaired.get("geometry", {}).get("noncollinear") is True
    subtitle = ("Part {0} | full verifier: {1} | noncollinear recorded motion: {2}"
                .format(part, "PASS" if passed else "not confirmed", noncollinear))
    fig.suptitle("Actual recorded ASAP trajectory\n" + subtitle, fontsize=13, y=.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06),
               ncol=4, frameon=False, fontsize=8.5)
    fig.text(.5, .022, "ASAP world units (not calibrated metres). Actual recorded paths; no additional physics.",
             ha="center", fontsize=9, color="#47515b")
    fig.tight_layout(rect=(0, .16, 1, .9))
    try:
        with target.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return {"status": "CREATED", "output": str(target), "source_run": str(root),
            "outcomes_sha256": result_sha, "failed_path_sha256": failed_sha,
            "repaired_path_sha256": repaired_sha,
            "failed_path_states": len(first), "repair_path_states": len(second),
            "switch_path_index": switch, "full_verifier_pass": passed,
            "noncollinear_recorded_motion": noncollinear,
            "physics_calls": 0, "api_calls": 0,
            "interpretation": "saved_simulator_coordinate_projection_not_new_verification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
