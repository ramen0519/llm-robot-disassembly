"""Record a local command without interpreting its exit code as plan success.

Use only for reviewed commands; never pass secrets in argv. Standard library only.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def run_recorded(command: list[str], cwd: Path, output: Path) -> dict:
    """Create an exclusive run directory and preserve raw process output."""
    if not command:
        raise ValueError("command must not be empty")
    cwd = cwd.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "command": command, "cwd": str(cwd),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exit_code": None, "launch_error": None, "plan_success": None,
    }
    manifest = output / "command.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    start = time.monotonic()
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
            record["exit_code"] = result.returncode
        except OSError as error:
            record["launch_error"] = str(error)
            stderr.write(str(error).encode("utf-8"))
    record["elapsed_seconds"] = time.monotonic() - start
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run_recorded(command, args.cwd, args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["exit_code"] if result["exit_code"] is not None else 127)
