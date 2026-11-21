"""Data-only child entry; never inherits provider credentials."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from orchestrator.pose_guard_runtime import evaluate

try:
    payload = sys.stdin.buffer.read(512001)
    if len(payload) > 512000:
        raise ValueError("input_limit")
    result = evaluate(**json.loads(payload))
    sys.stdout.write(json.dumps(result, allow_nan=False))
except Exception as error:
    sys.stdout.write(json.dumps(dict(error=type(error).__name__)))
