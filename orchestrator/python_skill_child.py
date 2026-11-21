"""Trusted disposable evaluator. Never exec/import generated source."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from orchestrator.python_skill_program import MAX_INPUT_BYTES, evaluate


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("python_input_byte_limit")
        packet = json.loads(raw)
        result = evaluate(packet["program"], packet["request"], packet["max_actions"],
                          extension=packet.get("extension"), diagnose=packet.get('diagnose', False),
                          observe_abstention=packet.get('observe_abstention', False))
        output = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(output) > 65536:
            raise ValueError("python_output_limit")
    except Exception as error:
        message = str(error)
        output = json.dumps({"error": message if message.startswith("python_") and len(message) < 120
                             else "python_runtime_error:" + type(error).__name__}).encode("utf-8")
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
