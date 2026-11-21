"""Disposable, metered separation diagnostics; never native-exec generated code."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from orchestrator.python_skill_program import MAX_INPUT_BYTES
from orchestrator.python_separation_probe import inspect


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES+1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError('python_input_byte_limit')
        packet = json.loads(raw)
        result = inspect(packet['program'], packet['request'], packet['choice'])
        out = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(out) > 65536:
            raise ValueError('python_output_limit')
    except Exception as error:
        message = str(error)
        out = json.dumps(dict(accepted=False, errors=[message if message.startswith('python_') and len(message)<120
            else 'python_separation_diagnostic_error:'+type(error).__name__], api_calls=0, physical_calls=0)).encode('utf-8')
    sys.stdout.buffer.write(out)


if __name__ == '__main__':
    main()
