#!/usr/bin/env bash
set -euo pipefail
root=/opt/asap-research
worker=$root/envs/asap
test -x "$worker/bin/python"
"$worker/bin/python" --version
cmake --version
g++ --version
# Same upstream C++/Release flags; cap parallel compilation for 7.5 GiB WSL RAM.
cmake -S "$root/ASAP/simulation" -B "$root/build-redmax" \
  -DPYTHON_EXECUTABLE="$worker/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$worker/lib/python3.8/site-packages"
cmake --build "$root/build-redmax" --target redmax_py --parallel 2
"$worker/bin/python" -c 'import redmax_py; print(redmax_py.__file__)'
