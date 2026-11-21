#!/usr/bin/env bash
# Run only in the dedicated ASAP-Ubuntu distribution, as root.
# Each phase is called separately through tools/run_recorded.py.
set -euo pipefail
phase=${1:?phase required}
root=/opt/asap-research
conda=$root/miniconda
worker=$root/envs/asap
source=/mnt/c/dev/disassembly_codex_ready_v2/external/ASAP
mkdir -p "$root"
case "$phase" in
  apt)
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git curl ca-certificates bzip2 unzip \
      libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxi-dev libxrandr-dev \
      libxxf86vm-dev libxinerama-dev libxcursor-dev libspatialindex-dev \
      xvfb xauth mesa-utils ffmpeg
    ;;
  bootstrap)
    test ! -e "$conda"
    curl --fail --location --retry 3 \
      https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh \
      --output "$root/miniconda-installer.sh"
    sha256sum "$root/miniconda-installer.sh"
    bash "$root/miniconda-installer.sh" -b -p "$conda"
    "$conda/bin/conda" --version
    ;;
  environment)
    test ! -e "$worker"
    "$conda/bin/python" - "$source/environment.yml" "$root" <<'PY'
import pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
packages, requirements = [], []
inside_dependencies = False
for line in lines:
    if line == 'dependencies:':
        inside_dependencies = True
    elif inside_dependencies and line.startswith('    - '):
        requirements.append(line[6:])
    elif inside_dependencies and line.startswith('  - ') and line != '  - pip:':
        packages.append(line[4:])
assert 'python=3.8.0' in packages and 'numpy==1.21.4' in requirements
root = pathlib.Path(sys.argv[2])
(root / 'conda-spec.txt').write_text('\n'.join(packages) + '\n')
(root / 'requirements.txt').write_text('\n'.join(requirements) + '\n')
PY
    "$conda/bin/conda" create -y -p "$worker" --file "$root/conda-spec.txt"
    "$worker/bin/python" -m pip install 'torch==1.13.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
    "$worker/bin/python" -m pip install -r "$root/requirements.txt"
    "$worker/bin/python" -m pip install torch_scatter==2.1.1+pt113cpu torch_sparse==0.6.16+pt113cpu \
      --find-links https://data.pyg.org/whl/torch-1.13.0+cpu.html
    "$worker/bin/python" -m pip check
    ;;
  checkout)
    test ! -e "$root/ASAP"
    git clone --no-hardlinks "$source" "$root/ASAP"
    git -C "$root/ASAP" rev-parse HEAD
    git -C "$root/ASAP" status --porcelain
    ;;
  build)
    cd "$root/ASAP/simulation"
    "$worker/bin/python" setup.py install
    "$worker/bin/python" -c 'import redmax_py; print(redmax_py.__file__)'
    ;;
  *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
