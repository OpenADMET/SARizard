#!/usr/bin/env bash
# Build osmordred (a C++ extension, not on PyPI) from source into the active
# sarizard-osmordred environment. Run after creating and activating that env:
#   conda env create -f envs/osmordred.yml
#   conda activate sarizard-osmordred
#   bash envs/build_osmordred.sh
#
# Requires the build toolchain from envs/osmordred.yml (cmake, ninja, a C++ compiler,
# boost, eigen, lapack, rdkit). Override the source ref with OSMORDRED_REF (defaults to
# the repo's main branch); pin a commit for full reproducibility.
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "no active conda env; run: conda activate sarizard-osmordred" >&2
    exit 1
fi

REPO="${OSMORDRED_REPO:-https://github.com/osmoai/osmordred}"
REF="${OSMORDRED_REF:-main}"
SRC="${OSMORDRED_SRC:-$(mktemp -d)/osmordred}"

echo "cloning $REPO@$REF -> $SRC"
git clone --depth 1 --branch "$REF" "$REPO" "$SRC"

# conda-forge installs Eigen under include/eigen3/Eigen, but the project's CMake find_path
# looks for Eigen/Dense directly under include/, so expose it there
if [[ ! -e "$CONDA_PREFIX/include/Eigen" ]]; then
    ln -s "$CONDA_PREFIX/include/eigen3/Eigen" "$CONDA_PREFIX/include/Eigen"
fi

# the build backend (setup.py + CMakeLists.txt) lives in skbuild/; setup.py reads ./README.md
cd "$SRC/skbuild"
cp -f ../README.md .
rm -rf _skbuild src/osmordred.egg-info dist
python -m build --wheel
pip install dist/osmordred-*.whl --force-reinstall

# verify from outside the source tree: importing while inside skbuild/ shadows the installed
# compiled package with the uncompiled osmordred/ source subdir (noted in the repo README)
( cd "$(mktemp -d)" && python -c "import osmordred; print('osmordred built and importable')" )
