#!/usr/bin/env bash
# ===========================================================================
#  WeldForge -- one command.
#
#    ./weldforge.sh                 install what is missing, build the docs
#                                   and the executables, then start the studio
#    ./weldforge.sh run             just start it
#    ./weldforge.sh install docs    install and build the docs, do not start
#    ./weldforge.sh --force         reinstall from scratch
#    ./weldforge.sh --help          every option
#
#  Everything it does lives in tools/bootstrap.py, which imports nothing but
#  the standard library so that it can run before anything is installed.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Once the virtual environment exists, use its interpreter directly: it
# already has everything, and it keeps the system Python out of the way.
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  cat >&2 <<'MESSAGE'

  Python was not found on PATH.

  Install Python 3.10 or newer, then run this again:
    macOS          brew install python@3.12
    Debian/Ubuntu  sudo apt install python3 python3-venv
    Fedora         sudo dnf install python3

MESSAGE
  exit 1
fi

exec "$PY" tools/bootstrap.py "$@"
