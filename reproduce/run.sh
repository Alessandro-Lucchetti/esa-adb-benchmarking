#!/bin/sh
# Build the reproduction image and run scripts/reproduce.py with the repository mounted.
#
#   reproduce/run.sh                      every stage, from the dataset download to the report
#   reproduce/run.sh report               one stage; any argument of scripts/reproduce.py
#
# The report is written to runs/reproduce/report.md. Works from Linux, macOS and Git Bash
# on Windows (Docker Desktop).
set -e
cd "$(dirname "$0")/.."
docker build -t esa-adb-reproduce reproduce

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # keep /work as a container path; Docker Desktop maps the Windows folder
        export MSYS_NO_PATHCONV=1
        exec docker run --rm -v "$(pwd -W):/work" esa-adb-reproduce "$@" ;;
    *)
        exec docker run --rm -v "$(pwd):/work" --user "$(id -u):$(id -g)" esa-adb-reproduce "$@" ;;
esac
