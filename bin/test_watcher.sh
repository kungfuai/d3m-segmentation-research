#!/usr/bin/env bash
#
# Watch project files and run tests when changes are detected. Specific files and
# directories can be given as arguments, otherwise all unit tests are run.

source bin/set_docker_runtime.sh

if [ $# -eq 0 ]; then
    TARGET="tests"
else
    TARGET="$*"
fi

# exit on interrupt
trap "exit" INT

while true; do
    find src tests -name '*.py' | entr -dc docker-compose run --rm \
        app python -m pytest -p no:warnings "$TARGET" -s -vv
done
