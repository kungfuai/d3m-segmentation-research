#!/usr/bin/env bash
#
# Format python source files using black formatter.

source bin/set_docker_runtime.sh

if [ $# -eq 0 ]; then
  TARGET=.
else
  TARGET=$*
fi

docker-compose run --rm --entrypoint black SERVICE_NAME ${TARGET}
