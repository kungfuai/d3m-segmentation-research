#!/usr/bin/env bash
#
# Run coverage report

source bin/set_docker_runtime.sh

docker-compose run --rm --env-file /dev/null --entrypoint coverage SERVICE_NAME \
  run -m pytest -o no:warnings "$@" tests/
docker-compose run --rm --env-file /dev/null --entrypoint coverage SERVICE_NAME \
  report -m
