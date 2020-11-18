#!/usr/bin/env bash
#
# Lint python source code using pylint static analysis tool.

source bin/set_docker_runtime.sh

docker-compose run --rm --entrypoint pylint SERVICE_NAME src/*.py src/**/*.py tests/*.py tests/**/*.py
