#!/usr/bin/env bash
#
# Run unit tests.

source bin/set_docker_runtime.sh

docker-compose --env-file ./.env.test -f docker-compose.test.yml run --use-aliases --rm SERVICE_NAME
