#!/usr/bin/env bash
#
# Stop docker services.

source bin/set_docker_runtime.sh

docker-compose down
