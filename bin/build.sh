#!/usr/bin/env bash
#
# Build docker image.

source bin/set_docker_runtime.sh

docker-compose build
