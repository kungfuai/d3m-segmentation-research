#!/usr/bin/env bash
#
# Run evaluation session.
#
# Run "bin/evaluate.sh --help" to see available options.

source bin/set_docker_runtime.sh

docker-compose run --rm project \
  python -m src.evaluation.evaluation_session "$@"
