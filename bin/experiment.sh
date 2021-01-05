#!/usr/bin/env bash
#
# Run experiment session.
#
# Run "bin/experiment.sh --help" to see available options.

source bin/set_docker_runtime.sh

docker-compose run --rm project \
  python -m src.experiment.experiment_session "$@"
