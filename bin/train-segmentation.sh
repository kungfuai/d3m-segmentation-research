#!/usr/bin/env bash
#
# Train model
#
# Run "bin/train.sh --help" to see available options.

source bin/set_docker_runtime.sh

docker-compose run --rm project \
  python -m src.training.segmentation_session "$@"
