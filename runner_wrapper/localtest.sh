#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${RUNNER_IMAGE:-scenegendeploybench-genie_sim-runner:local}"
CONTAINER="${RUNNER_CONTAINER:-genie-sim-localtest}"
HOST_PORT="${RUNNER_HOST_PORT:-58090}"
DATA_DIR="${RUNNER_DATA_DIR:-${REPO_ROOT}/data}"
REQUEST_FILE="${RUNNER_REQUEST_FILE:-${SCRIPT_DIR}/examples/generator_job_request.json}"

run_tests() {
  PYTHONPATH="${REPO_ROOT}" python3 -m unittest discover -s "${SCRIPT_DIR}/tests" -v
}

build_image() {
  docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"
}

run_container() {
  : "${RUNNER_SMOKE_IMAGE:?Set RUNNER_SMOKE_IMAGE to a real absolute-path 2:1 panorama}"
  [[ "${RUNNER_SMOKE_IMAGE}" = /* && -f "${RUNNER_SMOKE_IMAGE}" ]] || {
    echo "RUNNER_SMOKE_IMAGE must be an existing absolute file path" >&2
    return 1
  }
  if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "Container ${CONTAINER} already exists. Choose RUNNER_CONTAINER with a new name." >&2
    return 1
  fi
  mkdir -p "${DATA_DIR}/datasets/smoke" "${DATA_DIR}/model_cache" "${DATA_DIR}/pipelines" "${DATA_DIR}/output"
  local env_args=()
  local name
  for name in GENIESIM_SHARP_CHECKPOINT GENIESIM_DA360_CHECKPOINT; do
    if [[ -n "${!name:-}" ]]; then
      env_args+=(-e "${name}=${!name}")
    fi
  done
  docker run -d --name "${CONTAINER}" --gpus "${RUNNER_GPUS:-all}" \
    --user "$(id -u):$(id -g)" -p "127.0.0.1:${HOST_PORT}:58090" \
    -e PATH_DATASETS=/data/datasets -e PATH_MODEL_CACHE=/data/model_cache \
    -e PATH_OUTPUT=/data/output -e PATH_PIPELINES=/data/pipelines \
    "${env_args[@]}" \
    --mount "type=bind,source=${DATA_DIR},target=/data" \
    --mount "type=bind,source=${RUNNER_SMOKE_IMAGE},target=/data/datasets/smoke/image.png,readonly" \
    "${IMAGE}"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/status" | \
      python3 -c 'import json,sys; sys.exit(json.load(sys.stdin)["state"] != "idle")' 2>/dev/null; then
      echo "Runner idle at http://127.0.0.1:${HOST_PORT}"
      return
    fi
    sleep 1
  done
  docker logs "${CONTAINER}" >&2
  return 1
}

case "${1:-help}" in
  test) run_tests ;;
  build) build_image ;;
  run) run_container ;;
  smoke)
    run_container
    python3 "${SCRIPT_DIR}/smoke.py" --port "${HOST_PORT}" --request "${REQUEST_FILE}"
    ;;
  status) curl -fsS "http://127.0.0.1:${HOST_PORT}/status" ;;
  logs) docker logs "${CONTAINER}" ;;
  down) docker stop "${CONTAINER}" ;;
  *)
    echo "Usage: runner_wrapper/localtest.sh {test|build|run|smoke|status|logs|down}"
    echo "Smoke requires RUNNER_SMOKE_IMAGE and a built image. Use RUNNER_DATA_DIR outside the repo for cached weights and outputs."
    ;;
esac
