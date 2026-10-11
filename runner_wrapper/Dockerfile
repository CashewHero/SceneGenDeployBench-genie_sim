# syntax=docker/dockerfile:1.7
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/runner PATH=/opt/runner/bin:${PATH} UV_LINK_MODE=copy
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential ninja-build python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --break-system-packages --no-cache-dir uv==0.12.5 && \
    uv python install 3.11 && uv venv --python 3.11 /opt/runner

# Pin the private SHARP API used by upstream's predictor subclass and DA360's network implementation.
ARG SHARP_COMMIT=aed6527499ef91cba3b54c18d49a870f25947190
ARG DA360_COMMIT=a0e9d33d9bcfabd6842b255e4ce0ba728db52853
WORKDIR /app/model_repo
RUN mkdir -p source/external/ml-sharp source/external/DA360 && \
    curl -fSL --retry 3 https://github.com/apple/ml-sharp/archive/${SHARP_COMMIT}.tar.gz | \
      tar -xz --strip-components=1 -C source/external/ml-sharp && \
    curl -fSL --retry 3 https://github.com/Insta360-Research-Team/DA360/archive/${DA360_COMMIT}.tar.gz | \
      tar -xz --strip-components=1 -C source/external/DA360

COPY --link runner_wrapper/requirements.txt /build/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install -r /build/requirements.txt && \
    uv pip install --no-deps source/external/ml-sharp
COPY --link source/geniesim_world source/geniesim_world
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps source/geniesim_world && uv pip check

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ENV VIRTUAL_ENV=/opt/runner PATH=/opt/runner/bin:${PATH} PYTHONPATH=/app/model_repo \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1 \
    RUNNER_PORT=58090 RUNNER_NAME=genie_sim RUNNER_TYPE=generator RUNNER_VERSION=0.1.0 \
    RUNNER_CONTRACT_VERSION=1 RUNNER_IDLE_TIMEOUT_SECONDS=900 \
    RUNNER_ADAPTER=runner_wrapper.adapter:run_job GENIESIM_DEVICE=cuda:0
COPY --link --from=builder /opt/python /opt/python
COPY --link --from=builder /opt/runner /opt/runner
COPY --link --from=builder /app/model_repo/source/external /app/model_repo/source/external
COPY --link source/geniesim_world /app/model_repo/source/geniesim_world
COPY --link runner_wrapper /app/model_repo/runner_wrapper
WORKDIR /app/model_repo
RUN --network=none geniesim_world create --help && \
    python -c "import sys; sys.path.insert(0, 'source/external/DA360'); import networks; from runner_wrapper.adapter import run_job; print(networks.DA360.__name__, run_job.__name__)" && \
    python -m runner_wrapper.verify_geometry
EXPOSE 58090
CMD ["python", "-m", "runner_wrapper.server"]
