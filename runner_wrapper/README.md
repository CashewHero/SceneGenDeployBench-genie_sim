# Genie Sim PanoRecon runner

The `genie_sim` generator accepts one full 2:1 equirectangular RGB `image` and publishes the original merged Gaussian PLY as semantic `3dgs`. It invokes the public `geniesim_world create` CLI inside the shared server's disposable job process. It uses the repo's DA360 depth, cubemap split, depth-guided SHARP, culling, and merge. It does not install Isaac Sim, ROS, ComfyUI, or panorama generation models.

The HTTP server and filesystem publication helpers are unchanged. See [Runner API](docs/api.md) for the wire contract and [adaptation instructions](AGENTS.md) for maintenance rules.

## Defaults and job parameters

The default is upstream's intended inference configuration: DA360 Large, six 1536×1536 faces, depth-guided SHARP, bilateral cube-depth refinement, inverse-depth resizing, 91-degree PLY frustum culling, and radius compression above 40 scene units. There is no global depth clip or super-resolution.

The catalog is [config/runners/genie_sim.yaml](config/runners/genie_sim.yaml). Its example scheduling is four sequential jobs per batch, two attempts, a 30-minute job timeout including downloads, and a five-minute startup timeout. These are conservative limits; the local check below measures one job, not a full batch.

Only three job parameters are exposed:

| Parameter | Default | Options |
|---|---|---|
| `da360_model` | `large` | `small`, `base`, `large` |
| `cube_face_size` | `1536` | Integer from 64 to 4096 |
| `pano_depth_engine` | `da360` | `da360`, `sharp`, `both`, `fuse` |

Smaller DA360 weights reduce depth-estimation cost. Smaller cube faces reduce projection work and intermediate files, but SHARP still runs at its fixed internal 1536×1536 resolution. The `sharp` engine skips DA360. `both` and `fuse` add panorama SHARP depth inference and cost more than the default. There are no training steps in this feed-forward pipeline.

The original flow processes the six faces sequentially on one device. It has no multi-GPU inference path. Multiple DeployBench instances can use separate GPUs.

## Checkpoints and paths

The first valid job downloads missing public weights to `PATH_MODEL_CACHE/genie_sim/checkpoints`. Downloads are locked, checked against pinned SHA256 values, staged in the job workspace, and published with `publish_file`. No credentials are required. Weights are not bundled in the image.

- SHARP uses Apple's `sharp_2572gikvuh.pt`, about 2.81 GB.
- DA360 uses [hljiang/DA360](https://huggingface.co/hljiang/DA360/tree/b3bebecef1237c088b66034f5033ef628de5dc0a), pinned to that revision. Large is about 1.34 GB, Base 391 MB, and Small 100 MB.
- `GENIESIM_SHARP_CHECKPOINT` and `GENIESIM_DA360_CHECKPOINT` optionally select existing mounted files. They must match the selected public weights and are checksum-verified without copying.
- `GENIESIM_DEVICE` defaults to `cuda:0` within the container's visible GPUs.
- `PATH_MODEL_CACHE` is required unless all selected checkpoints use explicit overrides. The shared server supplies the job workspace and publishes final outputs to `runtime.output_dir`.

The Dockerfile pins Python 3.11, Torch 2.8 with CUDA 12.8, xformers, ml-sharp, and DA360. It keeps third-party source and licenses in the image but omits the rest of the simulator. CUDA architectures are configured as 7.5, 8.0, 8.6, and 8.9+PTX; inference uses prebuilt Torch and xformers binaries and needs no local CUDA compilation.

## Output and scale

Only the final `3DGS-scene-<id>.ply` is reusable output. The server also publishes `runner-scene-<id>.log` and `metrics-scene-<id>.json`. Intermediate cubes, depth EXRs, and per-face splats remain in the disposable job workspace. The report records resolved parameters, output mapping, Gaussian count, depth units, and measured resources. Failed jobs retain their log and report and do not publish a partial scene.

The merged scene uses `LDB` coordinates: positive X is left, Y is down, and Z is back relative to the input panorama center. The build verifies all six original cube transforms. No wrapper rotation or color conversion is applied.

`scene_scale=1.0` is an **uncalibrated starting value**, not a claim of metric accuracy. Default `da360` and `both` use min-normalized relative depth. `sharp` and `fuse` use SHARP-derived meter units. Calibrate the chosen mode against the benchmark before setting a deployment-wide scale.

## Local checks

From the repository root:

```bash
# Lightweight host tests, in an isolated Python environment.
python3 -m venv .venv
source .venv/bin/activate
pip install pillow==11.3.0 numpy==1.26.4 plyfile==1.1.2
runner_wrapper/localtest.sh test

runner_wrapper/localtest.sh build

# Use a real, full 2:1 panorama, not the old wrapper's text-file fixture.
RUNNER_SMOKE_IMAGE=/absolute/path/panorama.png \
RUNNER_DATA_DIR=/absolute/path/local-runner-data \
runner_wrapper/localtest.sh smoke

runner_wrapper/localtest.sh status
runner_wrapper/localtest.sh logs
runner_wrapper/localtest.sh down
```

The smoke command uses the example request with intended defaults and runs as the host UID/GID. `RUNNER_GPUS` takes Docker's GPU selector and defaults to `all`. To select one GPU, use `RUNNER_GPUS=device=0` or `RUNNER_GPUS=device=GPU-<uuid>`, not a bare index. Set `RUNNER_HOST_PORT` for a different port or `RUNNER_REQUEST_FILE` for job overrides. Each run requires a new `RUNNER_CONTAINER` name if the previous container still exists. `down` stops the container but does not delete it.

For cache overrides in this helper, supply paths inside the mounted data tree, such as `GENIESIM_SHARP_CHECKPOINT=/data/model_cache/existing/sharp_2572gikvuh.pt`.

The local GPU smoke passed on an RTX 2080 Ti with 11 GiB VRAM using a 1024×512 TartanAir panorama, DA360 Large, and all six 1536×1536 faces. With cached checkpoints, the job took 113 seconds, peaked at 9.9 GiB VRAM and 7.9 GiB host memory, and published a validated PLY containing 6,833,608 Gaussians. All 31 contract and adapter tests passed in the built image as a non-root user. This verifies inference and publication, not scene-quality or scale calibration. Do not silently reduce intended defaults to make a smoke test pass.

## Image publication

The checked-in [image workflow](../.github/workflows/runner-image.yaml) is adapted from the wrapper's template and uses `mode=min` build caching. It publishes version-tagged images to GHCR after contract tests. Build and run a real local GPU smoke test before tagging a release. The configured first image is `ghcr.io/cashewhero/scenegendeploybench-genie_sim:0.1.0`.

To recreate the shared workflow template before model-specific edits:

```bash
mkdir -p .github/workflows
cp runner_wrapper/examples/github-workflows/build-runner-image.yaml \
  .github/workflows/runner-image.yaml
```
