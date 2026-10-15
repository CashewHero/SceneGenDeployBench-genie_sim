"""DeployBench adapter for the original Genie Sim World PanoRecon CLI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from runner_wrapper.assets import ensure_checkpoint
from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = {"cube_face_size": 1536, "da360_model": "large", "pano_depth_engine": "da360"}


def normalize_parameters(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("job.parameters must be an object")
    unknown = raw.keys() - DEFAULT_PARAMETERS.keys()
    if unknown:
        raise ValueError(f"Unknown job parameters: {', '.join(sorted(unknown))}")
    parameters = {**DEFAULT_PARAMETERS, **raw}
    size = parameters["cube_face_size"]
    if type(size) is not int or not 64 <= size <= 4096:
        raise ValueError("cube_face_size must be an integer between 64 and 4096")
    if parameters["da360_model"] not in ("small", "base", "large"):
        raise ValueError("da360_model must be small, base, or large")
    if parameters["pano_depth_engine"] not in ("da360", "sharp", "both", "fuse"):
        raise ValueError("pano_depth_engine must be da360, sharp, both, or fuse")
    return parameters


def panorama_input(request: dict[str, Any]) -> Path:
    from PIL import Image

    job = request["job"]
    inputs = request.get("inputs", {})
    primary = job["primary_sample"]
    if inputs.get("candidate") or inputs.get("references"):
        raise ValueError("This generator accepts only inputs.data")
    data = inputs.get("data", {})
    if set(data) != {primary} or set(data[primary]) != {"image"}:
        raise ValueError("Provide exactly one primary sample with semantic type image")
    raw_path = data[primary]["image"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError("image must be an absolute file path")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Panorama not found: {path}")
    projection = job.get("primary_sample_metadata", {}).get("projection")
    if projection not in (None, "equirectangular"):
        raise ValueError("The image must be an equirectangular panorama")
    with Image.open(path) as image:
        width, height = image.size
        if width != 2 * height:
            raise ValueError(f"Expected a full 2:1 panorama, received {width}x{height}")
        image.verify()
    return path


def model_command(
    panorama: Path, work_dir: Path, parameters: dict[str, Any], checkpoints: dict[str, Path]
) -> list[str]:
    command = [
        "geniesim_world",
        "create",
        "--panorama",
        str(panorama),
        "--work-dir",
        str(work_dir),
        "--device",
        os.getenv("GENIESIM_DEVICE", "cuda:0"),
        "--checkpoint-path",
        str(checkpoints["sharp"]),
        "--cube-face-size",
        str(parameters["cube_face_size"]),
        "--pano-depth-engine",
        parameters["pano_depth_engine"],
    ]
    if parameters["pano_depth_engine"] != "sharp":
        command.extend(
            [
                "--da360-root",
                str(REPO_ROOT / "source/external/DA360"),
                "--da360-checkpoint",
                str(checkpoints["da360"]),
            ]
        )
    return command


def validate_scene(path: Path) -> int:
    import numpy as np
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    properties = [
        "x",
        "y",
        "z",
        *[f"f_dc_{i}" for i in range(3)],
        *[f"scale_{i}" for i in range(3)],
        *[f"rot_{i}" for i in range(4)],
    ]
    if not len(vertex) or not set([*properties, "opacity"]).issubset(vertex.dtype.names or ()):
        raise ValueError("The model output is not a nonempty 3DGS PLY")
    for key in properties:
        if not np.isfinite(vertex[key]).all():
            raise ValueError(f"The model produced nonfinite Gaussian {key} values")
    # +/- infinity encodes exactly zero/one opacity in the standard logit format.
    if np.isnan(vertex["opacity"]).any():
        raise ValueError("The model produced NaN opacity values")
    return len(vertex)


def invoke_model(command: list[str]) -> None:
    # Invoke the public Click CLI in the server's disposable job process.
    # The shared timeout/shutdown then stops inference without orphaning a GPU subprocess.
    from geniesim_world.cli_pano import cli

    cli.main(args=command[1:], prog_name=command[0], standalone_mode=False)


def _timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    workspace = Path(request["runtime"]["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(
        {"job_id": request["job"]["job_id"], "parameters": request["job"].get("parameters")}, sort_keys=True
    )
    variant = hashlib.sha256(identity.encode()).hexdigest()[:10]
    log_path = workspace / f"runner-scene-{variant}.log"
    report_path = workspace / f"metrics-scene-{variant}.json"
    monitor = ResourceMonitor(
        sample_data={
            "image": request.get("inputs", {}).get("data", {}).get(request["job"]["primary_sample"], {}).get("image")
        },
        output_dir=workspace,
    )
    with tee_job_output(log_path):
        monitor.start()
        result: dict[str, Any] = {"started_at": _timestamp(started_at), "failure": None}
        report: dict[str, Any] = {"inputs": request.get("inputs", {})}
        stage = "input"
        try:
            parameters = normalize_parameters(request["job"].get("parameters", {}))
            panorama = panorama_input(request)
            report["parameters"] = parameters
            stage = "checkpoint"
            checkpoints = {"sharp": ensure_checkpoint("sharp", workspace)}
            if parameters["pano_depth_engine"] != "sharp":
                checkpoints["da360"] = ensure_checkpoint(f"da360_{parameters['da360_model']}", workspace)
            model_dir = workspace / "model"
            command = model_command(panorama, model_dir, parameters, checkpoints)
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            print(f"Running original PanoRecon CLI: {' '.join(command)}", flush=True)
            stage = "inference"
            invoke_model(command)
            stage = "output"
            merged = model_dir / panorama.stem / "merged_gaussians.ply"
            count = validate_scene(merged)
            output_name = f"3DGS-scene-{variant}.ply"
            shutil.move(str(merged), workspace / output_name)
            outputs = {request["job"]["primary_sample"]: {"3dgs": output_name}}
            # Median of 100 converged TartanAir test trajectories for DA360 Large.
            # Other depth modes and weights have not been calibrated.
            scene_scale = 0.69 if parameters["pano_depth_engine"] == "da360" and parameters["da360_model"] == "large" else 1.0
            metadata = {"scene_scale": scene_scale, "scene_coordinate_system": "LDB"}
            result.update(status="completed", output_files=outputs, output_metadata=metadata)
            report = {
                "inputs": report["inputs"],
                "output_files": outputs,
                "parameters": parameters,
                "output_metadata": metadata,
                "model": {
                    "gaussian_count": count,
                    "depth_units": "relative" if parameters["pano_depth_engine"] in ("da360", "both") else "meters",
                },
            }
            print(f"PanoRecon produced {count} Gaussians", flush=True)
        except Exception as exc:
            traceback.print_exc()
            retryable = isinstance(exc, (ConnectionError, TimeoutError))
            result.update(
                status="failed",
                failure={
                    "code": "INVALID_INPUT" if stage == "input" else "MODEL_ERROR",
                    "message": f"{exc}; see {log_path.name}",
                    "retryable": retryable,
                    "stage": stage,
                },
            )
            report["failure"] = result["failure"]
        metrics = monitor.stop()
        report["resource_metrics"] = metrics
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result.update(
            completed_at=_timestamp(time.time()),
            metrics=metrics,
            artifacts=[
                {"artifact_type": "job_log", "path": log_path.name},
                {"artifact_type": "metric_summary", "path": report_path.name},
            ],
        )
        return result
