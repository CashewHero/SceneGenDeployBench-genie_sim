from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from runner_wrapper import adapter
from runner_wrapper.server import Runner, RunnerSettings


def write_scene(path: Path, count: int = 3) -> None:
    keys = [
        "x",
        "y",
        "z",
        *[f"f_dc_{i}" for i in range(3)],
        "opacity",
        *[f"scale_{i}" for i in range(3)],
        *[f"rot_{i}" for i in range(4)],
    ]
    vertex = np.zeros(count, dtype=[(key, "f4") for key in keys])
    vertex["z"] = -2
    vertex["rot_0"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def fake_model(command: list[str]) -> None:
    work = Path(command[command.index("--work-dir") + 1])
    pano = Path(command[command.index("--panorama") + 1])
    write_scene(work / pano.stem / "merged_gaussians.ply")
    print("fake model test output")


def mocked_adapter_job(request: dict) -> dict:
    # The shared server uses spawn, so install test doubles inside the child.
    with (
        mock.patch.object(adapter, "ensure_checkpoint", return_value=Path("/cache/model.pt")),
        mock.patch.object(adapter, "invoke_model", side_effect=fake_model),
    ):
        return adapter.run_job(request)


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "pano.png"
        Image.new("RGB", (128, 64), (70, 110, 150)).save(self.image)
        self.request = {
            "contract_version": 1,
            "job": {
                "job_id": "unit-job",
                "batch_id": "unit-batch",
                "primary_sample": "sample-1",
                "parameters": {},
                "timeout_seconds": 10,
            },
            "inputs": {"data": {"sample-1": {"image": str(self.image)}}},
            "runtime": {"workspace_dir": str(self.root / "workspace")},
        }
        monitor = mock.Mock()
        monitor.stop.return_value = [
            {
                "namespace": "performance",
                "name": "wall_time_ms",
                "type": "float",
                "value": 1.0,
                "unit": "ms",
                "source": "runner",
            }
        ]
        self.monitor_patch = mock.patch.object(adapter, "ResourceMonitor", return_value=monitor)
        self.monitor_patch.start()
        self.addCleanup(self.monitor_patch.stop)

    def test_upstream_defaults_and_simple_overrides(self) -> None:
        self.assertEqual(adapter.normalize_parameters({}), adapter.DEFAULT_PARAMETERS)
        self.assertEqual(adapter.normalize_parameters({"da360_model": "small"})["da360_model"], "small")
        for bad in (
            {"steps": 1},
            {"cube_face_size": True},
            {"cube_face_size": 1},
            {"cube_face_size": 1536.0},
            {"da360_model": "other"},
            {"pano_depth_engine": "other"},
            [],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                adapter.normalize_parameters(bad)

    def test_primary_panorama_validation(self) -> None:
        self.assertEqual(adapter.panorama_input(self.request), self.image)
        bad = copy.deepcopy(self.request)
        bad["inputs"]["data"]["other"] = {"image": str(self.image)}
        with self.assertRaises(ValueError):
            adapter.panorama_input(bad)
        Image.new("RGB", (100, 100)).save(self.image)
        with self.assertRaisesRegex(ValueError, "2:1"):
            adapter.panorama_input(self.request)

    def test_rejects_perspective_and_other_roles(self) -> None:
        self.request["job"]["primary_sample_metadata"] = {"projection": "perspective"}
        with self.assertRaisesRegex(ValueError, "equirectangular"):
            adapter.panorama_input(self.request)
        self.request["job"].pop("primary_sample_metadata")
        self.request["inputs"]["candidate"] = {"other": {"3dgs": "scene.ply"}}
        with self.assertRaisesRegex(ValueError, "only inputs.data"):
            adapter.panorama_input(self.request)

    def test_command_preserves_original_flow(self) -> None:
        command = adapter.model_command(
            self.image,
            self.root,
            adapter.DEFAULT_PARAMETERS,
            {"sharp": Path("/cache/sharp.pt"), "da360": Path("/cache/DA360_large.pth")},
        )
        self.assertEqual(command[:2], ["geniesim_world", "create"])
        self.assertEqual(command[command.index("--cube-face-size") + 1], "1536")
        for flag in ("--skip-sharp", "--skip-merge", "--no-depth-gt-init", "--depth-max", "--super-sample"):
            self.assertNotIn(flag, command)
        sharp = adapter.model_command(
            self.image,
            self.root,
            adapter.normalize_parameters({"pano_depth_engine": "sharp"}),
            {"sharp": Path("/cache/sharp.pt")},
        )
        self.assertNotIn("--da360-checkpoint", sharp)

    def test_invokes_the_public_cli_in_the_job_process(self) -> None:
        module = mock.Mock()
        with mock.patch.dict(sys.modules, {"geniesim_world.cli_pano": module}):
            adapter.invoke_model(["geniesim_world", "create", "--panorama", "/input.png"])
        module.cli.main.assert_called_once_with(
            args=["create", "--panorama", "/input.png"], prog_name="geniesim_world", standalone_mode=False
        )

    def test_success_and_metrics_mapping(self) -> None:
        with (
            mock.patch.object(adapter, "ensure_checkpoint", return_value=Path("/cache/model.pt")),
            mock.patch.object(adapter, "invoke_model", side_effect=fake_model),
        ):
            result = adapter.run_job(self.request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output_metadata"], {"scene_scale": 1.0, "scene_coordinate_system": "LDB"})
        output = self.root / "workspace" / result["output_files"]["sample-1"]["3dgs"]
        self.assertEqual(adapter.validate_scene(output), 3)
        report = json.loads((output.parent / result["artifacts"][1]["path"]).read_text())
        self.assertEqual(report["output_files"], result["output_files"])
        self.assertEqual(report["model"]["depth_units"], "relative")
        self.assertIn("fake model test output", (output.parent / result["artifacts"][0]["path"]).read_text())

    def test_server_publishes_only_final_scene_and_artifacts(self) -> None:
        settings = RunnerSettings(
            port=0,
            runner_name="genie_sim",
            runner_type="generator",
            runner_version="0.1.0",
            contract_version=1,
            idle_timeout_seconds=900,
            startup_timeout_seconds=60,
            adapter_target="unused:handler",
        )
        runner = Runner(settings=settings, run_job_handler=mocked_adapter_job, state="idle", batch_id="unit-batch")
        request = copy.deepcopy(self.request)
        request["runtime"] = {"output_dir": str(self.root / "output")}
        result = runner._run_job_in_child_process(request, timeout_seconds=10, kill_after_seconds=15)
        self.assertEqual(result["status"], "completed", result)
        names = {path.name for path in (self.root / "output").iterdir()}
        self.assertEqual(
            names, {result["output_files"]["sample-1"]["3dgs"], *[artifact["path"] for artifact in result["artifacts"]]}
        )

    def test_failure_has_logs_metrics_and_no_output(self) -> None:
        with (
            mock.patch.object(adapter, "ensure_checkpoint", return_value=Path("/cache/model.pt")),
            mock.patch.object(adapter, "invoke_model", side_effect=RuntimeError("CUDA inference failed")),
        ):
            result = adapter.run_job(self.request)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("output_files", result)
        self.assertIn("CUDA inference failed", result["failure"]["message"])
        self.assertEqual(len(result["artifacts"]), 2)

    def test_download_failure_is_retryable(self) -> None:
        with mock.patch.object(adapter, "ensure_checkpoint", side_effect=ConnectionError("download failed")):
            result = adapter.run_job(self.request)
        self.assertTrue(result["failure"]["retryable"])

    def test_invalid_input_does_not_download(self) -> None:
        self.request["job"]["parameters"] = {"cube_face_size": 0}
        with mock.patch.object(adapter, "ensure_checkpoint") as download:
            result = adapter.run_job(self.request)
        download.assert_not_called()
        self.assertEqual(result["failure"]["code"], "INVALID_INPUT")

    def test_ply_validation(self) -> None:
        path = self.root / "scene.ply"
        write_scene(path, 0)
        with self.assertRaises(ValueError):
            adapter.validate_scene(path)
        write_scene(path)
        ply = PlyData.read(path, mmap=False)
        ply["vertex"].data["x"][0] = np.nan
        ply.write(path)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            adapter.validate_scene(path)


if __name__ == "__main__":
    unittest.main()
