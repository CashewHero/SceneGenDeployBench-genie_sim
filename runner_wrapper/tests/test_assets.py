from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from runner_wrapper import assets


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"public checkpoint test"
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.asset_patch = mock.patch.dict(
            assets.ASSETS, {"sharp": ("sharp.pt", "https://example.invalid/model.pt", self.digest, len(self.data))}
        )
        self.asset_patch.start()
        self.addCleanup(self.asset_patch.stop)
        self.env_patch = mock.patch.dict(os.environ, {"PATH_MODEL_CACHE": str(self.root / "cache")})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.override_patch = mock.patch.dict(os.environ)
        self.override_patch.start()
        self.addCleanup(self.override_patch.stop)
        os.environ.pop("GENIESIM_SHARP_CHECKPOINT", None)

    def test_download_publishes_and_cache_reuses_verified_file(self) -> None:
        with (
            mock.patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(self.data)) as download,
            mock.patch.object(assets, "publish_file", wraps=assets.publish_file) as publish,
        ):
            path = assets.ensure_checkpoint("sharp", self.root)
            self.assertEqual(path.read_bytes(), self.data)
            self.assertEqual(assets.ensure_checkpoint("sharp", self.root), path)
            self.assertEqual(download.call_count, 1)
            publish.assert_called_once()
            self.assertFalse(list(self.root.glob("checkpoint-*")))

    def test_checksum_failure_never_publishes(self) -> None:
        with (
            mock.patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt")),
            mock.patch.object(assets, "publish_file") as publish,
            self.assertRaisesRegex(ValueError, "SHA256"),
        ):
            assets.ensure_checkpoint("sharp", self.root)
        publish.assert_not_called()
        self.assertFalse((self.root / "cache/genie_sim/checkpoints/sharp.pt").exists())

    def test_override_is_verified_without_copy_or_download(self) -> None:
        path = self.root / "override.pt"
        path.write_bytes(self.data)
        with (
            mock.patch.dict(os.environ, {"GENIESIM_SHARP_CHECKPOINT": str(path)}),
            mock.patch.object(assets.urllib.request, "urlopen") as download,
        ):
            self.assertEqual(assets.ensure_checkpoint("sharp", self.root), path)
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                assets.ensure_checkpoint("sharp", self.root)
        download.assert_not_called()

    def test_concurrent_download_is_single_publication(self) -> None:
        barrier = threading.Barrier(2)

        def fetch() -> Path:
            barrier.wait(timeout=5)
            return assets.ensure_checkpoint("sharp", self.root)

        with mock.patch.object(
            assets.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(self.data)
        ) as download:
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(lambda _: fetch(), range(2)))
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(download.call_count, 1)

    def test_network_error_is_reported(self) -> None:
        with (
            mock.patch.object(assets.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")),
            self.assertRaisesRegex(ConnectionError, "offline"),
        ):
            assets.ensure_checkpoint("sharp", self.root)


if __name__ == "__main__":
    unittest.main()
