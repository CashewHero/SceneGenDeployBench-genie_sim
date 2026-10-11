"""Submit a local HTTP smoke job and check the published result."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=58090)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    from pathlib import Path

    payload = json.loads(Path(args.request).read_text())
    endpoint = f"http://127.0.0.1:{args.port}"
    request = urllib.request.Request(
        endpoint + "/run-job", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        accepted = json.load(response)
    if not accepted.get("accepted"):
        raise RuntimeError(accepted)
    deadline = time.monotonic() + payload["job"]["timeout_seconds"] + 90
    while time.monotonic() < deadline:
        with urllib.request.urlopen(endpoint + "/status", timeout=10) as response:
            status = json.load(response)
        if status["state"] in ("finished", "failed"):
            print(json.dumps(status, indent=2))
            result = status["result"]
            if status["state"] != "finished" or result["status"] != "completed":
                raise RuntimeError(result.get("failure"))
            outputs = result["output_files"][payload["job"]["primary_sample"]]
            if set(outputs) != {"3dgs"} or not outputs["3dgs"].endswith(".ply"):
                raise RuntimeError(f"Unexpected output mapping: {outputs}")
            kinds = {artifact["artifact_type"] for artifact in result["artifacts"]}
            if not {"job_log", "metric_summary"}.issubset(kinds):
                raise RuntimeError("The runner omitted its log or metrics report")
            return
        time.sleep(2)
    raise TimeoutError("The smoke job did not reach a terminal state")


if __name__ == "__main__":
    main()
