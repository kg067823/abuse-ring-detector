"""Cross-platform AbuseRing Command Center launcher."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/model_f_r1_bundle.pkl"
EXPECTED = "3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff"


def verify_artifact() -> None:
    if not ARTIFACT.exists():
        raise SystemExit(f"R1 artifact missing: {ARTIFACT}")
    actual = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise SystemExit(f"R1 checksum mismatch: {actual}")


def wait_ready(url: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url.rstrip("/") + "/readiness", timeout=3) as response:
                if response.status == 200:
                    return
                last = f"HTTP {response.status}"
        except Exception as exc:
            last = str(exc)
        time.sleep(2)
    raise SystemExit(f"API did not become ready: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true", help="start API and Redis with Docker Compose")
    parser.add_argument("--api-url", default=os.getenv("ABUSERING_API_URL", "http://localhost:8000"))
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    verify_artifact()
    token = os.getenv("ADMIN_KILL_SWITCH_TOKEN", "demo-secret")
    os.environ.update({"DEMO_MODE": "true", "SHADOW_MODE": "true", "ENFORCE_DECISIONS": "false", "ABUSERING_API_URL": args.api_url, "ABUSERING_ADMIN_TOKEN": token})
    if args.docker:
        env = os.environ.copy()
        env["ADMIN_KILL_SWITCH_TOKEN"] = token
        subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=ROOT, env=env, check=True)
    wait_ready(args.api_url)
    print("AbuseRing Command Center — DEMO / SYNTHETIC DATA")
    print(f"API: {args.api_url} | UI: http://localhost:8501")
    print(f"Model F-R1: {EXPECTED} | SHADOW_MODE=true | ENFORCE_DECISIONS=false")
    if args.no_ui:
        return 0
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app/command_center.py", "--server.port", "8501"], cwd=ROOT, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
