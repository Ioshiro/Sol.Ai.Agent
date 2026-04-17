#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

LK_VERSION = "v2.16.0"
LK_URL = (
    "https://github.com/livekit/livekit-cli/releases/download/"
    f"{LK_VERSION}/lk_2.16.0_linux_amd64.tar.gz"
)
LK_BIN = Path("/usr/local/bin/lk")
LIVEKIT_API = "http://livekit:7880"
TRUNK_FILE = Path("/work/sip/trunk.json")
DISPATCH_FILE = Path("/work/sip/dispatch-rule.json")


def wait_for_livekit() -> None:
    print("Waiting for LiveKit API...", flush=True)
    while True:
        try:
            with urllib.request.urlopen(LIVEKIT_API, timeout=2):
                return
        except Exception:
            time.sleep(2)


def install_lk() -> None:
    print(f"Installing LiveKit CLI {LK_VERSION}...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        archive_path = Path(tmp.name)

    try:
        urllib.request.urlretrieve(LK_URL, archive_path)
        with tempfile.TemporaryDirectory() as extract_dir:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir)

            candidates = [path for path in Path(extract_dir).rglob("lk") if path.is_file()]
            if not candidates:
                raise RuntimeError("lk binary not found in LiveKit CLI tarball")

            LK_BIN.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], LK_BIN)
            LK_BIN.chmod(0o755)
    finally:
        archive_path.unlink(missing_ok=True)


def run_lk(*args: str) -> int:
    completed = subprocess.run(
        [str(LK_BIN), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.rstrip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), flush=True)
    return completed.returncode


def main() -> int:
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"

    install_lk()
    wait_for_livekit()

    print("Creating SIP inbound trunk...", flush=True)
    run_lk("sip", "inbound", "create", str(TRUNK_FILE))

    print("Creating SIP dispatch rule...", flush=True)
    run_lk("sip", "dispatch", "create", str(DISPATCH_FILE))

    print("Current SIP inbound trunks:", flush=True)
    run_lk("sip", "inbound", "list")

    print("Current SIP dispatch rules:", flush=True)
    run_lk("sip", "dispatch", "list")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
