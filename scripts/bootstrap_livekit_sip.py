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



def _delete_existing(*, resource: str, label: str) -> None:
    """Delete existing SIP resources of the given type before recreating."""
    try:
        result = subprocess.run(
            [str(LK_BIN), "sip", resource, "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            print(f"Could not list {label}s (exit {result.returncode}), skipping delete.", flush=True)
            return

        stripped = result.stdout.strip()
        ids: list[str] = []

        # Try JSON first
        if stripped.startswith("[") or stripped.startswith("{"):
            import json
            try:
                data = json.loads(stripped)
                if isinstance(data, list):
                    ids = [
                        str(entry[key])
                        for entry in data
                        for key in ("id", "sipInboundId", "sipDispatchRuleId")
                        if key in entry and not isinstance(entry.get(key), (list, dict))
                    ]
                elif isinstance(data, dict):
                    for key in ("id", "sipInboundId", "sipDispatchRuleId"):
                        if key in data:
                            ids.append(str(data[key]))
                            break
            except json.JSONDecodeError:
                pass

        # Fallback: parse table output (│ col1 │ col2 │ … │) or scan for known prefix IDs
        if not ids:
            for line in stripped.splitlines():
                line = line.strip()
                if not line.startswith("│") or "─" in line:
                    continue
                parts = [col.strip() for col in line.split("│") if col.strip()]
                if parts:
                    first_cell = parts[0]
                    if first_cell and not any(
                        keyword.lower() in first_cell.lower()
                        for keyword in ("id", "trunk", "dispatch", "name", "number", "sip")
                    ):
                        ids.append(first_cell)
            if not ids:
                # Last resort: scan for LiveKit ID patterns (ST_..., SDR_..., UUIDs)
                import re
                ids = re.findall(
                    r"[A-Z]{2,5}_[A-Za-z0-9]{8,}"
                    r"|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                    stripped,
                )

        for resource_id in ids:
            print(f"Deleting existing {label} {resource_id}...", flush=True)
            run_lk("sip", resource, "delete", resource_id)
    except Exception as exc:
        print(f"Delete-existing {label}s: {exc} — continuing anyway.", flush=True)

def main() -> int:
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"

    install_lk()
    wait_for_livekit()

    print("Ensuring SIP inbound trunk...", flush=True)
    _delete_existing(resource="inbound", label="trunk")
    run_lk("sip", "inbound", "create", str(TRUNK_FILE))

    print("Ensuring SIP dispatch rule...", flush=True)
    _delete_existing(resource="dispatch", label="dispatch rule")
    run_lk("sip", "dispatch", "create", str(DISPATCH_FILE))
    print("Current SIP inbound trunks:", flush=True)
    run_lk("sip", "inbound", "list")

    print("Current SIP dispatch rules:", flush=True)
    run_lk("sip", "dispatch", "list")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
