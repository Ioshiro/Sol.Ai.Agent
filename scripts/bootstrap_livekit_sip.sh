#!/bin/sh

export DEBIAN_FRONTEND=noninteractive

apt-get update >/dev/null
apt-get install -y --no-install-recommends curl ca-certificates jq tar >/dev/null

lk_version="v2.16.0"
lk_tarball="/tmp/lk_${lk_version}.tar.gz"

curl -fsSL "https://github.com/livekit/livekit-cli/releases/download/${lk_version}/lk_2.16.0_linux_amd64.tar.gz" -o "$lk_tarball"
tar -xzf "$lk_tarball" -C /usr/local/bin lk
chmod +x /usr/local/bin/lk
rm -f "$lk_tarball"

echo "Waiting for LiveKit API..."
until curl -fsS http://livekit:7880 >/dev/null; do
  sleep 2
done

echo "Creating SIP inbound trunk..."
lk sip inbound create /work/sip/trunk.json || echo "Inbound trunk create returned non-zero, continuing (it may already exist)."

echo "Creating SIP dispatch rule..."
lk sip dispatch create /work/sip/dispatch-rule.json || echo "Dispatch rule create returned non-zero, continuing (it may already exist)."

echo "Current SIP inbound trunks:"
lk sip inbound list || true

echo "Current SIP dispatch rules:"
lk sip dispatch list || true
