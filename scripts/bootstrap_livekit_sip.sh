#!/bin/sh

export DEBIAN_FRONTEND=noninteractive

apt-get update >/dev/null
apt-get install -y curl ca-certificates bash jq tar >/dev/null

curl -fsSL https://get.livekit.io/cli | bash >/dev/null
export PATH="/root/.livekit/bin:$PATH"

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
