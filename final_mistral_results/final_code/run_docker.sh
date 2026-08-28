#!/usr/bin/env bash
# filepath: /Users/lbartolome/safe-spoon/test_google/run_docker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/searxng"
SECRET_KEY="$(openssl rand -hex 32)"

mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_DIR/settings.yml" <<YAML
use_default_settings: true

server:
  secret_key: "$SECRET_KEY"
  limiter: false

search:
  formats:
    - html
    - json
YAML

if docker container inspect searxng >/dev/null 2>&1; then
    docker rm -f searxng
fi

docker run -d \
    --name searxng \
    -p 8100:8080 \
    -v "$CONFIG_DIR:/etc/searxng" \
    searxng/searxng:latest

echo "SearXNG iniciado en http://localhost:8100"