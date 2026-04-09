FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system --gid 999 appuser \
 && useradd --system --uid 999 --gid 999 --create-home appuser

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    uv sync --locked --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

ENTRYPOINT []

CMD ["kokoro-server"]
