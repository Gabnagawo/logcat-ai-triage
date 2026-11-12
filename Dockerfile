# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY samples ./samples
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 triage && mkdir -p /app/out && chown triage /app/out
USER triage

ENTRYPOINT ["logcat-triage"]
CMD ["triage", "samples", "--out-dir", "out"]
