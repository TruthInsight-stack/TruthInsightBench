FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5

ARG CLAUDE_CODE_VERSION=2.1.220
ARG CODEX_CLI_VERSION=0.149.0
ARG OPENSCIENCE_VERSION=2.0.1
ARG DEEPSEEK_HARNESS_VERSION=0.1.0rc7

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN npm install --global \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "@openai/codex@${CODEX_CLI_VERSION}" \
        "@synsci/openscience@${OPENSCIENCE_VERSION}"

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        "deepseek-harness-sdk==${DEEPSEEK_HARNESS_VERSION}" \
        "matplotlib==3.11.1" \
        "networkx==3.6.1" \
        "numpy==2.4.6" \
        "openpyxl==3.1.5" \
        "pandas==3.0.5" \
        "pillow==12.3.0" \
        "scikit-learn==1.9.0" \
        "scipy==1.17.1" \
        "seaborn==0.13.2" \
        "statsmodels==0.15.0" \
        "torch==2.12.0+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
USER node
