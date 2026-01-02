FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    # required by torch ROCm wheels on many distros
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"
ENV UV_CACHE_DIR=/app/.uv-cache

WORKDIR /app

# Install ROCm PyTorch wheels explicitly (no CUDA).
# We intentionally do this at image build time so uv/pip resolution doesn't
# accidentally pull CUDA wheels.
#
# If you want to change ROCm version, update the index-url.
RUN python -m pip install --no-cache-dir -U pip \
    && python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/rocm6.1 \
      torch torchvision

EXPOSE 5000 8000

CMD ["tail", "-f", "/dev/null"]
