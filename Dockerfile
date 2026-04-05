# Gradience: Evidence-based LoRA compression
# Build: docker build -t gradience .
# Run:   docker run --gpus all -it gradience gradience-bench --help

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

LABEL maintainer="johntnanney@gmail.com"
LABEL description="Gradience: Spectral audit and evidence-based compression for LoRA adapters"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set HuggingFace cache defaults
ENV HF_HOME=/workspace/hf_cache
ENV HF_HUB_CACHE=/workspace/hf_cache/hub
ENV HF_DATASETS_CACHE=/workspace/hf_cache/datasets

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install package
COPY pyproject.toml MANIFEST.in README.md LICENSE* ./
COPY gradience/ gradience/

# Install gradience with bench extras (PyTorch already in base image)
RUN pip install --no-cache-dir ".[bench]"

# Create non-root user for runtime security
RUN useradd -m -s /bin/bash gradience

# Create workspace directory for cache and outputs, owned by non-root user
RUN mkdir -p /workspace/hf_cache /workspace/output \
    && chown -R gradience:gradience /workspace

# Verify installation (as root, before switching user)
RUN gradience --help && gradience-bench --help

# Switch to non-root user
USER gradience

# Default working directory for outputs
WORKDIR /workspace

ENTRYPOINT ["gradience-bench"]
CMD ["--help"]
