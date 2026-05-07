# Base image with CUDA 12 support
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    openssh-server \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
    tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create shared venv
RUN uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone mujoco_playground at pinned commit
WORKDIR /app/mujoco_playground
RUN git clone https://github.com/google-deepmind/mujoco_playground.git . && \
    git checkout d43c7216bc892d59237335e83fa60c6da77a2698

# Install CUDA JAX
RUN uv pip install --python /opt/venv/bin/python -U "jax[cuda12]" --index-url https://pypi.org/simple

# Install mujoco_playground with all extras
RUN uv pip install --python /opt/venv/bin/python -e ".[all]"

# Verify installation
RUN /opt/venv/bin/python -c "import mujoco_playground; print('mujoco_playground installed successfully')"

# SSH setup (key injected at RunPod start)
RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# Setup script
COPY setup.sh /opt/setup.sh
RUN chmod +x /opt/setup.sh

# Workspace
RUN mkdir -p /workspace

# Greeting
RUN echo 'echo ""' >> /root/.bashrc && \
    echo 'echo "👋 Welcome! Run /opt/setup.sh to log in to GitHub and wandb."' >> /root/.bashrc && \
    echo 'echo ""' >> /root/.bashrc

WORKDIR /workspace

EXPOSE 22

CMD service ssh start && bash


# RUNPOD START COMMAND, will replace CMD:
# bash -c "echo <YOUR_PUBLIC_KEY> >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && service ssh start && sleep infinity"

# My personal dev tools are missing:

# uv pip install omegaconf
# uv pip install hydra-core 	
# uv pip install -e .
# uv pip install mujoco-warp

# For egl this were missing:

# apt-get update && apt-get install -y \
#     libegl1 \
#     libgles2 \
#     libgl1 \
#     libglfw3 \
#     libosmesa6




