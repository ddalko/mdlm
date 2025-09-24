# MDLM Docker Environment
# Base image with PyTorch 2.2+ and CUDA 12.1 support for H100 GPU
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    cmake \
    ninja-build \
    vim \
    nano \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Initialize conda (already available in pytorch image)
RUN conda init bash

# Set working directory
WORKDIR /workspace/mdlm

# Copy requirements file and install Python dependencies
COPY requirements.yaml .

# Install additional packages from requirements.yaml
RUN conda install -y jupyter=1.0.0 && \
    pip install \
        causal-conv1d==1.1.3.post1 \
        datasets==2.18.0 \
        einops==0.7.0 \
        fsspec==2024.2.0 \
        git-lfs==1.6 \
        h5py==3.10.0 \
        hydra-core==1.3.2 \
        ipdb==0.13.13 \
        lightning==2.2.1 \
        mamba-ssm==1.1.4 \
        notebook==7.1.1 \
        nvitop==1.3.2 \
        omegaconf==2.3.0 \
        packaging==23.2 \
        pandas==2.2.1 \
        rich==13.7.1 \
        seaborn==0.13.2 \
        scikit-learn==1.4.0 \
        timm==0.9.16 \
        transformers==4.38.2 \
        triton==2.2.0 \
        wandb==0.13.5 \
        flash-attn==2.5.6

# Verify CUDA installation and PyTorch CUDA support
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

# Copy the rest of the application
# COPY . .

# Set default command to start bash
CMD ["/bin/bash"]

# Alternative: If you want to run a specific script by default
# CMD ["python", "main.py"]