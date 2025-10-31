#!/bin/bash

# System Setup Script for Piper1-GPL on Linux Mint
# This script installs dependencies and sets up the development environment

set -e  # Exit on any error

echo "========================================="
echo "Starting System Setup"
echo "========================================="

# Get the directory where this setup script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Add deadsnakes PPA for Python 3.13
echo "Adding deadsnakes PPA..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install all required packages in one command
echo "Installing all required packages..."
sudo apt install -y \
    ffmpeg \
    build-essential \
    git \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3.13-tk \
    cmake \
    ninja-build

echo "========================================="
echo "System packages installed successfully!"
echo "========================================="

# Clone the piper1-gpl repository
echo "Cloning piper1-gpl repository to ~/piper1-gpl..."
cd ~
if [ -d "piper1-gpl" ]; then
    echo "Warning: ~/piper1-gpl already exists. Skipping clone."
    echo "If you want a fresh install, remove the directory first with: rm -rf ~/piper1-gpl"
else
    git clone https://github.com/OHF-voice/piper1-gpl.git
fi

echo "========================================="
echo "Repository cloned successfully!"
echo "========================================="

# Copy from arbitrary package directory ---
echo "Copying utilities from package directory to ~/piper1-gpl..."

# Copy all contents (excluding .git and this script) into ~/piper1-gpl
rsync -av --exclude='.git' --exclude='setup.sh' "$SCRIPT_DIR"/ ~/piper1-gpl/

echo "Files successfully copied to ~/piper1-gpl."
# ----------------------------------------------------------

# Change to the piper1-gpl directory
cd ~/piper1-gpl

# Create all required directories
echo "Creating project directories..."
mkdir -p ~/piper1-gpl/dataset
mkdir -p ~/piper1-gpl/my-training
mkdir -p ~/piper1-gpl/my-model
mkdir -p ~/piper1-gpl/cache
mkdir -p ~/piper1-gpl/audio_samples
mkdir -p ~/piper1-gpl/lightning_logs/version_0/checkpoints

echo "Directories created:"
echo "  - dataset (for metadata and training wavs)"
echo "  - my-training (for config.json)"
echo "  - my-model (for model output)"
echo "  - cache (cache directory)"
echo "  - audio_samples (callback wav files during testing)"
echo "  - lightning_logs/version_0/checkpoints (initial checkpoint file)"

# Create Python virtual environment
echo "Creating Python 3.13 virtual environment..."
python3.13 -m venv src/python/.venv

# Activate virtual environment
echo "Activating virtual environment..."
source ~/piper1-gpl/src/python/.venv/bin/activate

# Install development dependencies
echo "Installing development dependencies..."
python -m pip install -e .[dev]

# Install training dependencies
echo "Installing training dependencies..."
python -m pip install -e .[train]

# Build cython extensions
echo "Building monotonic align extensions..."
./build_monotonic_align.sh

# Build extensions in place
echo "Building extensions in place..."
python setup.py build_ext --inplace

# Build the package
echo "Building the package..."
python -m build

# Install specific torch versions
echo "Installing PyTorch 2.8.0 and torchaudio 2.8.0..."
python -m pip install torch==2.8.0 torchaudio==2.8.0

# Install sounddevice used by recording.py
echo "Installing sounddevice"
python -m pip install sounddevice

# Download checkpoint file from Hugging Face
echo "Downloading checkpoint file from Hugging Face..."
cd ~/piper1-gpl/lightning_logs/version_0/checkpoints
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/epoch%3D4641-step%3D3104302.ckpt

# Download config file from Hugging Face
echo "Downloading config.json from Hugging Face..."
cd ~/piper1-gpl/my-training
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/config.json


echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "Virtual environment location: ~/piper1-gpl/src/python/.venv"
echo ""
echo "To activate the environment in the future, run:"
echo "  source ~/piper1-gpl/src/python/.venv/bin/activate"
echo "========================================="
