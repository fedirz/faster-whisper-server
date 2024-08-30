#!/bin/bash

set -e

error_exit() {
    echo "Error: $1" >&2
    exit 1
}

install_dependencies() {
    sudo apt-get update || error_exit "Failed to update package list"
    sudo apt-get install -y curl gpg || error_exit "Failed to install dependencies"
}

install_source() {
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || error_exit "Failed to download and install GPG key"

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list || error_exit "Failed to add NVIDIA repository"

    sudo apt-get update || error_exit "Failed to update package list after adding NVIDIA repository"
}

install_nctk() {
    sudo apt-get install -y nvidia-container-toolkit || error_exit "Failed to install NVIDIA Container Toolkit"
}

configure_docker() {
    sudo nvidia-ctk runtime configure --runtime=docker || error_exit "Failed to configure Docker runtime"
    sudo systemctl restart docker || error_exit "Failed to restart Docker service"
}

verify_installation() {
    echo "Verifying installation..."
    sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi || error_exit "Failed to run NVIDIA SMI in Docker container"
}

generate_cdi_spec() {
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml || error_exit "Failed to generate CDI specification"
    nvidia-ctk cdi list || error_exit "Failed to list CDI devices"
}

main() {
    install_dependencies
    install_source
    install_nctk
    configure_docker
    verify_installation
    generate_cdi_spec

    echo "NVIDIA Container Toolkit installation and configuration completed successfully."
}

main



