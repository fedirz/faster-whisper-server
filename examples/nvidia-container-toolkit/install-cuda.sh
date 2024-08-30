#!/bin/bash

error_out_if_false() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

verify_cuda_capable_gpu() {
    lspci | grep -i nvidia
    error_out_if_false "No CUDA-Capable GPU Found"
}

# Uninstall NVIDIA
uninstall_nvidia() {
    # Remove CUDA Toolkit
    sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
        "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    sudo rm -rf /usr/local/cuda*

    # Uninstall NVIDIA Drivers
    sudo apt-get remove --purge "*nvidia-driver*" "libxnvctrl*"

    # Clean Up
    sudo apt-get autoremove --purge -V
}

fetch_linux_info() {
    echo "Host OS information:"
    uname -m && cat /etc/*release
    uname -r
    echo "Please verify compatibility at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#system-requirements"
}

verify_gcc() {
    gcc --version
}

clear_xorg_conf() {
    if [ -s /etc/X11/xorg.conf ]; then
        echo "Warning: /etc/X11/xorg.conf is not empty. Please clear it out or exit."
        read -p "Clear xorg.conf? (y/n): " clear_xorg
        if [ "$clear_xorg" = "y" ]; then
            sudo mv /etc/X11/xorg.conf /etc/X11/xorg.conf.bak
            echo "Moved xorg.conf to xorg.conf.bak"
        else
            echo "Exiting as requested."
            exit 1
        fi
    fi
}

ask_user_for_host_info() {
    read -p "Enter Linux distribution (e.g., ubuntu2204, debian12, ...): " linux_distro
    read -p "Enter host architecture (e.g., x86_64, sbsa, arm64, ...): " host_arch
    echo "$host_arch $linux_distro"
}

install_cuda_keyring() {
    local distro="$1"
    local arch="$2"
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb
    error_out_if_false "Failed to download CUDA keyring"
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    error_out_if_false "Failed to install CUDA keyring"
}

ask_user_for_cuda_toolkit_version() {
    read -p "Enter CUDA Toolkit version (e.g., 12-6): " cuda_version
    echo "$cuda_version"
}

install_cuda_on_ubuntu() {
    cuda_version="$1"
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-$cuda_version
    error_out_if_false "Failed to install CUDA Toolkit"
}

ask_user_for_nvidia_driver_version() {
    read -p "Enter NVIDIA driver version (e.g., 545): " driver_version
    echo "$driver_version"
}

ask_user_for_nvidia_legacy_private_kernels() {
    read -p "Do you need legacy private kernels for older GPUs? (y/n): " legacy_kernels
    echo "$legacy_kernels"
}

install_open_kernels_on_ubuntu() {
    driver_version="$1"
    sudo apt-get install -y nvidia-open-$driver_version
    error_out_if_false "Failed to install open NVIDIA drivers"
}

install_private_kernels_on_ubuntu() {
    driver_version="$1"
    sudo apt-get install -y cuda-drivers-$driver_version
    error_out_if_false "Failed to install private NVIDIA drivers"
}

cuda_environment_setup() {
    local cuda_version="$1"
    echo "export PATH=/usr/local/cuda-${cuda_version//-/.}/bin\${PATH:+:\${PATH}}" >>~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-${cuda_version//-/.}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >>~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-${cuda_version//-/.}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >>~/.bashrc
    source ~/.bashrc
}

cuda_post_install() {
    sudo /usr/bin/nvidia-persistenced --verbose

    echo "Cloning CUDA samples..."
    git clone https://github.com/NVIDIA/cuda-samples.git
    cd cuda-samples/Samples/1_Utilities/deviceQuery
    make
    ./deviceQuery
    error_out_if_false "deviceQuery failed. Installation may be incomplete."

    cd ../../bandwidthTest
    make
    ./bandwidthTest
    error_out_if_false "bandwidthTest failed. Installation may be incomplete."

    echo "NVIDIA driver version:"
    cat /proc/driver/nvidia/version
}

nvidia_driver_install() {
    driver_version=$(ask_user_for_nvidia_driver_version)
    legacy_kernels=$(ask_user_for_nvidia_legacy_private_kernels)

    if [ "$legacy_kernels" = "y" ]; then
        install_private_kernels_on_ubuntu "$driver_version"
    else
        install_open_kernels_on_ubuntu "$driver_version"
    fi
}

cuda_install() {
    host_arch="$1"
    linux_distro="$2"
    linux_version="$3"
    
    install_cuda_keyring "$host_arch" "$linux_distro" "$linux_version"

    cuda_version=$(ask_user_for_cuda_toolkit_version)

    if [ "$linux_distro" = "ubuntu" ]; then
        install_cuda_on_ubuntu "$cuda_version"
    else
        echo "Unsupported Linux distribution: $linux_distro"
        exit 1
    fi

    nvidia_driver_install

    cuda_environment_setup "$cuda_version"
}

cuda_pre_install() {
    # Step by Step
    # Check for CUDA-Capable GPU
    verify_cuda_capable_gpu

    # Verify You Have a Supported Version of Linux and Correct Kernel Headers
    fetch_linux_info

    # Verify the System Has gcc Installed
    verify_gcc

    # Handle Conflicting Installation Methods
    uninstall_nvidia

    # Address Custom xorg.conf, If Applicable
    clear_xorg_conf
}

cleanup() {
    rm -f cuda-keyring_1.1-1_all.deb
}

# Main execution
cuda_pre_install

read host_arch linux_distro <<<$(ask_user_for_host_info)

cuda_install "$host_arch" "$linux_distro"

cuda_post_install

echo "Installation complete. Please reboot your system to finalize the installation."

# Add this line at the end of the script
trap cleanup EXIT
