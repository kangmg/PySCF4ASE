#!/bin/bash

# Current directory
PWD=$(pwd)

echo -e "\033[1;34mPySCF4ASE Installation proceeding...\033[0m"
echo ""

# Create a temporary directory
TMP_DIR="$PWD/tmp"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR" || { echo "Failed to change directory to $TMP_DIR"; exit 1; }

# Clone the GitHub repository
echo "Cloning PySCF4ASE Repository..."
git clone -q https://github.com/kangmg/PySCF4ASE "$TMP_DIR/PySCF4ASE"
echo "Done!"
echo ""
cd PySCF4ASE

# Check for CUDA version
if ! command -v nvcc &> /dev/null; then
    echo "nvcc command not found. Please install CUDA toolkit."
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | cut -d'.' -f1)
echo -e "\033[1;32mYour CUDA Version:\033[0m $NVCC_VERSION"
echo ""


# Add packages to requirements.txt based on CUDA version
echo "" >> requirements.txt
if [ "$NVCC_VERSION" -eq 11 ]; then
    echo "gpu4pyscf-cuda11x" >> requirements.txt
    echo "cutensor-cu11" >> requirements.txt
elif [ "$NVCC_VERSION" -eq 12 ]; then
    echo "gpu4pyscf-cuda12x" >> requirements.txt
    echo "cutensor-cu12" >> requirements.txt
else
    echo -e "\033[1;31mWARNING:\033[0m gpu4pyscf not supported for your CUDA Version"
fi

# Install the packages
echo "Installing dependent packages..."
echo ""
pip install -q -r requirements.txt || { echo "Failed to install requirements"; exit 1; }
pip install -q . || { echo "Failed to install PySCF4ASE"; exit 1; }
echo "Done!"

# Remove the temporary directory
cd "$PWD" || { echo "Failed to change directory to $PWD"; exit 1; }
rm -rf "$TMP_DIR" || { echo "Failed to remove temporary directory $TMP_DIR"; exit 1; }

echo ""
echo -e "\033[1;34mInstallation completed successfully.\033[0m"
