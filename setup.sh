#!/bin/bash

PWD=$(pwd)

echo PySCF4ASE Installation proceed . . .
echo ''

# make tmp directory
mkdir $PWD/tmp
cd $PWD/tmp

# clone github repo
echo Cloning PySCF4ASE Repository
echo ''

git clone https://github.com/kangmg/PySCF4ASE $PWD/tmp

cd PySCF4ASE

# check CUDA VERSION
NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | cut -d'.' -f1)

echo -e "\033[1;34mYour CUDA Version :\033[0m $NVCC_VERSION"

if [ "$NVCC_VERSION" -eq 11 ]; then
  echo "gpu4pyscf-cuda11x" >> requirements.txt
  echo "cutensor-cu11" >> requirements.txt
elif [ "$NVCC_VERSION" -eq 12 ]; then
  echo "gpu4pyscf-cuda12x" >> requirements.txt
  echo "cutensor-cu12" >> requirements.txt
else
  echo -e "\033[1;31mWARNING\033[0m gpu4pyscf not supported your CUDA Version"
fi

pip install -r requirements.txt

pip install .