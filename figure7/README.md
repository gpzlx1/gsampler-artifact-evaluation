# Introduction

The guide is to help users to reproduce figure7 in the paper.  We need to build three systems firstly and then use `testbench.sh` to generate figure7.

# Build GunRock, SkyWalker and PyG

In `figure7` directory, run command below to clone baseline code for building：

```shell
git submodule update --init --recursive
```

## Build GunRock

```shell
cd gunrock/
git checkout gsampler-baseline
mkdir build
cd build 
cmake .. 
make sage
```

## Build Skywalker

Back to figure7, and execute the following command to build Skywalker

```shell
cd skywalker/
git checkout gsampler-baseline
mkdir build
cd build 
cmake .. 
make -j
```

## Build PyG

It is assumed that the pig the already installed in `gsampler-ae`, if not,  install it first:

```shell
pip install torch_geometric
```

In `/home/ubuntu` (or other directory you prefer)

```shell
git clone https://github.com/pyg-team/pyg-lib.git --recursive
cd pyg-lib
mkdir build
cd build
export Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
export CUDA_ARCH_LIST=75
cmake .. -GNinja -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_LIST}
cmake --build .
cd ..
```

Then execute with the `gsampler-ae` conda environment to install `pyg-lib`

```shell
pip install -e .
```

# Run Experiments

There are 6 system in total to be run. The experiments can be either run all together at once or seperately.

## Run all baseline at once

```shell
bash testbench.sh
```

## Run Seperately

### Run cuGraph

```shell
cd cuGraph
bash run_cugraph.sh
```

### Run gunrock

```shell
#in figure7/
bash run_gunrock.sh
```

### Run SkyWalker

```shell
#in figure7/
bash run_skywalker.sh
```

### Run DGL/PyG/gSampler

```shell
# in figure7/ run deepwalk
cd deepwalk
bash run_dgl.sh
bash run_gsampler.sh
bash run_pyg.sh

# in figure7/ run node2vec
cd node2vec
bash run_dgl.sh
bash run_gsampler.sh

# in figure7/ run graphsage
cd graphsage
bash run_dgl.sh
bash run_gsampler.sh
bash run_pyg.sh
```

