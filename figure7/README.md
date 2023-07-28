# Introduction

This guide aims to help users reproduce Figure 7 from the paper. To achieve this, three systems need to be built first, followed by using `testbench.sh` to generate Figure 7.

# Build GunRock, SkyWalker, and PyG

In the `figure7` directory, execute the following command to clone the baseline code for building:

```shell
git submodule update --init --recursive
```

## Build GunRock [optional]

```shell
git clone -b gsampler-baseline https://github.com/DanielMao1/gunrock-gsampler-baseline.git
cd gunrock/
git checkout gsampler-baseline
mkdir build
cd build 
cmake .. 
make sage
```

## Build SkyWalker [optional]

Return to the `figure7` directory and execute the following command to build SkyWalker:

```shell
git clone -b gsampler-baseline https://github.com/DanielMao1/skywalker-gsampler-baseline.git
cd skywalker/
git checkout gsampler-baseline
mkdir build
cd build 
cmake .. 
make -j
```

## Build PyG [optional]

Assuming that PyG is already installed in `gsampler-ae`, if not, install it first:

```shell
pip install torch_geometric
```

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

There are a total of 6 systems to be run. The experiments can be run all together at once or separately.

## Run all baselines at once

```shell
bash testbench.sh
```

## Run Separately

### Run gSampler 

```shell
cd gSampler
bash run_deepwalk_gsampler.sh
bash run_node2vec_gsampler.sh
bash run_graphsage_gsampler.sh
```



### Run GunRock

```shell
# In figure7/
bash run_gunrock.sh
```

### Run SkyWalker

```shell
# In figure7/
bash run_skywalker.sh
```

### Run DGL

```shell
cd dgl
bash run_deepwalk_dgl.sh
bash run_node2vec_dgl.sh
bash run_graphsage_dgl.sh
```

### Run PyG

```shell
cd PyG
bash run_deepwalk_pyg.sh
bash run_graphsage_pyg.sh
```

### Run cuGraph

```shell
cd cuGraph
bash run_cugraph.sh
bash uva_cugraph.sh # because loading large graph to cugraph is very slow(12h+), this is an optional operation
```
# Ploting
The result is in `figure7/outputs/result.csv`

In `figure7/`, run 

```shell
python plot.py
```

Then the figures will be generated at `outputs/`.