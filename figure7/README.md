# Introduction

The guide is to help users to reproduce figure7 in the paper.  We need to build three systems firstly and then use `testbench.sh` to generate figure7.

# Build GunRock, SkyWalker and PyG

In `figure7` directory, run command below to clone baseline code for building：

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

## Build Skywalker [optional]

Back to figure7, and execute the following command to build Skywalker

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

It is assumed that the PyG the already installed in `gsampler-ae`, if not,  install it first:

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

There are 6 system in total to be run. The experiments can be either run all together at once or seperately.

## Run all baseline at once

```shell
bash testbench.sh
```

## Run Seperately

### Run gSampler 

```shell
cd gSampler
bash run_deepwalk_gsampler.sh
bash node2vec_gsampler.py
bash run_graphsage_gsampler.sh
```



### Run GunRock

```shell
#in figure7/
bash run_gunrock.sh
```

### Run SkyWalker

```shell
#in figure7/
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
bash run_deepwalk_pyg.py
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
