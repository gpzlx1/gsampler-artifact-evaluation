# Introduction

This repository contains artifacts for evaluating [gSampler](https://github.com/gsampler9/gSampler). It includes code to reproduce Figures 7, 8, and 10 from the paper:

- **Figure 7**: Time comparison between gSampler and baseline systems for 3 simple graph sampling algorithms.
- **Figure 8**: Time comparison between gSampler and baseline systems for 4 complex graph sampling algorithms.
- **Figure 10**: Ablation study of gSampler's optimizations on PD and PP graphs.


To replicate the paper's findings, we recommend using the `p3.x16large` instance equipped with an NVIDIA V100 GPU, 64 vCPUs, and 480GB memory. Running all scripts may take approximately 6-8 hours.

Ensure your hardware has at least 256GB memory for preprocessing the two large-scale graphs, `ogbn_papers100M` and `friendster`, in CSC format.

To reproduce the paper's results, use the latest version from the main branch of the repository and the source code available at https://github.com/gsampler9/gSampler.

# Dataset Preparation

The data for this project should be stored in the `./dataset` directory and include the following datasets:

1. friendster
2. livejournal
3. ogbn_papers100M
4. ogbn_products

Download the `ogbn_products` and `ogbn_papers100M` datasets from [ogb](https://ogb.stanford.edu/), and the `livejournal` and `friendster` datasets from [SNAP](https://snap.stanford.edu/data/).

## For Other Datasets

To handle other datasets, follow these steps:

1. Prepare the graph in the CSC format.
2. Load the dataset using the `m.load_graph` API.

```python
# Prepare the graph in CSC format
csc_indptr, csc_indices = load_graph(...)

# Load the graph into GPU memory
m = gs.Matrix()
m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

# For large-scale graphs using Unified Virtual Addressing (UVA)
m.load_graph("CSC", [csc_indptr.pin_memory(), csc_indices.pin_memory()])

# To utilize super-batching, convert Matrix to BatchMatrix
bm = gs.BatchMatrix()
bm.load_from_matrix(m)
```

# Directory Structure

The repository contains four directories as follows:

```shell
gsampler-artifact-evaluation
├── README.md
├── fig_examples # Example output figures.
├── examples  # E2E demo with DGL
├── figure10  # reproduce figure10
├── figure7   # reproduce figure7
├── figure8   # reproduce figure8
├── clean.sh  # delete all results
└── run.sh    # run all reproduce workload
```

# Setup

To ease the burden of setting up a software-hardware environment, we provided reviewers with a direct instance. For login instructions, please refer to comments `A2` and `A3` in sosp23ae.

If using our AWS EC2 server, simply run `conda activate gsampler-ae` and proceed to step 4. For other setups, follow these instructions:

1. Git clone the repo:
```shell
git submodule update --init --recursive
```

## 3.1 Build gSampler

Follow [this guide](https://github.com/gsampler9/gSampler.git) to build gSampler. Ensure you're in the `gsampler-ae` Conda environment with dgl and gs library installed.

## 3.2 Install PyG [optional]

Refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for installation.

## 3.3 Install DGL [optional]

Refer to https://www.dgl.ai/pages/start.html for installation.

## 3.3 Install SkyWalker [optional]

```shell
cd figure7/skywalker/
git checkout gsampler-baseline
mkdir build
cd build 
cmake .. 
make -j
```

## 3.4 Install Gunrock [optional]

```shell
cd figure7/gunrock/
git checkout gsampler-baseline
mkdir build
cd build 
cmake .. 
make sage
```

## 3.5 Install CuGraph [optional]

Refer to https://github.com/rapidsai/cugraph for installation.

# Execution

To execute Figure 8, Figure 10, and Table 10 together, navigate to the project root directory and run the following command:

```
cd ${workspace}
bash clean.sh
bash run.sh
```

Results will be generated in the subdirectories. Note that Figure 7 requires building three additional systems, so it will be generated separately.

## 4.1 Build and Generate Figure 7

To build and run the multiple baseline code for Figure 7, you will need cuGraph, GunRock, SkyWalker, PyG, and DGL. Please install them first. Refer to [figure7](./figure7/README.md) for detailed instructions.

## 4.2 Figure 8

To build and run the multiple baseline code for Figure 8, you will need DGL and PyG. Please install them first. Refer to [figure8](./figure8/README.md) for detailed instructions.

## 4.3 Figure 10

To build and run the multiple baseline code for Figure 10, you will need DGL. Please install them first. Refer to [figure10](./figure10/README.md) for detailed instructions.
