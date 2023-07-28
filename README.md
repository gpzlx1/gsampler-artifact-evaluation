# 0.Introduction

This is the repo for artifact evaluation of gSampler.
* reproduce ...


... If ...

# 1.Data

The data used are stored in the directory `./dataset`

```
dataset
├── friendster
├── livejournal
├── ogbn_papers100M
└── ogbn_products
```

for ..., 

for ..., ..

# 2.Directory Structure

The repo cotains four directory as follows:

```shell
gsampler-artifact-evaluation
├── README.md
├── figure10  #reproduce figure10
├── figure7 #reproduce figure7
├── figure8 #reproduce figure8
└──  run.sh #reproduce figure8,10 table 8 together
```

# 3. Setup

If you are using AWS EC2 server we provided, then the environment is already build, just  run `conda activate gsampler-ae` and jump to step 4, otherwise, your should build the environment with the following instruction.



## 3.1 Build gSampler

Please refer to [this guide](https://github.com/gsampler9/gSampler.git) to build gSampler.  In the following instruction, we assume that the user is in a conda environment named `gsampler-ae`, with dgl and gs library installed.

### 3.2 Install PyG [optional]

### 3.3 Install DGL [optional]

### 3.3 Install SkyWalker [optional]

### 3.4 Install Gunrock [optional]

### 3.5 Install CuGraph [optional]



# 4. Execution

For convience, the Figure 8, Figure10, Table10 can be executed together. In the project root directory, just simply run 

```
# in the root directory
bash run.sh
```

This will generate the results in the subdirectories. Because Figure7 need to build another three systems, so it will be generated seperately.

## 4.1 Build and Generate Figure 7

For Figure 7, Please refer to [subdirectory README.md](https://github.com/gpzlx1/gsampler-artifact-evaluation/blob/main/figure7/README.md) to build and run the multiple baseline code.

## 4.2 Figure 8 

To generate figure8 independently: 

```shell
# in project root
cd figure8/
export epochs=6
bash testbench.sh
```

Then four result.csv will be generated at `ladies/outputs/` `asgcn/outputs/`, `pass/outputs/`, `shadowkhop/outputs/`respectively.

To generate the figures of the four sampling algorithm:

```
python plot.py
```

Then the .pdf format figure will be generated in `[algorithm]/outputs/` dir.

## 4.3 Figure 10

Similar to figure8, To generate figure10 independently: 

```shell
# in project root
cd figure10/
export epochs=6
bash testbench.sh
```

Then two result.csv will be generated at `ladies/outputs/` and `graphsage/outputs/`respectively.

To generate the figures of the two ablation study :

```
python plot.py
```

Then the .pdf format figure will be generated in `[algorithm]/outputs/` dir.