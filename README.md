# 0.Introduction

This is the repo for artifact evaluation of gSampler.

# 1.Data

The data used are all in the directory `/home/ubuntu/dataset`

```
dataset
├── friendster
├── livejournal
├── ogbn_papers100M
└── ogbn_products
```

# 2.Directory Structure

The repo cotains four directory as follows:

```shell
gsampler-artifact-evaluation
├── README.md
├── figure10  #reproduce figure10
│   ├── graphsage 
│   ├── ladies
│   ├── plot.py
│   └── testbench.sh
├── figure7 #reproduce figure7
│   ├── cugraph
│   ├── deepwalk 
│   ├── graphsage
│   ├── gunrock
│   ├── node2vec
│   ├── plot.py
│   ├── run_all.sh
│   ├── run_gunrock.sh
│   ├── run_skywalker.sh
│   ├── skywalker
│   └── skywalker.sh
├── figure8 #reproduce figure8
│   ├── asgcn
│   ├── ladies
│   ├── pass
│   ├── plot.py
│   ├── shadowkhop
│   └── testbench.sh
├── requirements.txt
├── run.sh #reproduce figure8,10 table 8 together
└── table8 #reproduce table8
    ├── graphsage
    ├── ladies
    └── testbench.sh
```

 # 3. Setup

If you are using AWS EC2 server we provided, then the environment is already build, just  run `conda activate gsampler-ae` and jump to step 4, otherwise, your should build the environment with the following instruction.



## 3.1 Build gSampler

Please refer to [this guide](https://github.com/gsampler9/gSampler.git) to build gSampler.  In the following instruction, we assume that the user is in a conda environment named `gsampler-ae`, with dgl and gs library installed.

## 3.2 Build AE project

Clone and install dependencies for artifact evaluation(AE) project:

```
# In your workspace, '/home/ubuntu' by default
git clone --recursive https://github.com/gpzlx1/gsampler-artifact-evaluation.git
cd gsampler-artifact-evaluation
pip install -r requirements.txt
```

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

## 4.4 Table8

 To generate table 8 independently: 

```shell
# in project root
cd table8/
export epochs=6
bash testbench.sh
```

Then two result.csv will be generated at `ladies/outputs/` and `graphsage/outputs` respectively.