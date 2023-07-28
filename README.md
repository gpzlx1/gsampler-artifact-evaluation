# 0.Introduction

This is the repo for artificial evaluation of gSampler.

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
├── run.sh
└── table8 #reproduce figure8
    ├── graphsage
    ├── ladies
    └── testbench.sh
```

 # 3. Setup

## 3.1 Build gSampler

Please refer to [this guide](https://github.com/gsampler9/gSampler.git) to build gSampler. 

## 3.2 Other Requirements

It is required to install some libraries to run evaluation code:

```
pip install -r requirements.txt
```

# 4. Execution

## 4.1 Figure 7

For Figure 7, Please refer to this doc to execute the code.

## 4.2 Figure 8

To run the experiments: 

```shell
# in project root
cd figure8/
bash testbench.sh
```

Then four result.csv will be generated at `ladies/outputs/` `asgcn/outputs/`, `pass/outputs/`, `shadowkhop/outputs/`respectively.

To generate the figures of the four sampling algorithm:

```
python plot.py
```

Then the .pdf format figure will be generated in `[algorithm]/outputs/` dir.

## 4.3 Figure 10

Similar as figure8, to run the experiments: 

```shell
# in project root
cd figure10/
bash testbench.sh
```

Then two result.csv will be generated at `ladies/outputs/` and `graphsage/outputs/`respectively.

To generate the figures of the two ablation study :

```
python plot.py
```

Then the .pdf format figure will be generated in `[algorithm]/outputs/` dir.

## 4.4 Table8

To run the experiments: 

```shell
# in project root
cd table8/
bash testbench.sh
```

Then two result.csv will be generated at `ladies/outputs/` and `graphsage/outputs` respectively.
