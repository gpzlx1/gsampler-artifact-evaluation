# Introduction

This guide is designed to help users replicate Figure 8 from the referenced paper.

# Running Experiments

To conduct the experiments, follow these steps to execute the four complex algorithms:

```shell
export epochs=6
bash testbench.sh
```

After executing these commands, you'll discover the resulting figures in PDF format within the `outputs/` directory. An illustrative sample can be found at `../fig_examples/figure8`.

# Running Partial Experiments

If you only wish to perform specific experiments, navigate to the respective subdirectory. For instance, for LADIES:

```shell
cd figure8/ladies
export epochs=6
bash testbench.sh
```

## Generating Plots

For generating plots within the `figure8/` directory, use the following steps:

```shell
cd figure8
mkdir outputs
python plot.py
```

This will generate the figures in the `outputs/` directory.