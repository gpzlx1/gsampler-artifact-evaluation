# Guide Introduction

This guide is designed to assist users in reproducing Figure 10 as presented in the referenced paper.

# Running Experiments

There are a total of 2 breakdown experiments that need to be conducted. Execute the following commands:

```shell
export epochs=6
bash testbench.sh
```

Upon completion, the results will yield 2 figures in PDF format located within the `outputs/` directory. For a visual reference, consult `../fig_examples/figure10`.

# Partial Experiment Execution

If you intend to run specific segments of the experiments, navigate to the respective subdirectory. For instance, for LADIES:

```shell
cd figure10/ladies
export epochs=6
bash testbench.sh
```

## Generating Plots

Inside the `figure10/` directory, proceed with:

```shell
cd figure10
mkdir outputs
python plot.py
```

Subsequently, the resulting figure will be available within the `outputs/` directory.