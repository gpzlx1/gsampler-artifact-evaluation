import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


def plt_init(figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )


def plt_save_and_final(save_path):
    print(f"[Note]Save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plt_bar(
    elements,
    labels,
    xlabels,
    ylabel=None,
    yticks=None,
    na_str="N/A",
    save_path=None,
    lengend=True,
    title=None,
):
    plt_init(figsize=(12, 3))
    # fix parameter
    font_size = 14
    hatch_list = [None, None, ".." "--", "//", "x", "||"]
    color_list = ["k", "w", "w", "w", "w", "w", "w"]
    ax = plt.gca()
    num_series = len(labels)
    num_elements_per_series = len(xlabels)
    value_limit = yticks[-1]
    width = 1.2
    for i in range(num_series):
        plot_x = [(num_series + 3) * j + i * width for j in range(num_elements_per_series)]
        # handle N/A
        plot_y = []
        plot_label = []
        for e in elements[i]:
            if isfloat(e) or e.isdigit():
                val_e = float(e)
                if val_e < value_limit:
                    plot_y.append(float(e))
                else:
                    plot_y.append(value_limit)
                if isfloat(e):
                    res = float(e)
                else:
                    res = int(e)
                plot_label.append(round(res, 2))
            else:
                plot_y.append(0.01)
                plot_label.append(na_str)

        container = ax.bar(
            plot_x,
            plot_y,
            width=width,
            edgecolor="k",
            hatch=hatch_list[i],
            color=color_list[i],
            label=labels[i],
            zorder=10,
        )
        # print("ok")
        ax.bar_label(container, plot_label, fontsize=12, zorder=20, fontweight="bold")

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)

    if yticks is not None:
        ax.set_yticks(yticks, yticks, fontsize=font_size)
        ax.set_ylim(0, value_limit)

    plot_xticks = [(num_series + 3) * j + (width / 2) * (num_series - 1) for j in range(num_elements_per_series)]

    ax.set_xticks(plot_xticks, xlabels, fontsize=font_size)
    if lengend:
        ax.legend(
            fontsize=font_size,
            edgecolor="k",
            ncol=6,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.30),
        )
    # ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(title, fontsize=18, fontweight="bold")
    ax.set_ylabel("Time (s)", fontsize=font_size, fontweight="bold")
    plt.grid(axis="y", linestyle="-.", zorder=0)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.show()
    if save_path is not None:
        plt_save_and_final(save_path=save_path)


def isfloat(val):
    return all([[any([i.isnumeric(), i in [".", "e"]]) for i in val], len(val.split(".")) == 2])


def draw_figure(input_path, yticks, lengend):
    opt_seq = ["P", "P+C", "P+C+D", "P+C+D+B"]
    dataset_seq = ["ogbn-products", "ogbn-papers100M"]
    title = input_path.split("/")[0]
    save_path = f"outputs/{title}.pdf"
    elements = []
    labels = []
    data = defaultdict(list)
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            system = row[0]
            dataset = row[1]
            time = row[2]
            data[dataset].append(time)
    for i, opt in enumerate(opt_seq):
        labels.append(opt)
        time_seq = []
        for name in dataset_seq:
            if name in data:
                time_seq.append(data[name][i])
            else:
                time_seq.append("N/A")
        elements.append(time_seq)
    plt_bar(elements, labels, ["PD", "PP"], save_path=save_path, title=title, lengend=lengend, yticks=yticks)


if __name__ == "__main__":
    draw_figure("graphsage/outputs/result.csv", np.arange(0, 3.6, 0.5), True)
    draw_figure("ladies/outputs/result.csv", np.arange(0, 21, 4), True)
