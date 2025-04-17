import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import ticker

import deeptan.constants as const
from deeptan.stat.metrics import MetricsDictMaker, format_ticks


def kde_grid_plot_data(
    metrics_data: MetricsDictMaker,
    seed: int,
    metrics: List[str],
    tasks: List[str],
    dim: str = "sample_metrics",
):
    _dataset = {}
    for _task in tasks:
        _dataset[_task] = {}
        _dataset[_task][dim] = {}
        for _met in metrics:
            _fname = metrics_data.metrics_dict["summary_recon"].filter((pl.col("task") == _task) & (pl.col("metric") == _met) & (pl.col("seed_num") == seed))["fname"].item()
            _dataset[_task][dim][_met] = metrics_data.metrics_dict["metrics"]["recon"][_fname][dim][_met]
    return _dataset


def kde_grid_plot(
    dataset: Dict,
    x_lab: str,
    y_labs: List[str],
    metrics: List[str],
    metrics_text: List[str],
    x_lab_text: str,
    y_labs_text: List[str],
    dim: str = "sample_metrics",
    fig_name: Optional[str] = None,
    dir4save: Optional[str] = None,
):
    try:
        plt.clf()
    except:
        pass

    # %config InlineBackend.figure_format = 'retina'
    # %config InlineBackend.figure_dpi = 300

    # a4_width_cm = 21
    # cm_to_inches = 0.393701
    # a4_width_inches = a4_width_cm * cm_to_inches
    n_cols = len(y_labs)
    n_rows = len(metrics)
    fig_width = 2.6 * n_cols
    fig_height = 2.4 * n_rows

    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    row_limits = []

    for i, _met in enumerate(metrics):
        row_x_min, row_x_max = -0.25, 1.0
        row_y_min, row_y_max = -0.25, 1.0

        for j, y_lab in enumerate(y_labs):
            ax = axes[i, j]
            x_data = dataset[x_lab][dim][_met]["values"]
            y_data = dataset[y_lab][dim][_met]["values"]

            # 绘制主KDE图
            sns.kdeplot(ax=ax, x=x_data, y=y_data, fill=True)

            # 创建边缘分布的坐标轴
            ax_histx = ax.inset_axes([0, 1.04, 1, 0.25], sharex=ax)
            ax_histy = ax.inset_axes([1.04, 0, 0.25, 1], sharey=ax)

            # 绘制边缘分布
            sns.kdeplot(x=dataset[x_lab][dim][_met]["values"], ax=ax_histx, fill=True, legend=False)
            sns.kdeplot(y=dataset[y_lab][dim][_met]["values"], ax=ax_histy, fill=True, legend=False)

            # 移除边缘分布图的刻度、标签和边框
            ax_histx.set_ylabel(None)
            ax_histy.set_xlabel(None)
            ax_histx.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_histy.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax_histx.spines.values():
                spine.set_visible(False)
            for spine in ax_histy.spines.values():
                spine.set_visible(False)
            # 禁用网格
            ax_histx.grid(False)
            ax_histy.grid(False)

            # 设置标题和标签
            ax.set_title(metrics_text[i], y=0.8)
            ax.set_xlabel(x_lab_text)
            ax.set_ylabel(y_labs_text[j])

            # 更新当前行的 x 和 y 的最小值和最大值
            row_x_min = row_x_min if np.isnan(min(x_data)) or np.isinf(min(x_data)) else min(row_x_min, min(x_data))
            row_x_max = row_x_max if np.isnan(max(x_data)) or np.isinf(max(x_data)) else max(row_x_max, max(x_data))
            row_y_min = row_y_min if np.isnan(min(y_data)) or np.isinf(min(y_data)) else min(row_y_min, min(y_data))
            row_y_max = row_y_max if np.isnan(max(y_data)) or np.isinf(max(y_data)) else max(row_y_max, max(y_data))

            # 设置刻度格式化器
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

            # 缩短刻度棒的长度
            ax.tick_params(axis="both", which="major", length=3)  # 主刻度棒长度
            ax.tick_params(axis="both", which="minor", length=2)  # 次刻度棒长度

        # 保存当前行的最小值和最大值
        row_limits.append(((row_x_min, row_x_max), (row_y_min, row_y_max)))

    # 统一设置每行的 x 和 y 轴范围
    for i, (_met, (x_limits, y_limits)) in enumerate(zip(metrics, row_limits)):
        for j, y_lab in enumerate(y_labs):
            ax = axes[i, j]
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)

    # fig.subplots_adjust(wspace=0.4, hspace=0.3)
    fig.tight_layout()
    # fig.show()

    if fig_name is not None and dir4save is not None:
        fig.savefig(os.path.join(dir4save, f"{fig_name}.png"), dpi=300)
        fig.savefig(os.path.join(dir4save, f"{fig_name}.pdf"))
    return fig


def unpivot_summary(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    r"""
    Unpivot a summary dataframe to long format.
    """
    if seed is not None:
        _df = df.filter(pl.col("seed_num") == seed)
    else:
        _df = df
    # print(_df.columns)
    cols2drop = ["fname", "seed", "seed_num", "split", "path"]
    if "feature_mean" in _df.columns:
        cols2drop.append("feature_mean")
        return _df.drop(cols2drop).rename({"sample_mean": "value"})
    else:
        _df = _df.drop(cols2drop)
    _df = _df.unpivot(index=["task"], on=_df.columns[:-1], variable_name="metric", value_name="value")
    return _df


def metrics_plot_data(metrics_data: MetricsDictMaker, seed: Optional[int] = None):
    _df_plot_label = unpivot_summary(metrics_data.metrics_dict["summary_label"], seed=seed)
    _df_plot_recon = unpivot_summary(metrics_data.metrics_dict["summary_recon"], seed=seed).select(_df_plot_label.columns)
    _df_plot_allmetrics = _df_plot_recon.vstack(_df_plot_label)

    # Pick metrics that are smaller the better and apply 1-value
    metrics_sb = ["mse", "mae", "jsd"]
    _df_plot_allmetrics = _df_plot_allmetrics.with_columns(pl.when(pl.col("metric").is_in(metrics_sb)).then(1 - pl.col("value")).otherwise(pl.col("value")).alias("value"))

    # Map metrics to a more readable name
    _df_plot_allmetrics = _df_plot_allmetrics.with_columns(pl.col("metric").map_elements(lambda x: const.dkey.title_metric_mapping.get(x, x), return_dtype=pl.Utf8).alias("metric"))
    _df_plot_allmetrics = _df_plot_allmetrics.with_columns(pl.col("task").map_elements(lambda x: const.dkey.title_task_mapping.get(x, x), return_dtype=pl.Utf8).alias("task"))
    _df_plot_allmetrics = _df_plot_allmetrics.rename(const.dkey.title_colnameC2_mapping)

    # Prepare data for radar chart
    _tasks = _df_plot_allmetrics.group_by("Task").agg(pl.col("Value").alias("Values"))
    _metrics = _df_plot_allmetrics["Metric"].unique().to_list()
    _task_dict = {_t: _tasks.filter(pl.col("Task") == _t)["Values"].to_list()[0] for _t in _tasks["Task"]}

    # Prepare angles for radar chart
    _n_metrics = len(_metrics)
    angles = np.linspace(0, 2 * np.pi, _n_metrics, endpoint=False).tolist()
    for key in _task_dict:
        _task_dict[key] += _task_dict[key][:1]
    angles += angles[:1]

    return _df_plot_allmetrics, _task_dict, angles


def metrics_plot(
    df4plot: pl.DataFrame,
    fig_name: Optional[str] = None,
    dir4save: Optional[str] = None,
):
    try:
        plt.clf()
    except:
        pass

    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.0)

    # Bar plot
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(df4plot, x="Metric", y="Value", hue="Task")
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=30, ha="right")
    plt.tick_params(axis="both", which="major", length=3)
    plt.tick_params(axis="both", which="minor", length=2)
    # plt.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    # plt.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    fig.tight_layout()

    if fig_name is not None and dir4save is not None:
        fig.savefig(os.path.join(dir4save, f"{fig_name}.png"), dpi=300)
        fig.savefig(os.path.join(dir4save, f"{fig_name}.pdf"))
    return fig
