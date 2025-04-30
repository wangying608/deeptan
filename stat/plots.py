import os
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pacmap
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
            _dataset[_task][dim][_met] = {}
            for _split in const.dkey.splits:
                try:
                    _fname = metrics_data.metrics_dict["summary_recon"].filter((pl.col("task") == _task) & (pl.col("metric") == _met) & (pl.col("seed_num") == seed) & (pl.col("split") == _split))["fname"].item()
                    _dataset[_task][dim][_met][_split] = metrics_data.metrics_dict["metrics"]["recon"][_fname][dim][_met]
                except:
                    continue
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
    split: str = "tst",
):
    try:
        plt.close("all")
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
    sns.set_context("paper", font_scale=1.1)

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
            x_data = dataset[x_lab][dim][_met][split]["values"]
            y_data = dataset[y_lab][dim][_met][split]["values"]

            # 绘制主KDE图
            sns.kdeplot(ax=ax, x=x_data, y=y_data, fill=True)

            # 创建边缘分布的坐标轴
            ax_histx = ax.inset_axes([0, 1.04, 1, 0.25], sharex=ax)
            ax_histy = ax.inset_axes([1.04, 0, 0.25, 1], sharey=ax)

            # 绘制边缘分布
            sns.kdeplot(x=dataset[x_lab][dim][_met][split]["values"], ax=ax_histx, fill=True, legend=False)
            sns.kdeplot(y=dataset[y_lab][dim][_met][split]["values"], ax=ax_histy, fill=True, legend=False)

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


def unpivot_summary(df: pl.DataFrame, split: str, seed: Optional[int] = None) -> pl.DataFrame:
    r"""
    Unpivot a summary dataframe to long format.
    """
    df = df.filter(pl.col("split") == split)
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
    ind_cols = ["Method", "Capability", "task"]
    on_cols = _df.columns
    for _c in ind_cols:
        on_cols.remove(_c)
    _df = _df.unpivot(index=ind_cols, on=on_cols, variable_name="metric", value_name="value")
    return _df


def metrics_plot_data(metrics_data: MetricsDictMaker, split: str, seed: Optional[int] = None):
    _df_plot_label = unpivot_summary(metrics_data.metrics_dict["summary_label"], split, seed=seed)
    _df_plot_recon = unpivot_summary(metrics_data.metrics_dict["summary_recon"], split, seed=seed).select(_df_plot_label.columns)
    _df_plot_clust = unpivot_summary(metrics_data.metrics_dict["summary_clustering"], split, seed=seed).select(_df_plot_recon.columns)
    _df_plot_allmetrics = _df_plot_recon.vstack(_df_plot_label).vstack(_df_plot_clust)

    # Pick metrics that are smaller the better and apply 1-value
    metrics_sb = ["mse", "mae", "jsd"]
    _df_plot_allmetrics = _df_plot_allmetrics.with_columns(pl.when(pl.col("metric").is_in(metrics_sb)).then(1 - pl.col("value")).otherwise(pl.col("value")).alias("value"))

    # Map metrics to a more readable name
    _df_plot_allmetrics = _df_plot_allmetrics.with_columns(pl.col("metric").map_elements(lambda x: const.dkey.title_metric_mapping.get(x, x), return_dtype=pl.Utf8).alias("metric"))
    _df_plot_allmetrics = _df_plot_allmetrics.with_columns(pl.col("task").map_elements(lambda x: const.dkey.title_task_mapping.get(x, x), return_dtype=pl.Utf8).alias("task"))
    _df_plot_allmetrics = _df_plot_allmetrics.rename(const.dkey.title_colnameC2_mapping)

    return _df_plot_allmetrics

    # Prepare data for radar chart
    # _tasks = _df_plot_allmetrics.group_by("Task").agg(pl.col("Value").alias("Values"))
    # _metrics = _df_plot_allmetrics["Metric"].unique().to_list()
    # _task_dict = {_t: _tasks.filter(pl.col("Task") == _t)["Values"].to_list()[0] for _t in _tasks["Task"]}

    # # Prepare angles for radar chart
    # _n_metrics = len(_metrics)
    # angles = np.linspace(0, 2 * np.pi, _n_metrics, endpoint=False).tolist()
    # for key in _task_dict:
    #     _task_dict[key] += _task_dict[key][:1]
    # angles += angles[:1]

    # return _df_plot_allmetrics, _task_dict, angles


def metrics_plot(
    df4plot: pl.DataFrame,
    fig_name: Optional[str] = None,
    dir4save: Optional[str] = None,
):
    try:
        plt.close("all")
    except:
        pass

    # 设置主题和上下文
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.1)

    # 假设 df4plot 是输入数据框
    # 计算每个 Capability 的 Metric 数量
    capability_metric_counts = df4plot.group_by("Capability").agg(pl.count("Metric").alias("num_metrics")).sort("Capability").to_pandas()

    # 动态计算每个子图的高度
    base_height = 0.06  # 每个 Metric 占用的高度
    heights = capability_metric_counts["num_metrics"] * base_height

    # 创建子图布局
    fig = plt.figure(figsize=(3.1, sum(heights)))  # 总高度为所有子图高度之和
    gs = gridspec.GridSpec(len(heights), 1, height_ratios=heights)

    # 收集所有图例的handles和labels
    all_handles = []
    all_labels = []

    # 遍历每个 Capability，绘制子图
    for i, (capability, num_metrics) in enumerate(zip(capability_metric_counts["Capability"], capability_metric_counts["num_metrics"])):
        # 筛选出当前 Capability 的数据
        data_subset = df4plot.filter(pl.col("Capability") == capability).to_pandas()

        # 创建子图
        ax = fig.add_subplot(gs[i])

        # 绘制水平条形图
        sns_plot = sns.barplot(
            data=data_subset,
            x="Value",
            y="Metric",
            hue="Task",
            orient="h",
            palette="colorblind",
            ax=ax,
            width=0.8,  # 固定柱子宽度
        )

        # 设置标题和标签
        ax.set_title(capability, y=0.98)
        ax.set_xlabel("Value" if i == len(heights) - 1 else "")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", length=2)
        ax.tick_params(axis="y", length=0)  # 移除 y 轴刻度线

        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=30, ha="right")

        # 移除子图图例但保存handles和labels
        if sns_plot.legend_ is not None:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
            sns_plot.legend_.remove()

        # 删除顶部和右侧边框
        ax.spines["top"].set_visible(False)  # 隐藏顶部边框
        ax.spines["right"].set_visible(False)  # 隐藏右侧边框

    # 创建全局图例（去重）
    unique_labels = []
    unique_handles = []
    seen_labels = set()

    for handle, label in zip(all_handles, all_labels):
        if label not in seen_labels:
            seen_labels.add(label)
            unique_labels.append(label)
            unique_handles.append(handle)
    if unique_handles:  # 只有存在图例项时才添加
        _legend = fig.legend(
            unique_handles,
            unique_labels,
            loc="lower center",
            ncol=min(3, len(unique_labels)),
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
        )

    # 调整布局
    plt.tight_layout()

    if fig_name is not None and dir4save is not None:
        os.makedirs(dir4save, exist_ok=True)
        fig.savefig(os.path.join(dir4save, f"{fig_name}.png"), dpi=300, bbox_extra_artists=[_legend], bbox_inches="tight", pad_inches=0.1)
        fig.savefig(os.path.join(dir4save, f"{fig_name}.pdf"), bbox_extra_artists=[_legend], bbox_inches="tight", pad_inches=0.1)

    return fig


def pacmap_plot_data(metrics_data: MetricsDictMaker, _tasks: List[str], split: str, seed: Optional[int] = None):
    # Get cell embeddings for each task
    cell_embs = {}
    _fnames = []
    for _task in _tasks:
        _fname = metrics_data.ident.filter((pl.col("task") == _task) & (pl.col("seed_num") == seed) & (pl.col("split") == split))["fname"].item()
        _fnames.append(_fname)
        cell_embs[_task] = metrics_data.metrics_dict["prediction"][_fname]["g_embedding"]

    # Get predicted cell labels for each task
    # 获取所有文件中唯一的细胞类型标签
    celltypes_uniq = metrics_data.metrics_dict["prediction"][_fnames[0]]["label_names"][1:]
    celltypes_uniq = [i.replace("ct_", "") for i in celltypes_uniq]
    print("Unique cell types: ", celltypes_uniq)

    ys_pred_numeric = {}
    ys_pred_text = {}
    for _fname in _fnames:
        _task = metrics_data.ident.filter(pl.col("fname") == _fname)["task"].item()
        ys_pred_numeric[_task] = metrics_data.metrics_dict["prediction"][_fname]["y"].argmax(axis=1)
        ys_pred_text[_task] = [celltypes_uniq[i] for i in ys_pred_numeric[_task]]

    # Get true cell types
    y_true_text = metrics_data.metrics_dict["true"][f"seed_{seed}_{split}"]["y_df_flatten"]["ct"].to_list()

    # =========================== Compute PaCMAP =================================
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.3, FP_ratio=4.0)

    # # fit the data (The index of transformed data corresponds to the index of the original data)
    # X_transformed = embedding.fit_transform(X, init="pca")

    cell_embs_pacmap = {}
    for _task in cell_embs.keys():
        cell_embs_pacmap[_task] = embedding.fit_transform(cell_embs[_task], init="pca")
        print(f"Task {_task}: {cell_embs_pacmap[_task].shape}")

    return cell_embs_pacmap, y_true_text, ys_pred_text


def pacmap_plot(
    cell_embs_pacmap: Dict,
    _tasks_text,
    y_true_text,
    ys_pred_text,
    fig_name: Optional[str] = None,
    dir4save: Optional[str] = None,
):
    try:
        plt.close("all")
    except:
        pass

    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.0)

    # 1. 准备统一的颜色映射
    # 收集所有可能的类别标签
    all_categories = set(y_true_text)
    for task in cell_embs_pacmap.keys():
        all_categories.update(ys_pred_text[task])
    all_categories = sorted(list(all_categories))

    # 创建统一的调色板
    palette = sns.color_palette("husl", len(all_categories))
    color_dict = {cat: palette[i] for i, cat in enumerate(all_categories)}

    # A4纸的宽度约为21厘米（8.27英寸），高度可以根据需要调整
    # a4_width_cm = 21
    # cm_to_inches = 0.393701
    # a4_width_inches = a4_width_cm * cm_to_inches
    n_cols = 2
    n_rows = len(cell_embs_pacmap)

    fig_width = 2.6 * n_cols  # 调整图表的整体宽度以适应列数
    fig_height = 2.4 * n_rows  # 调整图表的整体高度以适应行数

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

    # 用于收集图例项的字典
    legend_dict = {}
    for i, task in enumerate(cell_embs_pacmap.keys()):
        # 第一个子图 - 真实标签
        ax = axes[i, 0]
        ax.set_ylabel(_tasks_text[i])
        ax.set_title("Annotated", y=0.95)

        # 绘制散点图
        scatter = sns.scatterplot(
            x=cell_embs_pacmap[task][:, 0],
            y=cell_embs_pacmap[task][:, 1],
            alpha=0.5,
            hue=y_true_text,
            palette=color_dict,
            s=4,
            ax=ax,
        )

        # 收集图例项
        if scatter.legend_ is not None:
            handles, labels = ax.get_legend_handles_labels()
            for _h, _l in zip(handles, labels):
                if _l not in legend_dict:
                    legend_dict[_l] = _h
            scatter.legend_.remove()
        # 第二个子图 - 预测标签
        ax = axes[i, 1]
        ax.set_title("Predicted", y=0.95)
        scatter = sns.scatterplot(
            x=cell_embs_pacmap[task][:, 0],
            y=cell_embs_pacmap[task][:, 1],
            alpha=0.5,
            hue=ys_pred_text[task],
            palette=color_dict,
            s=4,
            ax=ax,
        )

        # 收集图例项
        if scatter.legend_ is not None:
            handles, labels = ax.get_legend_handles_labels()
            for _h, _l in zip(handles, labels):
                if _l not in legend_dict:
                    legend_dict[_l] = _h
            scatter.legend_.remove()
        # 统一设置子图样式
        for col in [0, 1]:
            ax = axes[i, col]
            # 移除边框
            for spine in ax.spines.values():
                spine.set_visible(False)
            # 移除刻度
            ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    # 添加全局图例
    if legend_dict:
        # 按类别名称排序
        sorted_items = sorted(legend_dict.items(), key=lambda x: x[0])
        sorted_labels = [item[0] for item in sorted_items]
        sorted_handles = [item[1] for item in sorted_items]

        fig.legend(
            sorted_handles,
            sorted_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05 - 0.02 * n_rows),
            ncol=min(6, len(sorted_labels)),
            frameon=False,
        )

    fig.tight_layout(pad=1.2)

    # 保存图像
    if fig_name and dir4save:
        os.makedirs(dir4save, exist_ok=True)
        save_path = os.path.join(dir4save, fig_name)

        # 确保图例被包含在保存的图像中
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight", pad_inches=0.1)

    return fig
