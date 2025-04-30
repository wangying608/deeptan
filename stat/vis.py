import sys
import polars as pl
import numpy as np
import graph_tool.all as gt
import matplotlib.cm
import matplotlib.pyplot as plt


def vis_gt_demo():
    r"""Visualize the food web dataset.
    Hierarchical community detection.
    """
    gt.seed_rng(47)

    g = gt.collection.ns["foodweb_baywet"]

    sargs = dict(recs=[g.ep.weight],
                rec_types=["real-exponential"])
    state = gt.minimize_nested_blockmodel_dl(g, state_args=sargs)

    state.draw(
        edge_color=gt.prop_to_size(
            g.ep.weight,
            power=1,
            log=True),
        ecmap=(matplotlib.cm.inferno, .6),
        eorder=g.ep.weight,
        edge_pen_width=gt.prop_to_size(
            g.ep.weight,
            1, 4,
            power=1,
            log=True),
        edge_gradient=[],
    )


def read_micg_to_gt(path_micg_npz: str, skip_na_nodes: bool = True) -> gt.Graph:
    data = np.load(path_micg_npz)
    # print(data.files)
    # ['mi_values', 'feat_pairs', 'processed_mat', 'mat_feat_indices', 'mat_simi_feat_pairs', 'thre_cv', 'thre_pcc', 'thre_mi', 'ratio_max_window', 'ratio_min_window', 'ratio_step_window', 'ratio_step_sliding']
    # max_feat_index = data["mat_feat_indices"].max()
    g = gt.Graph(directed=False)
    # g.add_vertex(max_feat_index + 1)

    # Init edge weights
    tmp_edge_list_np: np.ndarray = data["feat_pairs"]
    mic_values: np.ndarray = data["mi_values"].reshape(-1, 1)
    tmp_edge_list_pl = pl.DataFrame(tmp_edge_list_np, schema={"src": pl.Int64, "dst": pl.Int64}).hstack(pl.DataFrame(mic_values, schema={"mic": pl.Float64}))
    
    if skip_na_nodes:
        map_old2new = {old: new for new, old in enumerate(data["mat_feat_indices"])}
        edge_list = [(map_old2new[row["src"]], map_old2new[row["dst"]], row["mic"]) for row in tmp_edge_list_pl.iter_rows(named=True)]
    else:
        edge_list = [(row["src"], row["dst"], row["mic"]) for row in tmp_edge_list_pl.iter_rows(named=True)]

    # Add edges (mi_values are edge weights, feat_pairs are edge endpoints)
    edge_weights = g.new_ep("double")
    
    g.add_edge_list(edge_list, eprops=[edge_weights])
    # print(edge_weights.fa)
    g.edge_properties["weight"] = edge_weights

    return g


def vis_my_graph(path_micg_npz: str):
    r"""
    """
    # plt.switch_backend("cairo")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    g = read_micg_to_gt(path_micg_npz)
    # print(g.ep.weight)
    # <EdgePropertyMap object with value type 'double', for Graph 0x7c060ddefbc0, at 0x7c05dccca180>

    # Plot the graph
    p_a = gt.graph_draw(g, vertex_size=1.5, mplfig=ax[0])
    p_a.fit_view(yflip=True)
    # ax[0,0].set_xlabel("$x$ coordinate")
    # ax[0,0].set_ylabel("$y$ coordinate")

    _state = gt.minimize_nested_blockmodel_dl(g)
    p_a = _state.draw(
        mplfig=ax[1],
        edge_color=gt.prop_to_size(
            g.ep.weight,
            power=1,
            log=True),
        ecmap=(matplotlib.cm.inferno, .6),
        eorder=g.ep.weight,
        edge_pen_width=gt.prop_to_size(
            g.ep.weight,
            1, 4,
            power=1,
        )
    )
    # p_a.fit_view(yflip=True)
    # ax[0,1].set_xlabel("$x$ coordinate")
    # ax[0,1].set_ylabel("$y$ coordinate")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    # fig.savefig("gt_mpl.svg")
    # svg file is too large, so we use png (300 dpi)
    fig.savefig("gt_mpl.png", dpi=300)


if __name__ == "__main__":
    # vis_gt_demo()
    path_npz = sys.argv[1]
    vis_my_graph(path_npz)
