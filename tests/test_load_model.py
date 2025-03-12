import pickle

import torch
import torch.nn as nn

from deeptan.graph.model import DeepTAN
from deeptan.utils.uni import get_map_location

path_ckpt = ""
path_dict_new = ""

if __name__ == "__main__":
    # Load new dict node names
    # with open(path_dict_new, "rb") as f:
    #     others2save = pickle.load(f)
    #     dict_node_names_new = others2save["dict_node_names"]
    dict_node_names_new = {"aa1": 0, "aa2": 1, "aa3": 2}

    # Load model
    _model_pre = DeepTAN.load_from_checkpoint(path_ckpt, map_location=get_map_location())

    # Extract AMSGP module
    _model_amsgp = _model_pre.amsgp
    dict_node_names_former = _model_amsgp.node_embedding_layers.dict_node_names

    print(_model_amsgp)
    print("\n", dict_node_names_former)

    # If new dict_node_names has different keys than former dict_node_names, update the former model's NodeEmbedding

    if set(dict_node_names_new.keys()) != set(dict_node_names_former.keys()):
        print("Updating dict_node_names in NodeEmbedding")
        new_nodes_to_append = set(dict_node_names_new.keys()) - set(dict_node_names_former.keys())
        n_node_former = len(dict_node_names_former)
        n_node_add = len(new_nodes_to_append)
        dict_to_add = {node: n_node_former + i for i, node in enumerate(new_nodes_to_append)}
        print(dict_to_add)

        # Update dict_node_names in NodeEmbedding
        dict_node_names_former.update(dict_to_add)
        _model_amsgp.node_embedding_layers.dict_node_names = dict_node_names_former

        # Update node embedding weights by concatenating new weights
        emb_dim = _model_amsgp.node_embedding_layers.embed.weight.size(1)
        new_embed = nn.Embedding(n_node_former + n_node_add, emb_dim, scale_grad_by_freq=True, sparse=True)
        new_embed.weight.data[:n_node_former] = _model_amsgp.node_embedding_layers.embed.weight.data
        nn.init.xavier_uniform_(new_embed.weight.data[n_node_former:])
        _model_amsgp.node_embedding_layers.embed = new_embed

    else:
        print("dict_node_names in NodeEmbedding is the same")
