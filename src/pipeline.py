r"""
DeepTAN pipelines for fitting, hyperparameter tuning, inference, and testing.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import lightning as ltn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from deeptan.graph.model import AMSGPMTL
from torch_geometric.utils import erdos_renyi_graph

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def generate_random_graph(
    num_nodes: int, num_features: int, num_classes: int, is_regression: bool
) -> Data:
    """
    生成一个随机的图数据对象。

    参数:
        num_nodes (int): 图中的节点数量。
        num_features (int): 每个节点的特征维度。
        num_classes (int): 类别数量（用于分类任务）或输出维度（用于回归任务）。
        is_regression (bool): 是否为回归任务。

    返回:
        Data: 随机生成的图数据对象。
    """
    # 随机生成节点特征
    x = torch.randn(num_nodes, num_features)  # 节点特征矩阵 (num_nodes, num_features)

    # 随机生成边索引 (使用 Erdős-Rényi 模型生成随机图)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.2)  # 边索引 (2, num_edges)

    # 随机生成边属性
    edge_attr = torch.rand(edge_index.size(1), 1)  # 边属性矩阵 (num_edges, 1)

    # 随机生成节点名称 (假设节点名称为字符串)
    node_names = [f"node_{i}" for i in range(num_nodes)]

    # 随机生成标签
    if is_regression:
        y = torch.rand(1)  # 回归任务标签 (num_nodes, output_dim)
    else:
        y = torch.randint(0, num_classes, (1,))  # 分类任务标签 (num_nodes,)

    # 创建图数据对象
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_names=node_names,
    )

    return graph_data


if __name__ == "__main__":
    # 示例：生成训练、验证和测试数据集
    train_dataset = [
        generate_random_graph(
            num_nodes=50, num_features=32, num_classes=10, is_regression=False
        )
        for _ in range(100)
    ]
    val_dataset = [
        generate_random_graph(
            num_nodes=50, num_features=32, num_classes=10, is_regression=False
        )
        for _ in range(20)
    ]
    test_dataset = [
        generate_random_graph(
            num_nodes=50, num_features=32, num_classes=10, is_regression=False
        )
        for _ in range(20)
    ]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dict_node_names_values = [i for i in range(50)]
    dict_node_names_keys = [f"node_{i}" for i in range(50)]
    dict_node_names = dict(zip(dict_node_names_keys, dict_node_names_values))
    # print(dict_node_names)
    input_dim = 32
    output_dim = 10
    is_regression = False

    # 打印生成的图数据信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"单个图的节点数量: {train_dataset[0].num_nodes}")
    print(f"单个图的边数量: {train_dataset[0].edge_index.size(1)}")

    # Initialize the model
    model = AMSGPMTL(
        dict_node_names=dict_node_names,
        input_dim=input_dim,
        output_dim=output_dim,
        is_regression=is_regression,
        node_emb_dim=128,
        fusion_dims_node_emb=[256, input_dim],
        output_dim_g_emb=128,
        n_hop=2,
        threshold_edge_exist=0.1,
        threshold_subgraph_overlap=0.9,
        n_heads_node_emb=4,
        n_heads_pooling=4,
        dropout=0.2,
        lr=1e-3,
        negative_slope=0.2,
        alpha=0.7,
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = ltn.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        logger=True,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_total", patience=10, mode="min"),
            ModelCheckpoint(monitor="val_total", mode="min"),
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, dataloaders=test_loader)

    # Optionally, you can save the trained model
    torch.save(model.state_dict(), "amsgp_mtl_model.pth")
