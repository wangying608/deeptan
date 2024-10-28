import networkx as nx
import polars as pl


if __name__ == '__main__':
    # 创建一个带权无向图
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from([1, 2, 3, 4, 5])
    # 添加带权边
    edges = [
        (1, 2, 1.5),
        (1, 3, 2.0),
        (2, 4, 1.0),
        (3, 4, 1.8),
        (4, 5, 2.5)
    ]
    G.add_weighted_edges_from(edges)
    # 打印图的节点和边
    print("Nodes:", G.nodes())
    print("Edges:", G.edges(data=True))

    # 计算度中心性
    # 度中心性是指节点的度（即与该节点直接相连的边的数量）。度越高的节点越重要。
    degree_centrality = nx.degree_centrality(G)
    print("\nDegree Centrality:", degree_centrality)

    # 计算介数中心性
    # 介数中心性衡量一个节点在所有最短路径中出现的频率。介数越高的节点越重要。
    # 介数中心性是指节点在图中作为连接其他节点的桥梁的能力。介数越高的节点越重要。
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    print("Betweenness Centrality:", betweenness_centrality)

    # 计算紧密中心性
    # 紧密中心性衡量一个节点到所有其他节点的平均最短路径长度。紧密中心性越高的节点越重要。
    closeness_centrality = nx.closeness_centrality(G, distance='weight')
    print("Closeness Centrality:", closeness_centrality)

    # 计算特征向量中心性
    # 特征向量中心性不仅考虑节点的度，还考虑其邻居的度。一个节点的特征向量中心性与其邻居的特征向量中心性成正比。
    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
    print("Eigenvector Centrality:", eigenvector_centrality)

    # 计算PageRank
    # PageRank是一种基于随机游走的算法，用于衡量节点的重要性。
    # pagerank = nx.pagerank(G, weight='weight')
    # print("PageRank:", pagerank)

    # output_df = pl.DataFrame(betweenness_centrality, schema=[str(i) for i in list(G.nodes())])
    # print(output_df)

"""
选择哪种算法和指标取决于具体的研究问题和网络类型。例如，在研究蛋白质相互作用网络时，可能更关注介数中心性和K-核分解，而在研究代谢网络时，可能更关注紧密中心性和特征向量中心性。

idea: 依赖节点嵌入的动态加权中心性，也可通过查看训练好的模型来印证节点类别。

"""
