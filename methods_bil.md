Trait-associated multi-omics network inference via multi-task NMIC-guided adaptive multi-scale graph embedding
基于多任务NMIC引导的自适应多尺度图嵌入的性状关联多组学网络推断
Prior normalized maximal information coefficient (NMIC) network initialization: The network relationships are based on the raw feature values of multiple omics, without prior normalization or standardization. Feature value transformation is performed by the trainable multi-omics embedding module within the model, as normalization is unnecessary for NMIC computation. This approach also facilitates batch correction. For each feature, variable-sized sliding windows are applied to sorted values to compute coefficients of variation (CV), with the maximum value selected from the results of these sliding windows. The maximal coefficient of variation (MCV) of feature i can be expressed as:
先验归一化最大信息系数（NMIC）网络初始化：基于多组学的原始特征值构建网络中的节点关系，无需事先进行归一化或标准化处理。模型内部的可训练的多组学嵌入模块会对特征值进行变换，因此在计算NMIC时无需归一化，这种方式也有利于进行批次校正。对于每个特征，对排序后的数值应用大小可变的滑动窗口来计算变异系数（CV），并从这些滑动窗口的结果中选取最大值。特征 i 的最大变异系数（MCV）可表示为：

MCV_i=\max\below{1\le s\le\mathrm{count}\left(S\right)}\left\{\frac{\sigma_{i,s}}{\mu_{i,s}}\right\}	(1)
where \sigma_{i,s} and \mu_{i,s} represent the standard deviation and the average of the subsets of feature i extracted by the sliding window s respectively.
其中，\sigma_{i,s} 和 \mu_{i,s} 分别表示由滑动窗口 s 提取的特征 i 子集的标准差和平均值。
The features with MCVs below the specified threshold are removed. This step is beneficial for the interference of sparsity and outliers.
将MCV低于指定阈值的特征移除，有助于减少稀疏性和异常值的干扰。

To ensure a robust quantification of the relationships among features, the Normalized Maximal Mutual Information Coefficient (NMIC) is employed for each feature pair, utilizing two-dimensional variable-sized sliding windows. This approach effectively reduces the impact of noise, including interference from outliers and batch effects. Initially, the pairs of feature values are sorted based on the numerical order of one feature to preserve the one-to-one correspondence of the samples. Subsequently, variable-sized sliding windows are applied to extract distinct subsets of samples from the entire dataset of each feature pair, contingent upon the specified window size and step size. Following this, the normalized mutual information values are computed for these sample subsets, with the maximum value being designated as the NMIC. The mutual information between two variables can be estimated through the histogram-based method. To ascertain the optimal bin width for constructing the histogram, the Freedman-Diaconis rule (xxxxx) is applied. The {\mathrm{NMIC}}_{i,j} between the i-th feature X_i and the j-th feature X_j is defined as:
为确保对特征之间的关系进行稳健的量化，使用二维的可变大小的滑动窗口来计算每对特征的NMIC。这种方法可有效降低噪声影响，包括异常值和批次效应的干扰。首先，根据一个特征数值排序的样本顺序对一对特征的值进行排序，以保持样本的一一对应关系。随后，根据指定的窗口大小和步长，应用可变大小的滑动窗口提取不同的样本子集。接着，计算这些样本子集的归一化互信息值，并将最大值指定为NMIC。两个变量之间的互信息可通过基于直方图的方法进行估计。构建直方图的最佳分箱宽度由Freedman-Diaconis规则确定。第i个特征 X_i 与第j个特征 X_j 之间的 {\mathrm{NMIC}}_{i,j} 定义如下：

{\mathrm{NMIC}}_{i,j}=\max\below{1\le s\le\mathrm{count}\left(S\right)}xi,s∈Xi,sxj,s∈Xj,sPXi,s,Xj,sxi,s,xj,slog2PXi,s,Xj,sxi,s,xj,sPXi,sxi,sPXj,sxj,slog2kXi,skXj,s
(2)
h_{X_{i,s}}=2\times\frac{\mathrm{IQR} \left(X_{i,s}\right)}{\sqrt[3]{n_{X_{i,s}}}}	(3)
k_{X_{i,s}}=\left\lceil\frac{range\left(X_{i,s}\right)}{h_{X_{i,s}}}\right\rceil	(4)
P_{\left(X_{i,s},X_{j,s}\right)}\left(x_{i,s},x_{j,s}\right)=\frac{\mathrm{count} \left(x_{i,s},x_{j,s}\right)}{n_{X_{i,s}}}	(5)
P_{X_{i,s}}\left(x_{i,s}\right)=xj,s∈Xj,sPXi,s,Xj,sxi,s,xj,s
(6)
where X_{i,s}\subseteq X_i and X_{j,s}\subseteq X_j represent subsets of feature values extracted by the sliding window s; x_{i,s}\in X_{i,s} and x_{j,s}\in X_{j,s} are feature values; P_{\left(X_{i,s},X_{j,s}\right)} is the joint probability mass function of X_{i,s} and X_{j,s}; P_{X_{i,s}} and P_{X_{j,s}} are the marginal probability mass functions of X_{i,s} and X_{j,s} respectively. The Freedman-Diaconis rule is used to select the bin width h_{X_{i,s}} for the histogram of X_{i,s}. \mathrm{IQR}\left(X_{i,s}\right) represents the interquartile range of X_{i,s} and n_{X_{i,s}} is the number of observations in X_{i,s}. k_{X_{i,s}} is the number of bins for the histogram of X_{i,s}. The range of normalized {\mathrm{NMIC}}_{i,j}\in\left[0,1\right].
其中，X_{i,s}\subseteq X_i 表示 X_{j,s}\subseteq X_j 与 X_{j,s}\subseteq X_j 表示由滑动窗口 s 提取的特征数值子集；x_{i,s}\in X_{i,s} 和 x_{j,s}\in X_{j,s} 是特征的值；P_{\left(X_{i,s},X_{j,s}\right)} 是 X_{i,s} 和 X_{j,s} 的联合概率质量函数； P_{X_{i,s}}和 P_{X_{j,s}} 分别是 X_{i,s} 和 X_{j,s} 的边际概率质量函数。Freedman-Diaconis规则用于选择 X_{i,s} 直方图的分箱宽度 h_{X_{i,s}}。\mathrm{IQR}\left(X_{i,s}\right) 表示 X_{i,s} 的四分位数间距，n_{X_{i,s}} 是 X_{i,s} 中的观测数。k_{X_{i,s}} 是 X_{i,s} 直方图的分箱数量。归一化后的 {\mathrm{NMIC}}_{i,j} 取值范围为 \left[0,1\right] 。

An observation can be abstracted as a graph \mathcal{G}=\left(\mathcal{V},\ \mathcal{E}\right), where \mathcal{V} refers to a set of nodes (features) and \mathcal{E} refers to a set of undirected edges. The prior graph (network) is initialized with NMICs as edge weights and features values are converted into 1-dimensional vectors as initial node embeddings. Like feature selection, we retain edges with weights greater than a specified threshold (xxxxx) in this step, and features without edges are removed. To achieve efficient multi-threading, low memory usage, and significant time savings, we implemented the described algorithm using Rust, leveraging its performance and safety features to optimize the computational process.
一个观测值可抽象为一个图 \mathcal{G}=\left(\mathcal{V},\ \mathcal{E}\right)，其中 \mathcal{V} 指节点（特征）集，\mathcal{E} 指无向边集。先验图（网络）用NMIC作为边权重进行初始化，特征值被转换为一维向量作为初始节点嵌入。与特征选择类似，在这一步中，我们保留权重高于指定阈值的边，并移除没有边连接的特征。为实现高效的多线程处理、低内存占用并大幅节省时间，我们使用Rust语言实现上述算法，利用其性能和内存安全性特点优化计算过程。

NMIC-guided skip-connected graph attention layers for multi-omics feature representation learning. To learn comprehensive high dimensional latent representations for multi-omics features, we fuse feature inherences and values. A trainable feature embedding layer \mathbf{E}\in\mathbb{R}^{\left|\mathcal{V}\right|\times d} is constructed for inherence learning where d is the embedding dimension (d=64). A 1-dimensional feature value is transformed to a 64-dimensional vector by MLP1 and then concatenated with its inherence embedding to a 128-dimensional vector. The MLP2 transforms the concatenated vector into a 64-dimensional feature-specific value embedding. Multi-omics feature embeddings are generated through adding feature-specific value embeddings to inherence embeddings:
NMIC引导的带有跳跃连接的图注意力用于多组学特征表示学习：为学习多组学特征的全面高维潜在表示，我们融合特征固有属性和特征值。构建一个可训练的特征嵌入层 \mathbf{E}\in\mathbb{R}^{\left|\mathcal{V}\right|\times d} 用于学习固有属性，其中 d 是嵌入维度（d=64）。通过MLP1将一维特征值转换为64维向量，然后与它的固有属性嵌入连接成一个128维向量。MLP2再将连接后的向量转换为64维的特定于特征固有属性的值嵌入。多组学特征嵌入通过将特定于特征固有属性的值嵌入与固有属性嵌入相加生成：
\mathbf{H}_0={\mathrm{MLP}}_2\left({\mathrm{MLP}}_1\left(h_{i,0}\right)\parallel\mathbf{E}_i\right)\oplus\mathbf{E}_i	(7)
where \mathbf{H}_0 denotes the fused embedding of feature i, \mathbf{E}_i denotes the inherence embedding of feature i, h_{i,0} is the 1-dimensional raw value of feature i.
其中，\mathbf{H}_0 表示特征 i 的融合嵌入，\mathbf{E}_i 表示特征 i 的固有属性嵌入，h_{i,0} 是特征 i 的一维原始值。

To construct the fine-grained network representing cellular or sample state, we propose NMIC-guided graph attention mechanism with skip connections. Transformations are applied to the feature embeddings as follows:
为细致地构建能表示细胞或样本状态的网络，我们提出带有跳跃连接的NMIC引导图注意力机制。对特征嵌入进行如下转换：
e\left(h_i,h_j\right)=\mathrm{LeakyReLU}\left(\mathbf{\alpha}^\mathrm{T}\mathbf{W}_1\left[h_i\parallel h_j\right]\right)	(8)
a_{i,j}=\mathrm{softmax}\left({\mathrm{NMIC}}_{i,j}\otimes e\left(h_i,h_j\right)\right)	(9)
h_i^\prime=j∈Niai,jW2hi
(10)
where h_i and h_j are node embeddings of ith and jth node, j\in N_i; e\left(h_i,h_j\right) is the correlation coefficient between node i and j; \mathbit{a}^T represents the transpose of the trainable attention weights; \mathrm{LeakyReLU}\left(\bullet\right) denotes the Leaky Rectified Linear Unit activation function; \mathbf{W}_1\in\mathbb{R}^{{2d}^\prime\times d} is a trainable weight matrix; \alpha_{i,j} is the attention coefficient weighted by \mathbf{M}_{ij}; softmax\left(\bullet\right) represents the softmax activation function; j\in N_i represents node j is in the set of neighboring nodes of node i. Subsequently, a weighted sum of \mathbf{W}_2 transformed neighboring node embeddings is used to obtain the node embedding h_i^\prime transformed to specified dimension.
其中，h_i 和 h_j 是第i个和第j个节点的嵌入，j\in N_i；e\left(h_i,h_j\right) 是节点i和j之间的相关系数；\mathbit{a}^T 是可训练的注意力权重的转置；\mathrm{LeakyReLU}\left(\bullet\right) 表示带泄漏修正线性单元激活函数；\mathbf{W}_1\in\mathbb{R}^{{2d}^\prime\times d} 是一个可训练权重矩阵；\alpha_{i,j} 是由 \mathbf{M}_{ij} 加权的注意力系数；softmax\left(\bullet\right) 表示softmax激活函数；j\in N_i 表示节点j在节点i的相邻节点集中。随后，通过对 \mathbf{W}_2 转换后的相邻节点嵌入进行加权求和，得到转换为指定维度的节点嵌入 h_i^\prime 。
If multi-head attention is enabled, the transformed node embeddings are averaged per node.
如果启用多头注意力机制，则对每个节点的变换后的嵌入进行平均。
For situations that k (k\geq3) NMIC-guided graph attention layers are used, the skip connections are utilized which can be expressed as:
当使用 k（k\geq3）个NMIC引导的图注意力层时，采用跳跃连接：
h_k^\prime=h_k+\frac{1}{k-2}\sum_{m=1}^{k-2}{\mathbf{W}_mh_m},\mathrm{\ s.t.\ }k\geq3	(11)
where \mathbit{h}_k^\prime is the layers’ final output with former layers added; k is the total number of NMIC-guided graph attention layers; \mathbit{h}_m is the set of node embeddings of mth layer; \mathbit{W}_m\inR^{d_k\times d_m} is a trainable matrix for transforming embeddings in \mathbit{h}_m to the dimension d_k.
其中，\mathbit{h}_k^\prime 是最终输出的节点嵌入；k 是NMIC引导的图注意力层的总数；\mathbit{h}_m 是第m层的节点嵌入集；\mathbit{W}_m\inR^{d_k\times d_m} 是一个可训练的矩阵，用于将 \mathbit{h}_m 中的嵌入转换为 d_k 维度。

Adaptive multi-scale subgraph division: To retain more comprehensive graph structural information and get more representative graph embedding, we propose an adaptive multi-scale graph pooling method, which is also convenient for the key bio-molecular network inference in biological perspective. The correlation coefficients between nodes in \mathbit{h}_k^\prime are calculated as:
自适应多尺度子图划分：为保留更全面的图结构信息并获得更具代表性的图嵌入，我们提出一种自适应多尺度图池化方法，从生物学角度来看，这也便于关键生物分子网络的推断。首先计算 \mathbit{h}_k^\prime 中节点之间的相关系数：
\mathbit{A}_{ij}=e_{\mathrm{CS}}\left(i,j\right)=\left|cos\left(h_i,h_j\right)\right|=\left|\frac{h_i^Th_j}{\left|h_i\right|\bullet\left|h_j\right|}\right|	(12)
where \mathbit{A} is the adjacency matrix of \mathbit{h}_k^\prime, j\in N_i.
其中，\mathbit{A} 是 \mathbit{h}_k^\prime 的邻接矩阵，j\in N_i（即j与i之间存在连接）。

For subgraph division, ith node’s connectivity score is defined as:
对于子图划分，第i个节点的连接性得分定义为：
s_i=j∈Niai,j
(13)
Here, s_i represents the connectivity score of ith node, equals to the summary of ith node’s neighbors’ attention weights.
s_i 表示第i个节点的连接性得分，等于第i个节点的邻居注意力权重之和。

Nodes are ranked in descending order according to their connectivity scores. A histogram is employed, and adaptive binning is performed using the method described in xxxxx (specifically, the Freedman-Diaconis rule, as applied in NMIC steps), based on the parameter s. Multi-scale subgraphs are defined by aggregating nodes from individual bins and their combinations, ensuring coverage of adjacent bins. Local subgraphs are constructed by incorporating neighboring nodes up to a predefined depth, while global subgraphs are formed by including all key nodes from the top bins. It is important to note that subgraphs may overlap in terms of shared nodes, even at similar scales. Furthermore, nodes with low connectivity scores or those forming isolated subgraphs can still significantly influence target traits.
根据连接性得分对节点进行降序排序，使用直方图并采用Freedman-Diaconis规则进行自适应分箱。通过聚合单个分箱以及多个相邻分箱的组合中的节点来定义多尺度子图。通过纳入预定义深度内的相邻节点构建局部子图，全局子图包含连接性得分最高的分箱中的所有关键节点。需要注意的是，在相似尺度下，子图可能共享节点。连接性得分较低的节点或形成孤立子图的节点仍可能对目标性状产生显著影响。

Multi-scale graph attention pooling for graph-level representation learning: The subgraphs are processed through multiple pooling layers operating at distinct scales. At each scale, nodes are selectively routed to a shared NMIC-guided GAT based on dynamically computed connectivity scores, preserving skeletal structural information. Following multi-scale subgraph embedding, a NMIC-guided GAT and an attention-based learned aggregation mechanism synthesize subgraph embeddings into a unified graph-level representation. This fusion strategy augments the framework’s ability to jointly model fine-grained local features and coarse-grained global hierarchies. The resultant graph embedding integrates multi-scale structural information, facilitating downstream applications such as feature imputation, graph classification, and clustering.
多尺度图注意力池化用于图级表示学习：子图通过在不同尺度下的多个池化层进行处理。在每个尺度上，根据动态计算的连接性得分，将相应节点嵌入输入一个共享的NMIC引导的GAT层，以保留图的主要骨干结构信息。在进行多尺度子图嵌入后，另一个NMIC引导的GAT层和基于自注意力的学习聚合机制将多个子图嵌入合成为统一的图级表示。这种融合策略增强了DeepTAN框架对细粒度局部特征和粗粒度全局层次结构进行联合建模的能力。得到的图嵌入整合了多尺度结构信息，便于进行特征插补、图分类和聚类等下游应用。

Multi-task learning for trait-associated network inference
多任务学习赋能性状相关网络推断
Synergistically enhancing the performance across multiple tasks by iteratively updating their shared model parameters facilitates the integration and elucidation of latent knowledge in the relationships between phenotypes and omics patterns. This approach not only optimizes task-specific outcomes but also promotes a deeper understanding of the underlying biological mechanisms. Within the xxxxx framework, the multi-task paradigm leverages the flexibility afforded by graph deep learning to accommodate diverse input data structures. This enables the model to be trained on both unlabeled and labeled datasets by concurrently or independently optimizing for label prediction loss and feature imputation loss, thereby enriching its capacity to uncover complex biological insights.
通过迭代更新共享模型参数来协同提升多个任务的性能，有助于整合和阐释表型与组学模式之间关系中的潜在知识。这种方法不仅优化了特定任务的结果，还促进了对潜在生物学机制的更深入理解。在DeepTAN框架中，多任务范式利用图深度学习提供的灵活性来适应各种输入数据结构。这使得模型能够在未标记和标记数据集上进行训练，通过同时或独立优化标签预测损失和特征插补损失，丰富了其揭示复杂生物学现象的能力。

Feature Imputation Task. Given a graph-level embedding encoding a sample’s biological state (with each graph representing one sample) and the intrinsic node embeddings (where nodes correspond to omics features), the objective is to reconstruct node embeddings that align with the biological context captured by the graph-level representation. This constitutes a feature imputation task, inferring missing or corrupted features based on their structural topology within the graph and the holistic biological context derived from the graph-level embedding.
特征插补任务：给定编码样本生物学状态的图级嵌入（每个图代表一个样本）和节点固有属性嵌入（其中节点对应组学特征），目标是重建图级表示所捕获的生物学背景相符的节点嵌入。这构成了一个特征插补任务，即基于图内的结构拓扑以及从图级嵌入中的整体生物学状态，推断缺失或损坏的特征。
To achieve this, we designed a graph decoder which is expressed as:
为实现这一目标，我们设计了一个图解码器，表示为：
\widehat{\mathbf{H}_\mathrm{S}}={\rm FFN}_I(\mathbf{Z}_n\ ||\ \mathbf{E})\oplus\mathbf{E}	(14)
X_\mathrm{I}={\rm FFN}_{I,Q}(\widehat{\mathbf{H}_\mathrm{S}})	(15)
\mathcal{L}_I=KL(\mathbf{X}_\mathrm{I},\ \mathbf{X})+MSE(\mathbf{X}_\mathrm{I},\ \mathbf{X})	(16)
where \widehat{\mathbf{H}_\mathrm{S}} denotes the imputed biological state-specific feature embeddings, \mathbf{Z}_n is the graph-level embedding, and \mathbf{E} is feature inherence embeddings. {\rm FFN}_I is a three-layer feedforward neural network that transforms concatenated graph-level embedding and feature inherence embedding to the same shape as \mathbf{E}. The imputed feature values X_\mathrm{I} is converted from the biological state-specific feature embeddings \widehat{\mathbf{H}_\mathrm{S}} by the feedforward neural network {\rm FFN}_{I,Q}. The loss \mathcal{L}_I of feature imputation task is the sum of the Kullback-Leibler divergence and the mean squared error between raw feature values and predicted feature values.
其中，\widehat{\mathbf{H}_\mathrm{S}} 表示插补后的生物学状态特异的特征嵌入，\mathbf{Z}_n 是生物学状态特异的图级嵌入，\mathbf{E} 是特征固有属性嵌入。{\rm FFN}_I 是一个3层前馈神经网络，将连接后的图级嵌入和特征固有属性嵌入转换为与 \mathbf{E} 相同的形状。插补后的特征值 X_\mathrm{I} 由前馈神经网络 {\rm FFN}_{I,Q} 从生物学状态特异的特征嵌入 \widehat{\mathbf{H}_\mathrm{S}} 转换得到。特征插补任务的损失 \mathcal{L}_I 是原始特征值与预测特征值之间的Kullback-Leibler散度和均方误差之和。

The task for label prediction. Since the biological state-specific graph-level embedding is obtained through multi-scale graph embedding, it is highly suitable for further regression or classification tasks, such as phenotypic prediction and cell type annotation. In the xxxxx framework, multilayer perceptron (MLP) is implemented:
标签预测任务：通过多尺度图嵌入获得的生物学状态特异的图级嵌入适合用于进一步的回归或分类任务，如表型预测和细胞类型注释。在DeepTAN框架中，采用多层感知器：
\widehat{y_{sc}}={\rm MLP}_{sc}(\mathbf{Z}_n)	(17)
\widehat{y_{bulk}}={\rm MLP}_{bulk}(\mathbf{Z}_n)	(18)
\mathcal{L}_{P,R}=MSE(\widehat{y_R},\ y_R)	(19)
\mathcal{L}_{P,C}=CE(\widehat{y_C},\ y_C)	(20)
where {\rm MLP}_{sc} and {\rm MLP}_{bulk} are different MLPs for cell type classification and phenotypic prediction, respectively. The regression loss function is MSE and the classification loss function is cross entropy (CE).
其中，{\rm MLP}_{sc} 和 {\rm MLP}_{bulk} 分别是用于细胞类型分类和表型预测的不同MLP。回归损失函数是均方误差，分类损失函数是交叉熵。

Trait-associated network inference. By computing the cosine similarity of each pair of features in \widehat{\mathbf{H}_\mathrm{S}} and consider it as the edge weight in the graph (i.e., feature relational unit), the biological state-specific networks (where each network represent a sample) are formed:
性状相关网络推断：通过计算 \widehat{\mathbf{H}_\mathrm{S}} 中每对特征的余弦相似度，并将其视为图中的边权重（即特征关系单元），形成生物学状态特异的网络（一个网络代表一个样本）：
e_R\left(i,j\right)=cos\left(h_i,h_j\right)	(21)
where e_R\left(i,j\right) denotes the edge weight between feature i and feature j. cos\left(h_i,h_j\right) is the cosine similarity between the embeddings of feature i and feature j.
其中，e_R\left(i,j\right) 表示特征i和特征j之间的边权重。cos\left(h_i,h_j\right) 是特征i和特征j的嵌入之间的余弦相似度。
Utilizing the specified correlation function (such as the Pearson correlation coefficient or mutual information), we can quantify the relationships between phenotypes and feature relational units. The trait-associated network is constructed based on these relationships, with the edge weights representing the association strengths between the trait and the feature relational units. Furthermore, we can extract sub-networks that have various degrees of association with traits by setting the threshold for the association strength.
运用指定的相关性函数（如皮尔逊相关系数、互信息），我们可以量化表型与特征关系单元之间的关系。基于这些关系构建性状关联网络，边权重表示性状与特征关系单元之间的关联强度。进一步，我们可以通过设置关联强度阈值来提取与性状具有不同关联程度的子网络。
