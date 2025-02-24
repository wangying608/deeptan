argv <- commandArgs(trailingOnly = TRUE)
path_rds = argv[1]

if (!requireNamespace("Seurat", quietly = TRUE)) {
  install.packages("Seurat")
}
library(Seurat)

# 读取 RDS 文件
seurat_object <- readRDS(path_rds)

# 打印 orig.ident 列
# print(seurat_object@meta.data$orig.ident)

# 打印矩阵信息
print(colnames(seurat_object@assays[[1]]@data))

# 查看特征信息
head(seurat_object@assays[[1]]@data)

# 查看元数据表
head(seurat_object@meta.data)

# 查看所有元数据列名
colnames(seurat_object@meta.data)
