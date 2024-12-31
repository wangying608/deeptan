# 安装和加载必要的R包
# install.packages("ggplot2")
# install.packages("svglite")
library(ggplot2)
library(svglite)

plot_random_heatmap <- function(n_row, n_col, color_high, color_low="white", seed=1234) {
  set.seed(seed)
  data_matrix <- matrix(rnorm(n_row * n_col, sd = 2), nrow = n_row, ncol = n_col)
  
  # 将矩阵转换为数据框，并为行和列添加标签
  data_df <- as.data.frame(as.table(data_matrix))
  colnames(data_df) <- c("Row", "Column", "Value")
  
  # 绘制热图
  heatmap <- ggplot(data_df, aes(x = Column, y = Row, fill = Value)) +
    geom_tile(color = "white") +  # 添加色块
    scale_fill_gradient(low = color_low, high = color_high) +  # 设置颜色渐变
    theme_minimal() +  # 使用简洁的主题
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  # 旋转x轴标签
    labs(x = "", y = "")  # 移除轴标签
  return(heatmap)
  
}

mycolors <- c("darkblue", "darkgreen", "red", "cyan4", "coral", "#f5c71a")
mycolors.names <- c("darkblue", "darkgreen", "red", "cyan4", "coral", "deeplemon")
myseed <- c(42, 43, 44, 45, 46, 47)
# 保存为SVG格式
for (xc in 1:length(mycolors)) {
  ggsave(paste0("heatmap_", mycolors.names[xc], ".svg"), plot = plot_random_heatmap(100, 64, mycolors[xc], "white", myseed[xc]), width = 6, height = 10)
}
