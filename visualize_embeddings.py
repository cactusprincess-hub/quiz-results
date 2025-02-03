import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# 读取嵌入数据
embeddings_file1 = 'C:/Users/HP/Desktop/embeddings_output.txt'  # 第一个嵌入文件路径
embeddings_file2 = 'C:/Users/HP/Desktop/new_embeddings_output.txt'  # 第二个嵌入文件路径

# 加载嵌入数据
embeddings1 = np.loadtxt(embeddings_file1)
embeddings2 = np.loadtxt(embeddings_file2)

# 将两个嵌入合并为一个
embeddings = np.concatenate((embeddings1, embeddings2), axis=0)

# PCA降维到50维，减少t-SNE的计算压力
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(embeddings)

# 使用t-SNE进一步降维到2维
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_pca)

# 创建DataFrame方便管理
df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])

# 可视化
plt.figure(figsize=(10, 8))

# 为了增加可读性，可以使用不同的颜色标记来自两个不同文件的点
# 假设文件1的嵌入用红色，文件2的用蓝色
colors = ['red'] * len(embeddings1) + ['blue'] * len(embeddings2)
plt.scatter(df['x'], df['y'], c=colors, alpha=0.7, edgecolors='w', s=80)

# 优化图示样式
plt.title('2D Visualization of Embeddings (t-SNE)', fontsize=16)
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)

# 添加图例
plt.legend(['Embeddings from File 1', 'Embeddings from File 2'], loc='upper right', fontsize=12)

# 增加网格
plt.grid(True)

# 去掉坐标轴上的刻度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图形
plt.show()


