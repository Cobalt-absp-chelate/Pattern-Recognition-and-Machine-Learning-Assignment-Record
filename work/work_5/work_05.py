import numpy as np
import matplotlib.pyplot as plt

# 西瓜数据集4.0
data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]
])

# DBSCAN算法
def dbscan(X, eps, min_pts):
    n = X.shape[0]
    labels = np.full(n, -1)  # -1表示未分类
    cluster_id = 0

    def region_query(idx):
        dists = np.sqrt(np.sum((X - X[idx])**2, axis=1))
        return np.where(dists <= eps)[0] # 返回处在idx邻域内的点的索引

    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_pts:
            labels[i] = -1  
        else: # 是核心对象，开始扩展簇
            labels[i] = cluster_id
            Q = set(neighbors)
            Q.discard(i)
            while Q:
                j = Q.pop()
                if not visited[j]:
                    visited[j] = True
                    neighbors_j = region_query(j)
                    if len(neighbors_j) >= min_pts:
                        Q.update(neighbors_j) # 邻域点是核心对象，加入待处理队列
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels

# 绘制聚类结果
def plot_clusters(X, labels, title, save_path):
    plt.rcParams['font.sans-serif'] = [ 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8,6))
    unique_labels = set(labels)
    colors = plt.get_cmap('tab10', len(unique_labels))
    for k in unique_labels:
        if k == -1:
            plt.scatter(X[labels == k, 0], X[labels == k, 1], c='none', edgecolors='k', marker='o', s=80, linewidth=1.5, label='离散点')
        else:
            col = colors(k)
            plt.scatter(X[labels == k, 0], X[labels == k, 1], c=[col], marker='o', s=60, edgecolors='white', linewidth=0.5, label=f'簇 {k}')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# 主程序
if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "cluster_figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    # 选择不同的邻域参数
    param_list = [
        (0.11, 5),  
        (0.08, 5),  # 较小eps，更多簇
        (0.15, 5),  # 较大eps，更少簇
        (0.11, 3),  # 较小MinPts，更多簇
        (0.15, 3),  # 中等参数
    ]
    for eps, min_pts in param_list:
        labels = dbscan(data, eps, min_pts)
        save_path = os.path.join(fig_dir, f"dbscan_eps{eps}_minpts{min_pts}.png")
        plot_clusters(data, labels, f"DBSCAN 聚类结果 (ε={eps}, MinPts={min_pts})", save_path)
    