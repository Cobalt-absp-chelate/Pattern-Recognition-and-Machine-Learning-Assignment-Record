import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x1 = np.array([0, 0])
x2 = np.array([3, 8])
x3 = np.array([2, 2])
x4 = np.array([1, 1])
x5 = np.array([5, 3])
x6 = np.array([4, 8])
x7 = np.array([6, 3])
x8 = np.array([5, 4])
x9 = np.array([6, 4])
x10 = np.array([7, 5])

samples = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]  # 十个样本
names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
theta_list = [0.2,0.5,0.7]

# 存储聚类中心的类
class VectorStore:
    def __init__(self):
        self.centers = []
        self.center_ids = []  # 类的编号

# 增加一个新的聚类中心及其类别编号
    def add(self, v, class_id):
        self.centers.append(v)
        self.center_ids.append(class_id)

# 计算给定向量到当前所有聚类中心的欧氏距离
    def distances(self, v):
        if len(self.centers) == 0:
            return np.array([])
        m = np.array(self.centers)
        return np.sqrt(np.sum((m - v) ** 2, axis=1))


# 基于最大最小距离聚类算法的函数
def Max_Min_Dist(sample, theta, first_center=0):
    store = VectorStore()
    n = len(sample)
    labels = [-1] * n  # 标记点是否已经分配类别
    current_class = 0  # 当前类别编号/最大类别编号
    store.add(sample[first_center], current_class)  # 把第一个聚类中心加入centers，令其类别为0
    labels[first_center] = current_class  # 把第一个聚类中心的点的标签设置为0，代表属于第0类
    # 使用np.linalg计算距离并返回最大值的索引
    second_center = max([i for i in range(n) if i != first_center], key=lambda i: np.linalg.norm(sample[i] - sample[first_center]))
    # 第一，二个聚类中心的距离
    D12 = np.linalg.norm(sample[second_center] - sample[first_center])
    # 阈值T
    T = D12 * theta
    current_class += 1
    store.add(sample[second_center], current_class)
    labels[second_center] = current_class
    # 循环为每一个样本点计算最大的最小值并与阈值T比较，判断是否为一个新的聚类中心
    while True:
        min_dist = []
        for i in range(n):
            if labels[i] != -1:
                min_dist.append(0)  # 已经是聚类中心的点距离设为0
                continue
            dist = store.distances(sample[i])
            min_dist.append(np.min(dist))

        max_min_dist = max(min_dist)
        p_id = np.argmax(min_dist)  # 得到最大的最小值的对应的点的索引
        # 与阈值进行比较
        if max_min_dist > T:
            current_class += 1
            store.add(sample[p_id], current_class)
            labels[p_id] = current_class
        else:
            break
    # 将非聚类中心的点归入距离最近的聚类中心的类中
    for i in range(n):
        if labels[i] != -1:
            continue
        dist = store.distances(sample[i])
        nearest_id = np.argmin(dist)  # 返回距离最近的类的编号的索引
        labels[i] = nearest_id

    return labels, store.centers, store.center_ids,T  # 返回标签，聚类中心，类的编号，阈值


def plot(samples, labels, centers, theta, T_actual, first, title):
    samples_np = np.array(samples)
    labels_np = np.array(labels)
    k = len(centers)
    plt.figure(figsize=(6.5, 5.5))
    cmap = plt.get_cmap('tab20')

    for cls in sorted(set(labels_np.tolist())):
        idx = np.where(labels_np == cls)[0]
        pts = samples_np[idx]
        plt.scatter(
            pts[:, 0], pts[:, 1],
            s=60,
            color=cmap(cls % 20),
            label=f'Class {cls}',
            alpha=0.85,
            edgecolors='k',
            linewidths=0.3
        )

    p0 = samples_np[first]
    plt.scatter(p0[0], p0[1], s=220, marker='s', color='none',
                edgecolors='k', linewidths=1.8, label='First center')

    for ci, c in enumerate(centers):
        plt.scatter(c[0], c[1], c='k', marker='*', s=180, zorder=5)
        plt.text(c[0] + 0.15, c[1] + 0.15, f'C{ci}', fontsize=9, color='k')
    plt.title(f'{title}, θ={theta}, T={T_actual:.2f}, {k}类')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='best', fontsize=8, frameon=True)
    plt.tight_layout()
    fname = f'result_theta{theta}_T{T_actual:.2f}_{title.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

print("实验1: 取（0，0）作为第一个聚类中心，阈值不同")
for theta in theta_list:
    labels, centers, center_ids, T_actual = Max_Min_Dist(samples, theta, 0)
    print(f"θ={theta}, 实际T={T_actual:.2f}: 类别{labels}")
    plot(samples, labels, centers, theta, T_actual, first=0, title="中心=x1")

print("\n实验2: 不同第一个中心，不同theta")

first_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9,]
for idx in first_indices:
    for theta in theta_list:
        labels, centers, center_ids, T_actual = Max_Min_Dist(samples, theta, first_center=idx)
        print(f"中心={names[idx]}, θ={theta}, 实际T={T_actual:.2f}: 类别{labels}")
        plot(samples, labels, centers, theta, T_actual, first=idx, title=f"中心={names[idx]}")

