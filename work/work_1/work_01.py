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
T_list = [0.1, 1, 4, 7, 10]  # 四个阈值

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


# 基于最近邻规则的聚类算法的函数
def nearest_neighbor(sample, T, first_center=0):
    store = VectorStore()
    n = len(sample)
    labels = [-1] * n  # 标记点是否已经分配类别
    current_class = 0  # 当前类别编号/最大类别编号
    store.add(sample[first_center], current_class)  # 把第一个聚类中心加入centers，令其类别为0
    labels[first_center] = current_class  # 把第一个聚类中心的点的标签设置为0，代表属于第0类
    for i in range(n):
        if labels[i] != -1:
            continue
        dists = store.distances(sample[i])
        dist_list = dists.tolist()
        min_dist = dist_list[0]
        min_pos = 0
        for j in range(1, len(dist_list)):
            if dist_list[j] < min_dist:
                min_dist = dist_list[j]
                min_pos = j
        nearest_class = store.center_ids[min_pos]

        if min_dist < T:
            labels[i] = nearest_class
        else:
            current_class += 1
            store.add(sample[i], current_class)
            labels[i] = current_class
    return labels, store.centers, store.center_ids


def plot(samples, labels, centers, T, first, title):
    import numpy as np
    import matplotlib.pyplot as plt

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

    plt.title(f'{title}, T={T}, {k}类')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='best', fontsize=8, frameon=True)
    plt.tight_layout()

    fname = f'result_T{T}_{title.replace("=", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

print("实验1: 第一个中心=x1")
for T in T_list:
    labels, centers, _ = nearest_neighbor(samples, T, 0)
    print(f"T={T}: 类别{labels}")
    plot(samples, labels, centers, T, first=0, title="中心=x1")
print("\n实验2: 不同中心,不同阈值")
for idx in [0, 1, 2,3,4,5,6,7,8,9]:
    for T in T_list:
        labels, centers, _ = nearest_neighbor(samples, T, idx)
        print(f"中心={names[idx]}: 类别{labels}")
        plot(samples, labels, centers, T, first=idx, title=f"中心={names[idx]}")

