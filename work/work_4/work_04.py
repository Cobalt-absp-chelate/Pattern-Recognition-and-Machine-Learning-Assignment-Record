import os
import numpy as np
import matplotlib.pyplot as plt

# 保存输出图片
OUTPUT_DIR = "kmeans_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(filename):
    # 读取身高体重数据
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data.astype(float)


def distance(x, centers):
    # 计算每个样本到各聚类中心的平方距离
    n = x.shape[0]
    k = centers.shape[0]
    dist2 = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            diff = x[i] - centers[j]
            dist2[i, j] = np.sum(diff ** 2)

    return dist2


def initialize_centers(x, k, random_state=None):
    # 从样本中随机选取初始中心
    rng = np.random.default_rng(random_state)
    indices = rng.choice(x.shape[0], size=k, replace=False)
    return x[indices].copy()


def kmeans(x, k, init_centers=None, random_state=None, max_iter=100, tol=1e-6):
    # K-means 聚类过程
    n, d = x.shape

    if k <= 0 or k > n:
        raise ValueError("k 必须满足 1 <= k <= 样本总数")

    if init_centers is None:
        centers = initialize_centers(x, k, random_state=random_state)
    else:
        centers = np.array(init_centers, dtype=float)
        if centers.shape != (k, d):
            raise ValueError(f"init_centers 的形状应为 ({k}, {d})")

    labels = np.zeros(n, dtype=int)

    for it in range(max_iter):
        old_centers = centers.copy()

        # 按最近距离分配类别
        dist2 = distance(x, centers)
        labels = np.argmin(dist2, axis=1)

        # 更新聚类中心
        for i in range(k):
            cluster_points = x[labels == i]

            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
            else:
                rng = np.random.default_rng(random_state + it + i if random_state is not None else None)
                centers[i] = x[rng.integers(0, n)]

        # 判断是否收敛
        center_shift = np.sqrt(np.sum((centers - old_centers) ** 2))
        if center_shift < tol:
            break

    # 计算聚类指标 J
    final_dist2 = distance(x, centers)
    J = np.sum(final_dist2[np.arange(n), labels])

    return labels, centers, J, it + 1


def plot_clusters(x, labels, centers, gender_labels, title, save_name):
    # 绘制聚类结果图
    k = centers.shape[0]
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

    plt.figure(figsize=(8, 6))

    for i in range(k):
        # 当前类别中的女生样本
        female_points = x[(labels == i) & (gender_labels == 0)]
        if len(female_points) > 0:
            plt.scatter(
                female_points[:, 0],
                female_points[:, 1],
                c=colors[i % len(colors)],
                marker='o',
                label=f'Cluster {i + 1} - Female'
            )

        # 当前类别中的男生样本
        male_points = x[(labels == i) & (gender_labels == 1)]
        if len(male_points) > 0:
            plt.scatter(
                male_points[:, 0],
                male_points[:, 1],
                c=colors[i % len(colors)],
                marker='^',
                label=f'Cluster {i + 1} - Male'
            )

    # 绘制聚类中心
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c='black',
        marker='x',
        s=200,
        linewidths=3,
        label='Centers'
    )

    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # 图例放到图片下方
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False
    )

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图像: {save_path}")


def different_inits(x, gender_labels, dataset_name):
    # k=2 时测试不同初始值

    init1 = initialize_centers(x, 2, random_state=0)
    print(f'[{dataset_name}] k=2, 初值方案1选中的初始值:')
    print(init1)
    labels1, centers1, J1, iters1 = kmeans(x, k=2, init_centers=init1, random_state=0)
    print(f'[{dataset_name}] k=2, 初值方案1: J={J1:.4f}, 迭代次数={iters1}')
    print(f'最终聚类中心:\n{centers1}\n')
    plot_clusters(
        x, labels1, centers1, gender_labels,
        f'{dataset_name} - K=2 (Init 1), J={J1:.2f}',
        f'{dataset_name}_k2_init1.png'
    )

    init2 = initialize_centers(x, 2, random_state=42)
    print(f'[{dataset_name}] k=2, 初值方案2选中的初始值:')
    print(init2)
    labels2, centers2, J2, iters2 = kmeans(x, k=2, init_centers=init2, random_state=42)
    print(f'[{dataset_name}] k=2, 初值方案2: J={J2:.4f}, 迭代次数={iters2}')
    print(f'最终聚类中心:\n{centers2}\n')
    plot_clusters(
        x, labels2, centers2, gender_labels,
        f'{dataset_name} - K=2 (Init 2), J={J2:.2f}',
        f'{dataset_name}_k2_init2.png'
    )

    init3 = np.array([x[0], x[1]], dtype=float)
    print(f'[{dataset_name}] k=2, 初值方案3选中的初始值:')
    print(init3)
    labels3, centers3, J3, iters3 = kmeans(x, k=2, init_centers=init3, random_state=100)
    print(f'[{dataset_name}] k=2, 初值方案3(手动): J={J3:.4f}, 迭代次数={iters3}')
    print(f'最终聚类中心:\n{centers3}\n')
    plot_clusters(
        x, labels3, centers3, gender_labels,
        f'{dataset_name} - K=2 (Manual Init), J={J3:.2f}',
        f'{dataset_name}_k2_manual.png'
    )


def j_vs_k(x, gender_labels, dataset_name, k_values=(2, 3, 4, 5), trials=10):
    # 绘制 J 与类别数 k 的关系曲线
    best_J_list = []

    for k in k_values:
        best_J = None
        best_centers = None
        best_labels = None

        for seed in range(trials):
            labels, centers, J, _ = kmeans(x, k=k, random_state=seed)
            if (best_J is None) or (J < best_J):
                best_J = J
                best_centers = centers
                best_labels = labels

        best_J_list.append(best_J)
        print(f'[{dataset_name}] k={k}, 最优 J={best_J:.4f}')

        plot_clusters(
            x, best_labels, best_centers, gender_labels,
            f'{dataset_name} - K={k}, Best J={best_J:.2f}',
            f'{dataset_name}_k{k}_best.png'
        )

    k_array = np.array(k_values, dtype=int)
    best_J_array = np.array(best_J_list, dtype=float)

    # 计算 J 的下降量
    drop_array = np.zeros(len(k_array))
    for i in range(1, len(k_array)):
        drop_array[i] = best_J_array[i - 1] - best_J_array[i]

    # 简单找一个肘部候选点
    if len(k_array) >= 3:
        change_array = np.zeros(len(k_array))
        for i in range(2, len(k_array)):
            change_array[i] = drop_array[i - 1] - drop_array[i]
        elbow_index = np.argmax(change_array[2:]) + 2
        elbow_k = k_array[elbow_index]
        elbow_j = best_J_array[elbow_index]
    else:
        elbow_k = k_array[0]
        elbow_j = best_J_array[0]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 左轴画 J-k 曲线
    line1, = ax1.plot(k_array, best_J_array, marker='o', linewidth=2, label='J-K curve')
    elbow_point = ax1.scatter(elbow_k, elbow_j, s=100, marker='s', label=f'Elbow candidate: K={elbow_k}')
    elbow_line = ax1.axvline(elbow_k, linestyle='--', linewidth=1, label='Elbow reference')

    for i in range(len(k_array)):
        ax1.text(k_array[i], best_J_array[i] + 15, f'{best_J_array[i]:.1f}', ha='center', fontsize=9)

    ax1.set_xlabel('Number of clusters K')
    ax1.set_ylabel('J')
    ax1.set_title(f'{dataset_name} - J vs K')
    ax1.set_xticks(k_array)
    ax1.grid(True, axis='y')

    # 右轴画 J 的下降量
    ax2 = ax1.twinx()
    bar2 = ax2.bar(k_array, drop_array, alpha=0.25, width=0.35, label='Decrease of J')

    # 从 k=3 开始画下降趋势线，使用深绿色虚线
    line2 = None
    if len(k_array) >= 2:
        line2, = ax2.plot(
            k_array[1:],
            drop_array[1:],
            linestyle='--',
            linewidth=1.8,
            color='darkgreen',
            label='Decrease trend'
        )

    for i in range(1, len(k_array)):
        ax2.text(k_array[i], drop_array[i] + 5, f'{drop_array[i]:.1f}', ha='center', fontsize=8)

    ax2.set_ylabel('Decrease of J')
    ax2.set_xticks(k_array)

    # 所有图例统一放到图片下方
    handles = [line1, elbow_point, elbow_line, bar2]
    labels = ['J-K curve', f'Elbow candidate: K={elbow_k}', 'Elbow reference', 'Decrease of J']

    if line2 is not None:
        handles.append(line2)
        labels.append('Decrease trend')

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    save_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_J_vs_K.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图像: {save_path}")

    return k_array, best_J_array


def main():
    # 读取男女数据
    female_data = load_data('FEMALE.TXT')
    male_data = load_data('MALE.TXT')

    # 合并男女数据
    all_data = np.vstack((female_data, male_data))

    # 0 表示女生，1 表示男生
    gender_labels = np.array([0] * len(female_data) + [1] * len(male_data))

    print('FEMALE 数据形状:', female_data.shape)
    print('MALE 数据形状:', male_data.shape)
    print('ALL 数据形状:', all_data.shape)
    print()

    # 不同初值下的 k=2 聚类
    different_inits(all_data, gender_labels, 'ALL')

    # 不同类别数下的聚类指标曲线
    j_vs_k(all_data, gender_labels, 'ALL', k_values=(2, 3, 4, 5), trials=10)

    print("\n实验完成。")
    print(f"所有图片已保存到文件夹: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()