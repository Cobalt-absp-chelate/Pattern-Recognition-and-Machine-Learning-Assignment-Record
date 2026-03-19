import numpy as np
import matplotlib.pyplot as plt
import os

# 样本
x1 = np.array([0, 3, 1, 2, 0], dtype=float)
x2 = np.array([1, 3, 0, 1, 0], dtype=float)
x3 = np.array([3, 3, 0, 0, 1], dtype=float)
x4 = np.array([1, 1, 0, 2, 0], dtype=float)
x5 = np.array([3, 2, 1, 2, 1], dtype=float)
x6 = np.array([4, 1, 1, 1, 0], dtype=float)

X = np.array([x1, x2, x3, x4, x5, x6], dtype=float)
labels = ["x1", "x2", "x3", "x4", "x5", "x6"]

save_dir = r"C:\Users\11411\Desktop\Python\模式识别与机器学习\work\work_3"
os.makedirs(save_dir, exist_ok=True)

img_path = os.path.join(save_dir, "hierarchical_clustering_dendrogram_style.png")
txt_path = os.path.join(save_dir, "hierarchical_clustering_result.txt")

EPS = 1e-12

# 计算欧氏距离
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def cluster_distance(cluster_a, cluster_b, data):
    min_dist = float("inf")
    for i in cluster_a:
        for j in cluster_b:
            d = distance(data[i], data[j])
            if d < min_dist:
                min_dist = d
    return min_dist


# 给簇排序
def show_cluster(cluster, names):
    cluster = sorted(cluster)
    return "{" + ", ".join(names[i] for i in cluster) + "}"


def sort_clusters(cluster_list):
    return sorted([sorted(c) for c in cluster_list], key=lambda c: (c[0], len(c), c))


def agnes(data):
    n = data.shape[0]  # 得到样本数
    clusters = [{"id": i, "members": [i]} for i in range(n)]
    next_id = n
    merge_records = []

    while len(clusters) > 1:
        best_i = -1
        best_j = -1
        best_dist = float("inf")
        best_key = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = sorted(clusters[i]["members"])
                c2 = sorted(clusters[j]["members"])
                d = cluster_distance(c1, c2, data)

                pair_key = (tuple(c1), tuple(c2))
                if (d < best_dist - EPS) or (abs(d - best_dist) <= EPS and (best_key is None or pair_key < best_key)):
                    best_dist = d
                    best_i, best_j = i, j
                    best_key = pair_key

        # 合并簇
        c1 = clusters[best_i]
        c2 = clusters[best_j]

        left_members = sorted(c1["members"])
        right_members = sorted(c2["members"])
        merged_members = sorted(left_members + right_members)

        # 记录，作为分析与画图的数据来源
        merge_records.append({
            "step": len(merge_records) + 1,
            "left_id": c1["id"],
            "right_id": c2["id"],
            "new_id": next_id,
            "left_members": left_members[:],
            "right_members": right_members[:],
            "merged_members": merged_members[:],
            "distance": best_dist
        })

        # 删除被合并的簇，并把新的簇加入进去
        new_clusters = []
        for k in range(len(clusters)):
            if k != best_i and k != best_j:
                new_clusters.append(clusters[k])
        new_clusters.append({"id": next_id, "members": merged_members[:]})

        clusters = new_clusters
        next_id += 1

    return merge_records


# 按照阈值回放聚类结果，三个参数：样本个数，合并记录，阈值
def replay_clusters_under_threshold(n, merge_records, threshold):
    clusters = [[i] for i in range(n)]

    for record in merge_records:
        if record["distance"] <= threshold + EPS:  # 如果这一步合并的簇的距离不超过阈值
            left = sorted(record["left_members"])
            right = sorted(record["right_members"])

            new_clusters = []
            # 标记左右两个旧簇是否被删除
            removed_left = False
            removed_right = False

            for c in clusters:
                c_sorted = sorted(c)
                if c_sorted == left and not removed_left:
                    removed_left = True
                    continue
                elif c_sorted == right and not removed_right:
                    removed_right = True
                    continue
                else:
                    new_clusters.append(c_sorted)

            new_clusters.append(sorted(record["merged_members"]))
            clusters = sort_clusters(new_clusters)
        else:  # 如果大于阈值，停止合并
            break

    return sort_clusters(clusters)


def save_analysis_report(data, names, merge_records, out_path):
    # 四个阈值
    thresholds = [np.sqrt(3), np.sqrt(4), np.sqrt(5), np.sqrt(6)]

    lines = []
    lines.append("AGNES 最短距离层次聚类结果")
    lines.append("*" * 70)
    lines.append("样本：")
    for i in range(data.shape[0]):  # 写出所有样本
        lines.append(f"{names[i]} = {data[i].tolist()}")
    lines.append("")

    lines.append("每一步合并过程：")
    for record in merge_records:
        lines.append(
            f"第 {record['step']} 步："
            f"{show_cluster(record['left_members'], names)} 与 "
            f"{show_cluster(record['right_members'], names)} 合并，"
            f"距离 = {record['distance']:.6f}，"
            f"得到 {show_cluster(record['merged_members'], names)}"
        )

    # 处理阈值
    for t in thresholds:
        final_clusters = replay_clusters_under_threshold(len(names), merge_records, t)
        lines.append(f"阈值 = {t:.6f}")
        for idx, c in enumerate(final_clusters, 1):
            lines.append(f"  类 {idx}: {show_cluster(c, names)}")
        lines.append(f"  共 {len(final_clusters)} 类")
        lines.append("-" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# 绘制树状图
def draw_reference_style_dendrogram(data, out_path):

    y_display = {
        np.sqrt(3): 1.20,
        np.sqrt(4): 2.00,
        np.sqrt(5): 3.40,
        np.sqrt(6): 4.60
    }

    h12 = y_display[np.sqrt(3)]
    h56 = y_display[np.sqrt(4)]
    h124 = y_display[np.sqrt(5)]
    h_top = y_display[np.sqrt(6)]

    leaf_x = {
        "x1": 0.0,
        "x2": 1.0,
        "x4": 2.0,
        "x3": 3.0,
        "x5": 4.0,
        "x6": 5.0
    }

    node12_x = 0.50
    node124_x = 1.35
    node56_x = 4.35
    root_x = 3.00

    plt.figure(figsize=(8.6, 6.2), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    lw = 1.2
    c = "black"

    plt.plot([leaf_x["x1"], node12_x], [0, h12], color=c, linewidth=lw)
    plt.plot([leaf_x["x2"], node12_x], [0, h12], color=c, linewidth=lw)

    plt.plot([node12_x, node124_x], [h12, h124], color=c, linewidth=lw)
    plt.plot([leaf_x["x4"], node124_x], [0, h124], color=c, linewidth=lw)

    plt.plot([leaf_x["x5"], node56_x], [0, h56], color=c, linewidth=lw)
    plt.plot([leaf_x["x6"], node56_x], [0, h56], color=c, linewidth=lw)

    plt.plot([node124_x, root_x], [h124, h_top], color=c, linewidth=lw)
    plt.plot([leaf_x["x3"], root_x], [0, h_top], color=c, linewidth=lw)
    plt.plot([node56_x, root_x], [h56, h_top], color=c, linewidth=lw)

    y_levels = [y_display[np.sqrt(3)], y_display[np.sqrt(4)], y_display[np.sqrt(5)], y_display[np.sqrt(6)]]
    y_labels = [r"$\sqrt{3}$", r"$\sqrt{4}$", r"$\sqrt{5}$", r"$\sqrt{6}$"]

    for y in y_levels:
        plt.hlines(
            y, xmin=-0.10, xmax=5.80,
            colors="black",
            linestyles=(0, (4, 4)),
            linewidth=0.9
        )

    plt.xlim(-0.35, 5.85)
    plt.ylim(-0.25, 6.05)

    plt.xticks(
        [leaf_x["x1"], leaf_x["x2"], leaf_x["x4"], leaf_x["x3"], leaf_x["x5"], leaf_x["x6"]],
        [r"$x_1$", r"$x_2$", r"$x_4$", r"$x_3$", r"$x_5$", r"$x_6$"],
        fontsize=20
    )
    plt.yticks(y_levels, y_labels, fontsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.0)

    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["bottom"].set_position(("data", 0))

    ax.tick_params(axis="x", direction="out", length=4, width=0.8, pad=10)
    ax.tick_params(axis="y", direction="out", length=3, width=0.8, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()


def main():
    merge_records = agnes(X)
    save_analysis_report(X, labels, merge_records, txt_path)
    draw_reference_style_dendrogram(X, img_path)

    print("结果已保存：")
    print(txt_path)
    print(img_path)


if __name__ == "__main__":
    main()