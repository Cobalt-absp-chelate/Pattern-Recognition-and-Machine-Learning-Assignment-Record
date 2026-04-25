import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = r"C:\Users\11411\Desktop\Python\Pattern-Recognition-and-Machine-Learning-Assignment-Record\work\work_8\image"
os.makedirs(save_dir, exist_ok=True)

# 样本一和样本二的数据，分别为10个三维样本点
w1 = np.array([
    [-0.4,   0.58,  0.089],
    [-0.31,  0.27, -0.04],
    [-0.38,  0.055, -0.035],
    [-0.15,  0.53,  0.011],
    [-0.35,  0.47,  0.034],
    [-0.17,  0.69,  0.1],
    [-0.011, 0.55, -0.18],
    [-0.27,  0.61,  0.12],
    [-0.065, 0.49,  0.0012],
    [-0.12,  0.054, -0.063]
], dtype=float)

w2 = np.array([
    [0.8,   1.6,  -0.014],
    [1.1,   1.6,   0.48],
    [-0.44, -0.41, 0.32],
    [0.047, -0.45, 1.4],
    [0.28,  0.35,  3.1],
    [-0.39, -0.48, 0.11],
    [0.34, -0.079, 0.14],
    [-0.3,  -0.22, 2.2],
    [1.1,   1.2,  -0.46],
    [0.18, -0.11, -0.49]
], dtype=float)

#  需要分类的两个样本点
xx1 = np.array([-0.7, 0.58, 0.089], dtype=float)
xx2 = np.array([0.047, -0.4, 1.04], dtype=float)

#  计算类内散度矩阵和类间散度矩阵，并求解Fisher线性判别的投影向量w
def mean_w(x):
    return np.mean(x, axis=0)

def scatter(X, m):
    S = np.zeros((X.shape[1], X.shape[1]))
    for i in X:
        diff = (i - m).reshape(-1, 1)
        S += diff @ diff.T
    return S

#  计算Fisher线性判别的投影向量w，并返回相关的统计量
def fisher_calculate_w(w1, w2):
    m1 = mean_w(w1)
    m2 = mean_w(w2)
    S1 = scatter(w1, m1)
    S2 = scatter(w2, m2)
    Sw = S1 + S2
    w = np.linalg.solve(Sw, (m1 - m2))
    w = w / np.linalg.norm(w)
    return w, m1, m2, S1, S2, Sw

def project_samples(X, w):
    return X @ w

#  根据Fisher线性判别的投影结果，对新的样本点进行分类
def classify_sample(x, w, m1, m2):
    y = x @ w
    y1 = m1 @ w
    y2 = m2 @ w
    if abs(y - y1) < abs(y - y2):
        return "w1"
    else:
        return "w2"

#  构建投影平面的基底向量，并将样本点投影到该平面上
def build_projection_basis(w):
    e1 = w / np.linalg.norm(w)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, e1)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e2 = ref - np.dot(ref, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2

def project_to_view(X, center, e1, e2):
    Xc = X - center
    u = Xc @ e1
    v = Xc @ e2
    return u, v

#  绘制原始样本点的三维分布和Fisher线性判别的投影结果
def plot_original_samples(w1, w2, m1, m2):
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(w1[:, 0], w1[:, 1], w1[:, 2], s=55, alpha=0.9, color='royalblue', label='w1', edgecolors='white', linewidths=0.7)
    ax.scatter(w2[:, 0], w2[:, 1], w2[:, 2], s=55, alpha=0.9, color='red', label='w2', edgecolors='white', linewidths=0.7)

    ax.scatter(m1[0], m1[1], m1[2], marker='X', s=180, color='navy', label='m1', edgecolors='black', linewidths=0.8)
    ax.scatter(m2[0], m2[1], m2[2], marker='X', s=180, color='darkred', label='m2', edgecolors='black', linewidths=0.8)

    ax.set_title('Original Sample Distribution', pad=14)
    ax.set_xlabel('x1', labelpad=8)
    ax.set_ylabel('x2', labelpad=8)
    ax.set_zlabel('x3', labelpad=8)
    ax.view_init(elev=22, azim=-55)
    ax.grid(alpha=0.25)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "original_sample_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fisher_result(w1, w2, w, m1, m2, xx1=None, xx2=None):
    center = (m1 + m2) / 2
    e1, e2 = build_projection_basis(w) #  构建投影平面的基底向量

    u1, v1 = project_to_view(w1, center, e1, e2)
    u2, v2 = project_to_view(w2, center, e1, e2)

    um1, vm1 = project_to_view(m1.reshape(1, -1), center, e1, e2)
    um2, vm2 = project_to_view(m2.reshape(1, -1), center, e1, e2)
    um1, vm1 = um1[0], vm1[0]
    um2, vm2 = um2[0], vm2[0]

    threshold = (um1 + um2) / 2 #  计算Fisher判别的阈值

    if xx1 is not None:
        uxx1, vxx1 = project_to_view(xx1.reshape(1, -1), center, e1, e2)
        uxx1, vxx1 = uxx1[0], vxx1[0]

    if xx2 is not None:
        uxx2, vxx2 = project_to_view(xx2.reshape(1, -1), center, e1, e2)
        uxx2, vxx2 = uxx2[0], vxx2[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.4, 1]})

    ax = axes[0]
    ax.scatter(u1, v1, s=42, alpha=0.92, color='royalblue', label='w1', edgecolors='white', linewidths=0.5)
    ax.scatter(u2, v2, s=42, alpha=0.92, color='red', label='w2', edgecolors='white', linewidths=0.5)

    for u, v in zip(u1, v1): #  绘制每个样本点到Fisher轴的垂线
        ax.plot([u, u], [0, v], color='royalblue', alpha=0.18, linewidth=0.8)
    for u, v in zip(u2, v2):
        ax.plot([u, u], [0, v], color='red', alpha=0.18, linewidth=0.8)

    ax.scatter(u1, np.zeros_like(u1), marker='s', s=18, color='royalblue', alpha=0.85)
    ax.scatter(u2, np.zeros_like(u2), marker='s', s=18, color='red', alpha=0.85)

    ax.scatter(um1, vm1, marker='X', s=150, color='navy', label='m1', edgecolors='black', linewidths=0.7)
    ax.scatter(um2, vm2, marker='X', s=150, color='darkred', label='m2', edgecolors='black', linewidths=0.7)

    ax.axhline(0, linestyle='--', linewidth=1.3, color='gray', label='Fisher axis')
    ax.axvline(threshold, linestyle='-.', linewidth=1.3, color='black', label='threshold')

    if xx1 is not None:
        ax.scatter(uxx1, vxx1, marker='*', s=220, color='gold', label='xx1', edgecolors='black', linewidths=0.7)
        ax.plot([uxx1, uxx1], [0, vxx1], linestyle=':', linewidth=1.0, color='goldenrod')

    if xx2 is not None:
        ax.scatter(uxx2, vxx2, marker='*', s=220, color='limegreen', label='xx2', edgecolors='black', linewidths=0.7)
        ax.plot([uxx2, uxx2], [0, vxx2], linestyle=':', linewidth=1.0, color='green')

    ax.set_title('Fisher Projection View')
    ax.set_xlabel('Along Fisher direction')
    ax.set_ylabel('Orthogonal direction')
    ax.grid(alpha=0.2)
    ax.legend(loc='best', fontsize=9)

    ax2 = axes[1] #  绘制Fisher轴上的投影分布直方图
    all_u = np.concatenate([u1, u2])
    bins = np.linspace(all_u.min() - 0.3, all_u.max() + 0.3, 12)

    ax2.hist(u1, bins=bins, alpha=0.75, color='royalblue', label='w1')
    ax2.hist(u2, bins=bins, alpha=0.75, color='red', label='w2')

    ax2.axvline(um1, linestyle='--', linewidth=1.2, color='navy', label='m1 proj')
    ax2.axvline(um2, linestyle='--', linewidth=1.2, color='darkred', label='m2 proj')
    ax2.axvline(threshold, linestyle='-.', linewidth=1.4, color='black', label='threshold')

    ymax = ax2.get_ylim()[1]

    if xx1 is not None:
        ax2.scatter(uxx1, ymax * 0.92, marker='*', s=180, color='gold', label='xx1', edgecolors='black', linewidths=0.7)

    if xx2 is not None:
        ax2.scatter(uxx2, ymax * 0.82, marker='*', s=180, color='limegreen', label='xx2', edgecolors='black', linewidths=0.7)

    ax2.set_title('1D Distribution on Fisher Direction')
    ax2.set_xlabel('Projection value')
    ax2.set_ylabel('Count')
    ax2.grid(alpha=0.2)
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fisher_classification_result.png"), dpi=300, bbox_inches='tight')
    plt.close()

w, m1, m2, S1, S2, Sw = fisher_calculate_w(w1, w2) #  计算Fisher线性判别的投影向量w，并获取相关统计量

print("m1 =\n", m1)
print("m2 =\n", m2)
print("S1 =\n", S1)
print("S2 =\n", S2)
print("Sw =\n", Sw)
print("w =\n", w)

#  根据Fisher线性判别的投影结果，对新的样本点进行分类
result_xx1 = classify_sample(xx1, w, m1, m2)
result_xx2 = classify_sample(xx2, w, m1, m2)

print("xx1 class =", result_xx1)
print("xx2 class =", result_xx2)

plot_original_samples(w1, w2, m1, m2)
plot_fisher_result(w1, w2, w, m1, m2, xx1, xx2)

print("Images saved to:", save_dir)