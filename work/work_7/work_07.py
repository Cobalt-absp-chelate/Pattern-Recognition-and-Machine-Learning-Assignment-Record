import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path(r"C:\Users\11411\Desktop\Python\work_7\ex2data1.txt")
image_dir = file_path.parent / "image"
image_dir.mkdir(exist_ok=True)

data = pd.read_csv(file_path, header=None, names=["x_1", "x_2", "y"])

def plot_data(data, save_path, theta=None, title="Logistic Regression"):
    positive = data[data["y"] == 1]
    negative = data[data["y"] == 0]

    x_min, x_max = 20, 105
    y_min, y_max = 20, 105

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(positive["x_1"], positive["x_2"], s=55, marker="+", linewidths=2, label="y = 1")
    plt.scatter(negative["x_1"], negative["x_2"], s=40, marker="o", alpha=0.8, label="y = 0")

    if theta is not None and abs(theta[2]) > 1e-10:
        x1 = np.linspace(x_min, x_max, 400)
        x2 = -(theta[0] + theta[1] * x1) / theta[2]
        mask = (x2 >= y_min) & (x2 <= y_max)
        if np.any(mask):
            plt.plot(x1[mask], x2[mask], linewidth=2.2, label="Decision Boundary")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_loss(loss_history, save_path, mark_iters):
    x = np.arange(len(loss_history))
    y = np.array(loss_history)

    step = max(1, len(x) // 3000)
    x_plot = x[::step]
    y_plot = y[::step]

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(x_plot, y_plot, linewidth=2)

    y_min, y_max = y.min(), y.max()
    y_span = y_max - y_min

    valid_iters = [it for it in mark_iters if 0 <= it < len(loss_history)]

    for it in valid_iters:
        loss_val = loss_history[it]
        plt.scatter(it, loss_val, s=40, zorder=3)
        plt.plot([0, it], [loss_val, loss_val], linestyle="--", linewidth=1.2)
        plt.plot([it, it], [y_min, loss_val], linestyle="--", linewidth=1.2)

    xticks = [0] + valid_iters
    yticks = [loss_history[it] for it in valid_iters]

    plt.xticks(xticks, [str(v) for v in xticks])
    plt.yticks(yticks, [f"{v:.4f}" for v in yticks])

    plt.xlim(-50000, len(loss_history) - 1 + 50000)
    plt.ylim(y_min - 0.02 * y_span, y_max + 0.05 * y_span)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def sigmoid(z):  # sigmoid函数
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):  # 损失函数
    h = sigmoid(X @ theta)
    eps = 1e-5
    return -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))

def gradient(theta, X, y):  # 梯度函数
    h = sigmoid(X @ theta)
    return (X.T @ (h - y)) / len(y)

def gradient_descent(X, y, theta, alpha, iterations):  # 手写梯度下降函数，发现预测准确率比scipy.optimize.minimize更高
    theta_history = [theta.copy()]
    loss_history = [cost(theta, X, y)]

    for _ in range(iterations):  # 迭代更新theta
        theta = theta - alpha * gradient(theta, X, y) 
        theta_history.append(theta.copy())
        loss_history.append(cost(theta, X, y))

    return theta, theta_history, loss_history

def predict(theta, X):  # 预测函数，返回0或1的预测结果
    probs = sigmoid(X @ theta)
    return (probs >= 0.5).astype(int)

X = data[["x_1", "x_2"]].values
y = data["y"].values
X = np.insert(X, 0, 1, axis=1)

theta_initial = np.zeros(X.shape[1])

alpha = 0.0002  # 学习率
iterations = 1000000  # 迭代次数，增加次数可以获得更好的预测准确率

theta_opt, theta_history, loss_history = gradient_descent(X, y, theta_initial, alpha, iterations)

predictions = predict(theta_opt, X)
accuracy = np.mean(predictions == y) * 100

print("最优权重参数：")
print(theta_opt)
print("最小代价：", loss_history[-1])
print("预测准确率：{:.2f}%".format(accuracy))

mid_iter_1 = 50000
mid_iter_2 = 250000

plot_data(data, image_dir / "01_data_distribution.png", title="Data Distribution")
plot_data(data, image_dir / f"02_decision_boundary_iter_{mid_iter_1}.png", theta_history[mid_iter_1], f"Decision Boundary at Iteration {mid_iter_1}")
plot_data(data, image_dir / f"03_decision_boundary_iter_{mid_iter_2}.png", theta_history[mid_iter_2], f"Decision Boundary at Iteration {mid_iter_2}")
plot_data(data, image_dir / "04_decision_boundary_final.png", theta_opt, "Final Decision Boundary")
plot_loss(loss_history, image_dir / "05_loss_curve.png", [mid_iter_1, mid_iter_2, iterations])

print("图片已保存到：", image_dir)