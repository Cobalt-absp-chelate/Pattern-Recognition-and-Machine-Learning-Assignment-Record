import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def get_image_dir():
    save_dir = Path(__file__).resolve().parent / 'images'
    save_dir.mkdir(exist_ok=True)
    return save_dir


def save_plot(fig, file_name):
    save_path = get_image_dir() / file_name
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f'图片已保存: {save_path}')


class Perceptron:
    def __init__(self, lr=1.0, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.weight = None
        self.bias = None
        self.error_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape   # 样本的个数和维数
        self.weight = np.zeros(n_features)  # 样本为n维，权重就设为n维，这里和书上不同，把偏置单独拿出来方便计算
        self.bias = 0.0
        self.error_history = [] # 记录每轮迭代的错误数，方便后面做可视化展示

        for epoch in range(1, self.max_iter + 1):
            errors = 0
            for xi, target in zip(X, y):
                output = np.dot(xi, self.weight) + self.bias # 计算输出
                prediction = 1 if output >= 0 else -1

                if prediction != target: # 如果预测错误，更新权重和偏置
                    update = self.lr * target
                    self.weight += update * xi
                    self.bias += update
                    errors += 1

            self.error_history.append(errors)
            if errors == 0:
                print(f"在第{epoch}轮迭代后分类成功")
                break
        else:
            print(f"已达到最大迭代次数 {self.max_iter}，最后一轮仍有 {errors} 个错误")

        return epoch, errors, self.error_history

    def predict(self, X): # 预测函数
        X = np.atleast_2d(X)
        output = np.dot(X, self.weight) + self.bias
        return np.where(output >= 0, 1, -1)

    def decision_function(self, X): # 计算决策函数值，方便后面画分类面
        X = np.atleast_2d(X)
        return np.dot(X, self.weight) + self.bias


def plot_decision_boundary_experiment(model, X, y, test_points, lr):
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, marker, color in [(-1, 'o', '#4C72B0'), (1, 's', '#C44E52')]:
        mask = y == label
        ax.scatter(
            X[mask, 0], X[mask, 1],
            marker=marker,
            color=color,
            label=f'class {label}',
            edgecolors='k',
            s=70,
            alpha=0.9,
        )

    if test_points is not None:
        for name, pts in test_points.items():
            preds = model.predict(pts)
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=110,
                facecolors='none',
                edgecolors='#2E8B57',
                linewidths=2,
                label=f'{name} 测试点',
            )
            for pt, pred in zip(pts, preds):
                ax.text(pt[0] + 0.02, pt[1] + 0.02, f'{name}:{pred}', color='#2E8B57', fontsize=10)

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    if abs(model.weight[1]) < 1e-6:
        x_values = np.array([-(model.bias) / model.weight[0]] * 2)
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        ax.plot(x_values, [y_min, y_max], color='#333333', linestyle='--', linewidth=1.5, label='分类面')
    else:
        x_values = np.linspace(x_min, x_max, 200)
        y_values = -(model.weight[0] * x_values + model.bias) / model.weight[1]
        ax.plot(x_values, y_values, color='#333333', linestyle='--', linewidth=1.5, label='分类面')

    ax.set_xlabel('触角长度')
    ax.set_ylabel('翅膀长度')
    ax.set_title(f'学习率={lr} 的分类结果')
    ax.legend(frameon=False)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_facecolor('#F5F5F5')
    fig.tight_layout()
    save_plot(fig, f'decision_lr{str(lr).replace(".", "p")}.png')


def plot_training_errors_experiment(error_history, lr):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(1, len(error_history) + 1))
    ax.plot(x, error_history, marker='o', color='#7FB77E', linewidth=1.3, markersize=6, markerfacecolor='#E5F3E5', markeredgecolor='#4C8C4A')

    zero_epoch = next((i + 1 for i, e in enumerate(error_history) if e == 0), None)
    if zero_epoch is not None:
        ax.axvline(zero_epoch, color='gray', linestyle='--', linewidth=1)
        xticks = [int(tick) for tick in ax.get_xticks() if float(tick).is_integer()]
        if zero_epoch not in xticks:
            xticks.append(zero_epoch)
            xticks.sort()
            ax.set_xticks(xticks)

    ax.set_xlabel('迭代轮次')
    ax.set_ylabel('错误分类数')
    ax.set_title(f'学习率={lr} 的训练误差变化')
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#F9F9F9')
    fig.tight_layout()
    save_plot(fig, f'error_lr{str(lr).replace(".", "p")}.png')


def plot_experiment(model, X, y, error_history, test_points, lr):
    plot_decision_boundary_experiment(model, X, y, test_points, lr)
    plot_training_errors_experiment(error_history, lr)


def main():
    X = np.array([
        [1.24, 1.27],
        [1.36, 1.74],
        [1.38, 1.64],
        [1.38, 1.82],
        [1.38, 1.90],
        [1.40, 1.70],
        [1.48, 1.82],
        [1.54, 1.82],
        [1.56, 2.08],
        [1.14, 1.82],
        [1.18, 1.96],
        [1.20, 1.86],
        [1.26, 2.00],
        [1.28, 2.00],
        [1.30, 1.96],
    ])
    y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1])

    test_points = {'Apf': np.array([[1.24, 1.80], [1.28, 1.84]]), 'Af': np.array([[1.40, 2.04]])}
    learning_rates = [1.0, 0.1, 1.5]

    for lr in learning_rates:
        print(f'\n====== 学习率 = {lr} 实验 ======')
        model = Perceptron(lr=lr, max_iter=1000)
        epoch, errors, error_history = model.fit(X, y)

        print('训练得到的权重向量:', model.weight)
        print('训练得到的偏置项:', model.bias)

        predictions = model.predict(X)
        correct = np.sum(predictions == y)
        print(f'原始样本分类正确率: {correct}/{len(y)}')
        print('测试点 Apf 预测:', model.predict(test_points['Apf']))
        print('测试点 Af 预测:', model.predict(test_points['Af']))

        plot_experiment(model, X, y, error_history, test_points, lr)


if __name__ == '__main__':
    main()
