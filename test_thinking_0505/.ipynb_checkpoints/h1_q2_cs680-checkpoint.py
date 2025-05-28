import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载MNIST数据集
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X.astype(float)
y = y.astype(int)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为numpy数组以避免索引问题
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 添加偏置项
X_train_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))


class BinaryPerceptron:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

    def train(self, X, y, epochs=10):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            errors = 0

            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                if y_i * np.dot(self.weights, x_i) <= 0:  # 如果分类错误
                    self.weights += y_i * x_i
                    errors += 1

            error_rate = errors / n_samples
            if epoch == epochs - 1:
                print(f"Final epoch error: {error_rate:.4f}")


class OneVsAllPerceptron:
    def __init__(self, n_classes, n_features, epochs=10):
        self.n_classes = n_classes
        self.epochs = epochs
        self.classifiers = [BinaryPerceptron(n_features) for _ in range(n_classes)]

    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            for k in range(self.n_classes):
                # 为当前类别创建二元标签
                binary_y = np.where(y == k, 1, -1)

                # 训练当前类别的二元感知机
                self.classifiers[k].train(X, binary_y, epochs=1)

    def predict(self, X):
        # 对每个样本，计算所有类别的分数
        scores = np.zeros((X.shape[0], self.n_classes))

        for k in range(self.n_classes):
            # 计算第k个分类器对每个样本的分数
            scores[:, k] = np.dot(X, self.classifiers[k].weights)

        # 返回得分最高的类别
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return 1 - accuracy  # 返回错误率

# 训练One vs. All感知机
num_classes = 10  # MNIST有10个类别
perceptron = OneVsAllPerceptron(num_classes, X_train_bias.shape[1], epochs=10)
perceptron.train(X_train_bias, y_train)

# 评估模型
train_error = perceptron.evaluate(X_train_bias, y_train)
test_error = perceptron.evaluate(X_test_bias, y_test)

print(f"Final Training Error: {train_error:.4f}")
print(f"Final Test Error: {test_error:.4f}")