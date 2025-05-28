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


class MulticlassPerceptron:
    def __init__(self, n_classes, n_features, epochs=10):
        self.n_classes = n_classes
        self.epochs = epochs
        self.weights = np.zeros((n_classes, n_features))

    def train(self, X, y):
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            errors = 0

            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                # 计算每个类别的得分
                scores = np.dot(self.weights, x_i)

                # 找出得分最高的类别
                predicted_class = np.argmax(scores)

                # 如果预测错误，更新权重
                if predicted_class != y_i:
                    errors += 1
                    self.weights[y_i] += x_i  # 增加正确类别的权重
                    self.weights[predicted_class] -= x_i  # 减少错误类别的权重

            error_rate = errors / n_samples
            print(f"Epoch {epoch + 1}/{self.epochs}, Training Error: {error_rate:.4f}")

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            scores = np.dot(self.weights, X[i])
            pred = np.argmax(scores)
            predictions.append(pred)
        return np.array(predictions)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return 1 - accuracy  # 返回错误率


# 训练多类感知机
num_classes = 10
multi_perceptron = MulticlassPerceptron(num_classes, X_train_bias.shape[1], epochs=10)
multi_perceptron.train(X_train_bias, y_train)

# 评估模型
train_error_multi = multi_perceptron.evaluate(X_train_bias, y_train)
test_error_multi = multi_perceptron.evaluate(X_test_bias, y_test)

print(f"Final Training Error (Multiclass): {train_error_multi:.4f}")
print(f"Final Test Error (Multiclass): {test_error_multi:.4f}")