import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None  # 斜率
        self.intercept_ = None  # 截距

    def fit(self, X, y):
        X = X.flatten()
        y = y.flatten()
        print("222", X, y)
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # 计算斜率和截距
        num = np.sum((X - x_mean) * (y - y_mean))
        den = np.sum((X - x_mean) ** 2)
        self.coef_ = num / den
        self.intercept_ = y_mean - self.coef_ * x_mean

    def predict(self, X):
        return self.coef_ * X + self.intercept_


# 使用示例
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
print("111", X, y)
model = SimpleLinearRegression()
model.fit(X, y)
print("斜率 coef_:", model.coef_)
print("截距 intercept_:", model.intercept_)

# 预测 X=6 时的 y
y_pred = model.predict(np.array([6]))
print("X=6 时的预测值:", y_pred[0])