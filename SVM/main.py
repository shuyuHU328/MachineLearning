import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 生成SVM数据
    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2],
              np.random.randn(20, 2) + [2, 2]]
    Y = [-1] * 20 + [1] * 20
    model = svm.SVC(kernel="linear")
    model.fit(X, Y)
    # 输出支持向量
    print("support vectors: ", end='')
    print(model.support_vectors_)
    print(model.n_support_)
    support_vectors = model.support_vectors_
    # 绘制样本
    plt.scatter(list(X[:20, 0]), list(X[:20, 1]), label='Y = -1')
    plt.scatter(list(X[20:, 0]), list(X[20:, 1]), label='Y = 1')
    plt.scatter(list(support_vectors[:, 0]), list(support_vectors[:, 1]), label="support vector")
    # 绘制超平面
    weight = model.coef_[0]
    # 计算具体斜截式公式
    bias = model.intercept_[0]
    k = -weight[0] / weight[1]
    b = -bias / weight[1]
    xx = np.linspace(-5, 4, 10)
    y_support_vector = k * xx + b
    plt.plot(xx, y_support_vector, color='black')
    # 绘制支持向量对应直线
    b_0 = support_vectors[0][1] - k * support_vectors[0][0]
    b_1 = support_vectors[model.n_support_[0]][1] - k * support_vectors[model.n_support_[0]][0]
    y_0 = k * xx + b_0
    y_1 = k * xx + b_1
    plt.plot(xx, y_0, color='black', linestyle='--')
    plt.plot(xx, y_1, color='black', linestyle='--')
    plt.xlim(-5, 4)
    plt.legend()
    plt.show()
