import os
import numpy as np

class ThreeLayerMLP:
    def __init__(self, input_dim=12288, h1=768, h2=384, output_dim=10, lr=0.002, weight_decay=1e-6):
        # He 初始化
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, output_dim) * np.sqrt(2 / h2)
        self.b3 = np.zeros((1, output_dim))

        self.lr = lr
        self.base_lr = lr
        self.weight_decay = weight_decay

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.prob = self.softmax(self.z3)
        return self.prob

    def backward(self, y_true):
        n = y_true.shape[0]
        y_onehot = np.zeros((n, 10))
        y_onehot[range(n), y_true] = 1

        # 输出层梯度
        dz3 = self.prob - y_onehot
        dW3 = self.a2.T @ dz3 / n + self.weight_decay * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True) / n

        # 第二层梯度
        dz2 = dz3 @ self.W3.T * (self.z2 > 0)
        dW2 = self.a1.T @ dz2 / n + self.weight_decay * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        # 第一层梯度
        dz1 = dz2 @ self.W2.T * (self.z1 > 0)
        dW1 = self.x.T @ dz1 / n + self.weight_decay * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        # 更新
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def cross_entropy_loss(self, y_true):
        n = len(y_true)
        log_prob = -np.log(np.clip(self.prob[range(n), y_true], 1e-8, 1-1e-8))
        ce = np.mean(log_prob)
        l2 = 0.5 * self.weight_decay * (np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
        return ce + l2

    def predict(self, x):
        p = self.forward(x)
        return np.argmax(p, axis=1)

    # 余弦学习率衰减（现在已正确放入类中）
    def update_lr_cosine(self, epoch, total_epoch):
        self.lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epoch))
