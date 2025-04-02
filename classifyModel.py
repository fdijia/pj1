import numpy as np
import os
import json

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        isFile = os.path.isfile('model.txt')
        # 初始化权重和偏置
        if isFile:
            with open('model.json', 'r') as f:
                model = json.load(f)
                self.W1 = np.array(model['W1'])
                self.b1 = np.array(model['b1'])
                self.W2 = np.array(model['W2'])
                self.b2 = np.array(model['b2'])
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation == 'relu':
            self.a1 = self.relu(self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate, reg_lambda):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差
        delta2 = self.a2 - y    
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        if self.activation == 'relu':
            delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        elif self.activation == 'sigmoid':
            delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # 正则化
        dW2 += reg_lambda * self.W2
        dW1 += reg_lambda * self.W1
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)