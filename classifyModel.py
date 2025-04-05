import numpy as np
import os

class NeuralNetwork:
    """
    这是一个输入层-自定义隐藏层层数及大小-输出层的神经网络
    layer_sizes: 层, 是一个列表, 如果没有指定, 默认是[3072, 200, 60, 10]
    activations: 激活函数, 可以是一个字符串, 也可以是一个列表, 如果没有指定, 默认是都是'relu;, 层数与layer_sizes对应
    model_name: 模型名称, 默认是'model', 训练好的模型会根据名称加上layer_sizes保存, 例如model3072_200_60_10.npz
    如果给了函数明则各参数由函数名决定
    """
    def __init__(self, layer_sizes=[3072, 256, 64, 10], activations='relu', learning_rate=0.01, reg_lambda=0.01, model_name=None):
        if model_name:
            dict = decode_model_name(model_name)
            self.layer_sizes = dict['layer_sizes']
            self.activations = dict['activations']
            self.name = model_name
            self.reg_lambda = dict['reg_lambda']
            self.layers = len(self.layer_sizes) - 1
            self.learning_rate = dict['learning_rate']
        else:
            self.layer_sizes = layer_sizes
            self.layers = len(self.layer_sizes) - 1
            self.activations = activations if isinstance(activations, list) else [activations] * (self.layers - 1) + ['softmax']
            self.reg_lambda = reg_lambda
            self.learning_rate = learning_rate
            self.name = encode_model_name(self.layer_sizes, self.activations, self.learning_rate, self.reg_lambda)
        
        isFile = os.path.isfile(self.name)
        # 初始化权重和偏置
        self.W = []
        self.b = []
        if isFile:
            data = np.load(self.name)
            for i in range(self.layers):
                self.W.append(data['W' + str(i+1)])
                self.b.append(data['b' + str(i+1)])
        else:
            for i in range(self.layers):
                W, b = self.generate_W_b(self.layer_sizes[i], self.layer_sizes[i + 1])
                self.W.append(W)
                self.b.append(b)
            
    def generate_W_b(self, size1, size2):
            W = np.random.randn(size1, size2) * np.sqrt(1. / size1)
            b = np.zeros((1, size2))
            return W, b

    def forward(self, X):
        self.z = []
        self.a = []
        for i in range(self.layers):
            if i == 0:
                self.z.append(np.dot(X, self.W[i]) + self.b[i])
                self.a.append(self.activate(self.z[i], self.activations[i]))
            else:
                self.z.append(np.dot(self.a[i - 1], self.W[i]) + self.b[i])
                self.a.append(self.activate(self.z[i], self.activations[i]))
        return self.a[-1]
    
    def activate(self, z, activation):
        if activation == 'relu':
            return self.relu(z)
        if activation == 'sigmoid':
            return self.sigmoid(z)
        if activation == 'softmax':
            return self.softmax(z)


    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        deltas = [0 for i in range(self.layers)]
        for i in reversed(range(self.layers)):
            if i == self.layers - 1:
                delta = self.a[i] - y
            else:
                delta = self.derivate(deltas[i + 1], self.W[i + 1], self.z[i], self.activations[i])
            deltas[i] = delta

        for i in range(self.layers):
            if i == 0:
                dW = np.dot(X.T, deltas[i]) / m + self.reg_lambda * self.W[i]
            else:
                dW = np.dot(self.a[i - 1].T, deltas[i]) / m + self.reg_lambda * self.W[i]
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            self.W[i] -= learning_rate * dW
            self.b[i] -= learning_rate * db
    
    def derivate(self, delta, W, z, activation):
        if activation == 'relu':
            derivative = self.relu_derivative(z)
        elif activation == 'sigmoid':
            derivative = self.sigmoid_derivative(z)
        return np.dot(delta, W.T) * derivative
        

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
    
    def evaluate_model(self, X, y):
        predictions = self.forward(X)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy
    def compute_loss(self, y_pred, y_true):
            m = y_true.shape[0]
            # 交叉熵损失
            log_likelihood = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)])
            data_loss = np.sum(log_likelihood) / m
            # L2正则化损失
            reg_loss = 0.5 * self.reg_lambda * (sum(np.sum(w**2) for w in self.W))
            return data_loss + reg_loss
    

def encode_model_name(layer_sizes, activations, learning_rate, reg_lambda):
    """
    生成模型文件名
    参数:
        layer_sizes: 各层尺寸列表 (list), 如 [3072, 1024, 512, 128, 32, 10]
        activations: 各层激活函数列表 (list), 如 ['relu', 'relu', 'sigmoid', 'relu', 'softmax']
        learning_rate: 学习率 (float)
        reg_lambda: 正则化系数 (float)
    返回:
        模型名称 (str)
    """
    # 检查输入合法性
    assert len(layer_sizes) - 1 == len(activations), "不合法"
    
    name_parts = [f"model"] # model
    for size, act in zip(layer_sizes[:-2], activations[:-1]):
        name_parts.append(f"{size}{act[0]}")  # model3072r1024r512s128r
    name_parts.append(f"{layer_sizes[-2]}_{layer_sizes[-1]}")  # model3072r1024r512s128r32_10
    
    # 组合超参数部分 (如 "learning_rate0.01rl0.1")
    name_parts.append(f"lr{learning_rate:.4f}".rstrip('0').rstrip('.'))  # model3072r1024r512s128r32_10lr0.01
    name_parts.append(f"rl{reg_lambda:.4f}".rstrip('0').rstrip('.')) # model3072r1024r512s128r32_10lr0.01rl0.1
    
    return f"{''.join(name_parts)}.npz" # model3072r1024r512s128r32_10lr0.01rl0.1.npz


def decode_model_name(filename):
    """
    从文件名提取模型参数
    参数:
        filename: 模型文件名 (str)
    返回:
        dict: {
            'layer_sizes': list,
            'activations': list,
            'learning_rate': float,
            'reg_lambda': float
        }
    """
    name = filename.replace("model", "").replace(".npz", "")
    # learning_rate and reg_lambda
    parts = name.split("lr")
    learning_rate_reg_part = parts[1].split("rl")
    learning_rate = float(learning_rate_reg_part[0])
    reg_lambda = float(learning_rate_reg_part[1])
    
    # 解析层结构
    layer_part = parts[0]
    segments = [] # ['3072', '1024', '512', '128', '32', '10']
    current = ""
    act = []
    for c in layer_part:
        if c.isdigit():
            current += c
        else:
            if current:
                segments.append(current)
                current = ""
            act.append(c)
    if current:
        segments.append(current)
    
    # 提取层尺寸和激活函数
    layer_sizes = [int(s) for s in segments]
    activations = []
    for f in act:
        if f == 'r':
            activations.append('relu')
        elif f == 's':
            activations.append('sigmoid')
        elif f == '_':
            activations.append('softmax')
    
    return {
        'layer_sizes': layer_sizes,
        'activations': activations,
        'learning_rate': learning_rate,
        'reg_lambda': reg_lambda
    }